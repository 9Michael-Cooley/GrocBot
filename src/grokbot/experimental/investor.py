"""Investor — portfolio management and trading. UNRELEASED, no target.

Gives a bot a portfolio, a market feed, a risk envelope, and a broker, then lets
it reason about all four through tools. The model does not move money directly:
it *proposes* trades with a written rationale, risk checks the proposal, and a
human approves it. The gap between "the model wants to buy" and "the order left
the building" is the entire safety story of this module.

    from grokbot.experimental.investor import TradingSession, InvestorBot

    session = TradingSession.paper(["AAPL", "BTC-USD"], cash=100_000)
    bot = InvestorBot(engine, session)

    print(bot.brief())                       # market + portfolio state
    proposals = bot.propose("rebalance toward the stronger trend")
    session.approve(proposals[0].id)         # human in the loop
    print(session.report())

Three autonomy modes, and the default is the paranoid one:

    advisory   the model may only talk. Proposals are never executable.
    confirm    the model proposes; a human calls approve(). DEFAULT.
    auto       proposals execute immediately. Paper broker only, always.

The broker is `PaperBroker` unless you go out of your way. There is no live
broker adapter in this tree and `LiveBroker` refuses to construct — see the
STATUS notes before you write one.

STATUS — this is a simulator, not a trading system:
  - `SyntheticFeed` is the only feed. It is a seeded geometric random walk with
    a plausible-looking drift and vol per symbol. It is to prices what
    SyntheticBackend is to tokens: deterministic garbage that exercises the full
    path. Do not evaluate a strategy on it. Do not screenshot its P&L.
  - No persistence. The book dies with the process — same open problem as pets
    (GROK-4611). Rebuilding positions from the journal on restart is not written.
  - No corporate actions, dividends, borrow, funding rates, margin interest, or
    taxes. Crypto and equities are the same object with different vol and hours,
    which is wrong in ways that matter.
  - Fills are optimistic: full size at the touch plus fixed slippage, no
    partials, no queue position, no impact. Anything that would move a real book
    looks better here than it is.
  - Risk limits are pre-trade only. There is no intraday monitor, so a limit
    breached by a *price move* rather than by an order goes unnoticed until the
    next proposal.
  - Safety has not reviewed the investor persona. It is written to refuse advice
    framing and to state uncertainty, but nobody has red-teamed it.
  - Nothing here is investment advice, and a model wired to `auto` against a real
    broker is a way to lose real money in a novel fashion. That is not a
    hypothetical caveat; it is why the live path is unimplemented.
"""

from __future__ import annotations

import math
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal

from ..agent.persona import Persona
from ..errors import GrokBotError
from ..tools.registry import ToolRegistry, tool
from ..utils.logging import get_logger
from ..utils.rng import Rng, seed_from_text

log = get_logger(__name__)

__all__ = [
    "ENABLE_FLAG",
    "LIVE_FLAG",
    "Fill",
    "Instrument",
    "InvestorBot",
    "InvestorDisabled",
    "Order",
    "PaperBroker",
    "Portfolio",
    "Position",
    "Proposal",
    "Quote",
    "RiskLimits",
    "RiskViolation",
    "SyntheticFeed",
    "TradingSession",
    "investor_enabled",
    "investor_persona",
    "market_is_open",
]

ENABLE_FLAG = "GROKBOT_ENABLE_INVESTOR"
LIVE_FLAG = "GROKBOT_INVESTOR_ALLOW_LIVE"

_TRUTHY = ("1", "true", "yes", "on")


def investor_enabled() -> bool:
    return os.environ.get(ENABLE_FLAG, "").lower() in _TRUTHY


class InvestorDisabled(RuntimeError):
    pass


class RiskViolation(GrokBotError):
    """An order breached the risk envelope.

    Never swallowed internally: if a proposal violates, it stays a proposal.
    """

    status_code = 400

    def __init__(self, message: str, *, reasons: list[str] | None = None):
        super().__init__(message)
        self.reasons = reasons or []


def _require_enabled() -> None:
    if not investor_enabled():
        raise InvestorDisabled(
            f"investor is unreleased and disabled by default; set {ENABLE_FLAG}=1 to "
            f"use it. It is a paper simulator — read the STATUS block first."
        )


DISCLAIMER = (
    "Simulated portfolio on synthetic prices. Not investment advice, not a "
    "recommendation, and not a claim about any real security."
)


# --------------------------------------------------------------------------
# instruments
# --------------------------------------------------------------------------

AssetClass = Literal["equity", "crypto"]
Side = Literal["buy", "sell"]
OrderType = Literal["market", "limit"]
Autonomy = Literal["advisory", "confirm", "auto"]


@dataclass(frozen=True)
class Instrument:
    symbol: str
    asset_class: AssetClass
    name: str
    reference_price: float          # anchor for the synthetic walk
    annual_vol: float               # lognormal sigma
    annual_drift: float
    lot_size: float = 1.0           # whole shares; crypto is fractional
    quote_currency: str = "USD"

    @property
    def is_crypto(self) -> bool:
        return self.asset_class == "crypto"

    def round_qty(self, qty: float) -> float:
        """Snap to a tradeable size. Always down — rounding up an order size is
        how you discover you cannot afford it at the risk check."""
        if self.lot_size <= 0:
            return qty
        lots = math.floor(abs(qty) / self.lot_size + 1e-9)
        return math.copysign(lots * self.lot_size, qty)


# A deliberately small universe. This is a demo surface, not a security master;
# anything real belongs in the caller's own instrument table.
UNIVERSE: dict[str, Instrument] = {
    i.symbol: i
    for i in (
        Instrument("AAPL", "equity", "Apple Inc.", 190.0, 0.26, 0.09),
        Instrument("MSFT", "equity", "Microsoft Corp.", 410.0, 0.24, 0.11),
        Instrument("NVDA", "equity", "NVIDIA Corp.", 118.0, 0.52, 0.18),
        Instrument("SPY", "equity", "S&P 500 ETF", 540.0, 0.15, 0.07),
        Instrument("TSLA", "equity", "Tesla Inc.", 240.0, 0.55, 0.04),
        Instrument("BTC-USD", "crypto", "Bitcoin", 62000.0, 0.60, 0.15, lot_size=1e-6),
        Instrument("ETH-USD", "crypto", "Ether", 3100.0, 0.70, 0.12, lot_size=1e-5),
        Instrument("SOL-USD", "crypto", "Solana", 145.0, 0.95, 0.10, lot_size=1e-4),
    )
}


def lookup(symbol: str) -> Instrument:
    inst = UNIVERSE.get(symbol.upper().strip())
    if inst is None:
        raise RiskViolation(
            f"unknown symbol {symbol!r}; universe: {', '.join(sorted(UNIVERSE))}"
        )
    return inst


# US cash equities, in UTC. Half-days and holidays are not modelled.
_EQUITY_OPEN_MIN = 13 * 60 + 30
_EQUITY_CLOSE_MIN = 20 * 60


def market_is_open(inst: Instrument, now: datetime | None = None) -> bool:
    if inst.is_crypto:
        return True
    now = now or datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    return _EQUITY_OPEN_MIN <= now.hour * 60 + now.minute < _EQUITY_CLOSE_MIN


# --------------------------------------------------------------------------
# market data
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Quote:
    symbol: str
    bid: float
    ask: float
    ts: datetime

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_bps(self) -> float:
        return (self.ask - self.bid) / self.mid * 10_000.0 if self.mid else 0.0

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "bid": round(self.bid, 4),
            "ask": round(self.ask, 4),
            "mid": round(self.mid, 4),
            "spread_bps": round(self.spread_bps, 2),
            "ts": self.ts.isoformat(timespec="seconds"),
        }


_ANCHOR = datetime(2024, 1, 2, tzinfo=timezone.utc)
_TRADING_DAYS = 252
_CALENDAR_DAYS = 365


class SyntheticFeed:
    """Deterministic price paths. Same seed + same symbol => same history.

    A lognormal walk anchored at the instrument's reference price, stepped once
    per calendar day since 2024-01-02, plus an intraday wobble derived from the
    hour so a quote taken twice in a day is not identical.

    This is the price-side equivalent of SyntheticBackend: it makes the whole
    path runnable offline and it tells you nothing about the world.
    """

    def __init__(self, seed: int = 0, universe: dict[str, Instrument] | None = None):
        self.seed = seed
        self.universe = universe or UNIVERSE
        self._paths: dict[str, list[float]] = {}

    # -- path construction -------------------------------------------------

    @staticmethod
    def _day_index(now: datetime) -> int:
        return max(0, (now - _ANCHOR).days)

    def _path(self, inst: Instrument, days: int) -> list[float]:
        """Closes from the anchor through `days`, inclusive.

        Cached and regrown rather than continued — the rng is reseeded from the
        symbol each time, so a regrow reproduces the same prefix exactly. A
        carried rng would make history depend on call order, which made two
        sessions on the same seed disagree about the past.
        """
        cached = self._paths.get(inst.symbol)
        if cached is not None and len(cached) > days:
            return cached[: days + 1]

        rng = Rng(seed_from_text(f"{self.seed}:{inst.symbol}"))
        # Crypto steps every calendar day, equities only on weekdays, so the
        # per-step scaling has to use each one's own year length or crypto ends
        # up with 365 steps' worth of variance from a 252-step parameter.
        steps = _CALENDAR_DAYS if inst.is_crypto else _TRADING_DAYS
        step_vol = inst.annual_vol / math.sqrt(steps)
        step_drift = inst.annual_drift / steps - 0.5 * step_vol**2

        price = inst.reference_price
        out = [price]
        for d in range(1, days + 1):
            # Equities do not print on weekends; the close just repeats.
            if not inst.is_crypto and (_ANCHOR + timedelta(days=d)).weekday() >= 5:
                out.append(price)
                continue
            price = max(0.01, price * math.exp(step_drift + rng.gauss(0.0, step_vol)))
            out.append(price)
        self._paths[inst.symbol] = out
        return out

    # -- public ------------------------------------------------------------

    def closes(self, symbol: str, days: int = 30, now: datetime | None = None) -> list[float]:
        inst = lookup(symbol)
        now = now or datetime.now(timezone.utc)
        path = self._path(inst, self._day_index(now))
        return [round(p, 4) for p in path[-max(1, days):]]

    def last(self, symbol: str, now: datetime | None = None) -> float:
        inst = lookup(symbol)
        now = now or datetime.now(timezone.utc)
        base = self._path(inst, self._day_index(now))[-1]
        wobble = Rng(seed_from_text(f"{self.seed}:{inst.symbol}:{now.date()}:{now.hour}"))
        return max(0.01, base * (1.0 + wobble.gauss(0.0, 0.0015)))

    def quote(self, symbol: str, now: datetime | None = None) -> Quote:
        inst = lookup(symbol)
        now = now or datetime.now(timezone.utc)
        mid = self.last(inst.symbol, now)
        # Wider where the book would be thin: crypto always, equities off-hours.
        half_bps = 8.0 if inst.is_crypto else (2.0 if market_is_open(inst, now) else 25.0)
        half = mid * half_bps / 10_000.0
        return Quote(inst.symbol, round(mid - half, 6), round(mid + half, 6), now)

    def stats(self, symbol: str, days: int = 30, now: datetime | None = None) -> dict:
        """Return, realised vol, and two moving averages.

        Enough for the model to say something grounded instead of inventing a
        chart, which is what it does when handed a bare price.
        """
        inst = lookup(symbol)
        closes = self.closes(inst.symbol, days, now)
        # Weekend closes repeat, so a zero return is a non-observation rather
        # than a quiet day. Leaving them in drags reported equity vol down by
        # ~sqrt(5/7) and made every name look calmer than it was parameterised.
        rets = [
            math.log(closes[i] / closes[i - 1])
            for i in range(1, len(closes))
            if closes[i - 1] > 0 and closes[i] > 0 and closes[i] != closes[i - 1]
        ]
        periods = _CALENDAR_DAYS if inst.is_crypto else _TRADING_DAYS
        vol = _stdev(rets) * math.sqrt(periods) if len(rets) > 1 else 0.0
        short = closes[-min(len(closes), 10):]
        return {
            "symbol": inst.symbol,
            "days": len(closes),
            "last": closes[-1],
            "change_pct": round((closes[-1] / closes[0] - 1.0) * 100.0, 2) if closes[0] else 0.0,
            "annualised_vol_pct": round(vol * 100.0, 1),
            "high": max(closes),
            "low": min(closes),
            "sma_10": round(sum(short) / len(short), 4),
            "sma_all": round(sum(closes) / len(closes), 4),
        }


def _stdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = sum(xs) / len(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


# --------------------------------------------------------------------------
# book
# --------------------------------------------------------------------------


@dataclass
class Position:
    symbol: str
    qty: float = 0.0
    avg_price: float = 0.0
    realised_pnl: float = 0.0

    def market_value(self, price: float) -> float:
        return self.qty * price

    def unrealised(self, price: float) -> float:
        return (price - self.avg_price) * self.qty

    def as_dict(self, price: float) -> dict:
        return {
            "symbol": self.symbol,
            "qty": round(self.qty, 8),
            "avg_price": round(self.avg_price, 4),
            "last": round(price, 4),
            "market_value": round(self.market_value(price), 2),
            "unrealised_pnl": round(self.unrealised(price), 2),
            "realised_pnl": round(self.realised_pnl, 2),
        }


@dataclass(frozen=True)
class Order:
    symbol: str
    side: Side
    qty: float
    order_type: OrderType = "market"
    limit_price: float | None = None

    def notional(self, price: float) -> float:
        return abs(self.qty) * price

    def describe(self) -> str:
        px = f"@ {self.limit_price:g} limit" if self.order_type == "limit" else "@ market"
        return f"{self.side} {self.qty:g} {self.symbol} {px}"


@dataclass(frozen=True)
class Fill:
    symbol: str
    side: Side
    qty: float
    price: float
    commission: float
    ts: datetime

    @property
    def cash_delta(self) -> float:
        gross = self.qty * self.price
        return -gross - self.commission if self.side == "buy" else gross - self.commission

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "qty": round(self.qty, 8),
            "price": round(self.price, 4),
            "commission": round(self.commission, 4),
            "ts": self.ts.isoformat(timespec="seconds"),
        }


@dataclass
class Portfolio:
    cash: float = 100_000.0
    base_currency: str = "USD"
    positions: dict[str, Position] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)
    starting_cash: float = 0.0

    def __post_init__(self):
        if not self.starting_cash:
            self.starting_cash = self.cash

    def qty(self, symbol: str) -> float:
        pos = self.positions.get(symbol.upper())
        return pos.qty if pos else 0.0

    def apply(self, fill: Fill) -> None:
        pos = self.positions.setdefault(fill.symbol, Position(fill.symbol))
        if fill.side == "buy":
            total = pos.avg_price * pos.qty + fill.price * fill.qty
            pos.qty += fill.qty
            pos.avg_price = total / pos.qty if pos.qty else 0.0
        else:
            if fill.qty - pos.qty > 1e-9:
                # Shorting is refused at the risk layer. Reaching here means the
                # book and the check disagree, which is a bug, not a user error.
                raise RiskViolation(
                    f"sell of {fill.qty:g} exceeds position of {pos.qty:g} in {fill.symbol}"
                )
            pos.realised_pnl += (fill.price - pos.avg_price) * fill.qty
            pos.qty -= fill.qty
            if pos.qty <= 1e-12:
                pos.qty = 0.0
                pos.avg_price = 0.0
        self.cash += fill.cash_delta
        self.fills.append(fill)

    def _price_of(self, symbol: str, prices: dict[str, float]) -> float:
        # Falling back to avg_price marks an unpriced position flat rather than
        # to zero, which would show a fake 100% loss on any symbol the feed
        # dropped.
        return prices.get(symbol, self.positions[symbol].avg_price)

    def market_value(self, prices: dict[str, float]) -> float:
        return sum(p.market_value(self._price_of(s, prices)) for s, p in self.positions.items())

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + self.market_value(prices)

    def gross_exposure(self, prices: dict[str, float]) -> float:
        return sum(
            abs(p.market_value(self._price_of(s, prices))) for s, p in self.positions.items()
        )

    def weights(self, prices: dict[str, float]) -> dict[str, float]:
        eq = self.equity(prices)
        if eq <= 0:
            return {}
        return {
            s: p.market_value(self._price_of(s, prices)) / eq
            for s, p in self.positions.items()
            if p.qty
        }

    def snapshot(self, prices: dict[str, float]) -> dict:
        eq = self.equity(prices)
        return {
            "cash": round(self.cash, 2),
            "equity": round(eq, 2),
            "invested": round(self.market_value(prices), 2),
            "total_return_pct": round((eq / self.starting_cash - 1.0) * 100.0, 2)
            if self.starting_cash
            else 0.0,
            "realised_pnl": round(sum(p.realised_pnl for p in self.positions.values()), 2),
            "unrealised_pnl": round(
                sum(
                    p.unrealised(self._price_of(s, prices))
                    for s, p in self.positions.items()
                ),
                2,
            ),
            "positions": [
                p.as_dict(self._price_of(s, prices))
                for s, p in sorted(self.positions.items())
                if p.qty
            ],
            "fills": len(self.fills),
        }


# --------------------------------------------------------------------------
# risk
# --------------------------------------------------------------------------


@dataclass
class RiskLimits:
    """Pre-trade envelope. Every field is a hard stop, not a hint.

    Defaults are deliberately tight. A model that has to argue its way up to a
    bigger limit is a better failure mode than one that discovers it never had
    a limit at all.
    """

    max_order_notional: float = 10_000.0
    min_order_notional: float = 50.0
    max_position_weight: float = 0.25       # fraction of equity per symbol
    max_gross_exposure: float = 0.90        # fraction of equity invested
    min_cash_buffer: float = 0.05           # fraction of equity kept in cash
    daily_loss_limit_pct: float = 5.0       # halt after this intraday drawdown
    max_orders_per_day: int = 20
    allow_short: bool = False
    allow_leverage: bool = False
    allowlist: tuple[str, ...] = ()         # empty = the whole universe
    halted: bool = False                    # kill switch, see TradingSession.halt

    def check(
        self,
        order: Order,
        quote: Quote,
        portfolio: Portfolio,
        prices: dict[str, float],
        *,
        orders_today: int = 0,
        day_start_equity: float = 0.0,
        now: datetime | None = None,
    ) -> list[str]:
        """Every reason this order is refused. Empty list means allowed.

        Returns all of them rather than the first: a model that fixes one
        violation per turn burns the whole iteration budget on one order.
        """
        reasons: list[str] = []
        inst = lookup(order.symbol)
        equity = portfolio.equity(prices)
        price = quote.ask if order.side == "buy" else quote.bid
        notional = order.notional(price)

        if self.halted:
            reasons.append("trading is halted (kill switch is on)")
        if self.allowlist and order.symbol not in self.allowlist:
            reasons.append(f"{order.symbol} is not on the allowlist {sorted(self.allowlist)}")
        if order.qty <= 0:
            reasons.append("quantity must be positive; use side='sell' to reduce a position")
        elif abs(order.qty - inst.round_qty(order.qty)) > 1e-9:
            reasons.append(
                f"quantity {order.qty:g} is not a multiple of lot size {inst.lot_size:g}"
            )
        if not market_is_open(inst, now):
            reasons.append(f"{order.symbol} market is closed")
        if notional > self.max_order_notional:
            reasons.append(
                f"notional {notional:,.0f} exceeds the per-order cap "
                f"{self.max_order_notional:,.0f}"
            )
        if 0 < notional < self.min_order_notional:
            reasons.append(
                f"notional {notional:,.0f} is below the {self.min_order_notional:,.0f} floor"
            )
        if orders_today >= self.max_orders_per_day:
            reasons.append(f"daily order cap reached ({self.max_orders_per_day})")

        if day_start_equity > 0:
            drawdown = (1.0 - equity / day_start_equity) * 100.0
            if drawdown >= self.daily_loss_limit_pct:
                reasons.append(
                    f"daily loss limit hit: down {drawdown:.1f}% against a "
                    f"{self.daily_loss_limit_pct:.1f}% limit"
                )

        if order.side == "buy":
            if not self.allow_leverage and notional > portfolio.cash:
                reasons.append(
                    f"insufficient cash: need {notional:,.0f}, have {portfolio.cash:,.0f}"
                )
            if equity > 0:
                if portfolio.cash - notional < self.min_cash_buffer * equity:
                    reasons.append(f"would break the {self.min_cash_buffer:.0%} cash buffer")
                held_value = portfolio.qty(order.symbol) * price
                weight = (held_value + notional) / equity
                if weight > self.max_position_weight:
                    reasons.append(
                        f"{order.symbol} would be {weight:.0%} of equity, cap is "
                        f"{self.max_position_weight:.0%}"
                    )
                gross = (portfolio.gross_exposure(prices) + notional) / equity
                if gross > self.max_gross_exposure:
                    reasons.append(
                        f"gross exposure would be {gross:.0%}, cap is "
                        f"{self.max_gross_exposure:.0%}"
                    )
        else:
            held = portfolio.qty(order.symbol)
            if order.qty - held > 1e-9 and not self.allow_short:
                reasons.append(
                    f"shorting is off: selling {order.qty:g} {order.symbol} with {held:g} held"
                )
        return reasons

    def enforce(self, *args, **kwargs) -> None:
        reasons = self.check(*args, **kwargs)
        if reasons:
            raise RiskViolation("order refused: " + "; ".join(reasons), reasons=reasons)


# --------------------------------------------------------------------------
# brokers
# --------------------------------------------------------------------------


class PaperBroker:
    """Simulated execution: the whole order at the touch, plus slippage.

    Optimistic on purpose. It is the simplest model that is obviously a model —
    a partial-fill and queue simulation would look far more credible without
    being meaningfully more accurate, which is the worse outcome.
    """

    name = "paper"
    live = False

    def __init__(
        self,
        feed: SyntheticFeed,
        *,
        slippage_bps: float = 5.0,
        commission_bps: float = 2.0,
        min_commission: float = 0.0,
    ):
        self.feed = feed
        self.slippage_bps = slippage_bps
        self.commission_bps = commission_bps
        self.min_commission = min_commission

    def place(self, order: Order, quote: Quote) -> Fill:
        slip = self.slippage_bps / 10_000.0
        if order.side == "buy":
            price = quote.ask * (1.0 + slip)
            if order.order_type == "limit" and order.limit_price is not None:
                if price > order.limit_price:
                    raise RiskViolation(
                        f"limit {order.limit_price:g} is not marketable; ask is "
                        f"{quote.ask:g} before {self.slippage_bps:g}bps slippage"
                    )
        else:
            price = quote.bid * (1.0 - slip)
            if order.order_type == "limit" and order.limit_price is not None:
                if price < order.limit_price:
                    raise RiskViolation(
                        f"limit {order.limit_price:g} is not marketable; bid is "
                        f"{quote.bid:g} before {self.slippage_bps:g}bps slippage"
                    )

        commission = max(
            self.min_commission, order.qty * price * self.commission_bps / 10_000.0
        )
        return Fill(order.symbol, order.side, order.qty, price, commission, quote.ts)


class LiveBroker:
    """Placeholder for a real venue. Does not exist and will not construct.

    Wiring one means, at minimum: idempotent order submission, reconciliation
    against the venue's book on startup (this module's position state is
    authoritative only because nothing else touches it), an order audit trail
    that survives the process, and a second pair of eyes on the credentials.
    None of that is here.
    """

    name = "live"
    live = True

    def __init__(self, adapter=None):
        raise NotImplementedError(
            "no live broker adapter ships in this tree. An LLM with market access and "
            "no reconciliation is a way to lose money in a novel fashion. If you are "
            f"building one anyway: gate it behind {LIVE_FLAG}, keep autonomy at "
            "'confirm', and read the STATUS block in this module first."
        )


# --------------------------------------------------------------------------
# proposals
# --------------------------------------------------------------------------


@dataclass
class Proposal:
    id: str
    order: Order
    rationale: str
    quote: Quote
    created_at: datetime
    status: str = "pending"          # pending | advisory | filled | rejected | refused
    reasons: list[str] = field(default_factory=list)
    fill: Fill | None = None

    @property
    def allowed(self) -> bool:
        return not self.reasons

    def summary(self) -> str:
        lines = [
            f"[{self.id}] {self.order.describe()}  ({self.status})",
            f"  est. notional {self.order.notional(self.quote.mid):,.2f}",
        ]
        if self.rationale.strip():
            lines.append(f"  why: {self.rationale.strip()}")
        if self.reasons:
            lines.append(f"  refused: {'; '.join(self.reasons)}")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "symbol": self.order.symbol,
            "side": self.order.side,
            "qty": self.order.qty,
            "order_type": self.order.order_type,
            "limit_price": self.order.limit_price,
            "status": self.status,
            "rationale": self.rationale,
            "estimated_notional": round(self.order.notional(self.quote.mid), 2),
            "risk_reasons": self.reasons,
            "fill": self.fill.as_dict() if self.fill else None,
        }


# --------------------------------------------------------------------------
# session
# --------------------------------------------------------------------------


class TradingSession:
    """Feed, book, risk, broker, and journal in one object.

    The session owns the only path from a model output to a fill, so every order
    is journalled with the rationale that produced it. A trade with no written
    reason is not something this module can express.
    """

    def __init__(
        self,
        feed: SyntheticFeed,
        portfolio: Portfolio,
        broker: PaperBroker,
        limits: RiskLimits | None = None,
        *,
        autonomy: Autonomy = "confirm",
        symbols: list[str] | None = None,
    ):
        _require_enabled()
        if getattr(broker, "live", False):
            if os.environ.get(LIVE_FLAG, "").lower() not in _TRUTHY:
                raise InvestorDisabled(
                    f"live brokers require {LIVE_FLAG}=1 and an adapter you wrote yourself"
                )
            if autonomy == "auto":
                raise InvestorDisabled("autonomy='auto' is refused against a live broker")

        self.feed = feed
        self.portfolio = portfolio
        self.broker = broker
        self.limits = limits or RiskLimits()
        self.autonomy: Autonomy = autonomy
        self.symbols = [lookup(s).symbol for s in (symbols or list(UNIVERSE))]
        if not self.limits.allowlist:
            self.limits.allowlist = tuple(self.symbols)

        self.proposals: dict[str, Proposal] = {}
        self.journal: list[dict] = []
        self.equity_curve: list[tuple[datetime, float]] = []
        self._orders_today = 0
        self._day = datetime.now(timezone.utc).date()
        self._day_start_equity = portfolio.equity(self.prices())
        self.mark()

    @classmethod
    def paper(
        cls,
        symbols: list[str] | None = None,
        *,
        cash: float = 100_000.0,
        seed: int = 0,
        limits: RiskLimits | None = None,
        autonomy: Autonomy = "confirm",
    ) -> TradingSession:
        """The only constructor most callers want."""
        _require_enabled()
        feed = SyntheticFeed(seed=seed)
        return cls(
            feed,
            Portfolio(cash=cash),
            PaperBroker(feed),
            limits,
            autonomy=autonomy,
            symbols=symbols,
        )

    # -- marks -------------------------------------------------------------

    def prices(self, now: datetime | None = None) -> dict[str, float]:
        wanted = set(self.portfolio.positions) | set(getattr(self, "symbols", []))
        return {s: self.feed.last(s, now) for s in wanted if s in UNIVERSE}

    def mark(self, now: datetime | None = None) -> float:
        """Mark to market, roll the day if needed, record the equity point."""
        now = now or datetime.now(timezone.utc)
        if now.date() != self._day:
            self._day = now.date()
            self._orders_today = 0
            self._day_start_equity = self.portfolio.equity(self.prices(now))
        equity = self.portfolio.equity(self.prices(now))
        self.equity_curve.append((now, equity))
        return equity

    def _record(self, kind: str, **payload) -> dict:
        entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "kind": kind}
        entry.update(payload)
        self.journal.append(entry)
        return entry

    # -- proposal lifecycle ------------------------------------------------

    def propose(
        self,
        symbol: str,
        side: Side,
        qty: float,
        rationale: str,
        *,
        order_type: OrderType = "market",
        limit_price: float | None = None,
    ) -> Proposal:
        """Register an intended trade. Executes only under autonomy='auto'."""
        if not rationale or len(rationale.strip()) < 20:
            raise RiskViolation(
                "a proposal needs a rationale of at least 20 characters: what you expect, "
                "on what evidence, and what would prove you wrong"
            )
        inst = lookup(symbol)
        if side not in ("buy", "sell"):
            raise RiskViolation(f"side must be 'buy' or 'sell', got {side!r}")

        now = datetime.now(timezone.utc)
        self.mark(now)
        quote = self.feed.quote(inst.symbol, now)
        order = Order(inst.symbol, side, inst.round_qty(float(qty)), order_type, limit_price)

        proposal = Proposal(
            id=uuid.uuid4().hex[:8],
            order=order,
            rationale=rationale.strip(),
            quote=quote,
            created_at=now,
            reasons=self.limits.check(
                order,
                quote,
                self.portfolio,
                self.prices(now),
                orders_today=self._orders_today,
                day_start_equity=self._day_start_equity,
                now=now,
            ),
        )
        if proposal.reasons:
            proposal.status = "refused"
        elif self.autonomy == "advisory":
            proposal.status = "advisory"
        self.proposals[proposal.id] = proposal
        self._record("proposal", **proposal.as_dict())

        if proposal.allowed and self.autonomy == "auto":
            self.approve(proposal.id)
        return proposal

    def approve(self, proposal_id: str) -> Fill:
        """Execute a pending proposal. This is the human's decision point."""
        proposal = self.proposals.get(proposal_id)
        if proposal is None:
            raise RiskViolation(f"no proposal {proposal_id!r}")
        if self.autonomy == "advisory":
            raise RiskViolation("session autonomy is 'advisory'; proposals cannot be executed")
        if proposal.status != "pending":
            raise RiskViolation(f"proposal {proposal_id} is {proposal.status}, not pending")

        now = datetime.now(timezone.utc)
        quote = self.feed.quote(proposal.order.symbol, now)
        # Re-check against a fresh quote. The price that made this sensible is
        # not necessarily the price we are about to pay.
        self.limits.enforce(
            proposal.order,
            quote,
            self.portfolio,
            self.prices(now),
            orders_today=self._orders_today,
            day_start_equity=self._day_start_equity,
            now=now,
        )
        fill = self.broker.place(proposal.order, quote)
        self.portfolio.apply(fill)
        self._orders_today += 1
        proposal.status = "filled"
        proposal.fill = fill
        self._record(
            "fill", proposal_id=proposal.id, rationale=proposal.rationale, **fill.as_dict()
        )
        self.mark(now)
        log.info("filled %s: %s", proposal.id, proposal.order.describe())
        return fill

    def reject(self, proposal_id: str, reason: str = "") -> Proposal:
        proposal = self.proposals.get(proposal_id)
        if proposal is None:
            raise RiskViolation(f"no proposal {proposal_id!r}")
        proposal.status = "rejected"
        proposal.reasons.append(reason or "rejected by operator")
        self._record("rejected", proposal_id=proposal.id, reason=reason)
        return proposal

    def pending(self) -> list[Proposal]:
        return [p for p in self.proposals.values() if p.status == "pending"]

    def halt(self, reason: str = "manual") -> None:
        """Kill switch. Blocks every subsequent order until cleared by hand."""
        self.limits.halted = True
        self._record("halt", reason=reason)
        log.warning("trading halted: %s", reason)

    def resume(self) -> None:
        self.limits.halted = False
        self._record("resume")

    # -- reporting ---------------------------------------------------------

    def performance(self) -> dict:
        curve = [e for _, e in self.equity_curve] or [self.portfolio.starting_cash]
        peak, max_dd = curve[0], 0.0
        for e in curve:
            peak = max(peak, e)
            if peak > 0:
                max_dd = max(max_dd, (peak - e) / peak)
        start = self.portfolio.starting_cash
        return {
            "start_equity": round(start, 2),
            "equity": round(curve[-1], 2),
            "total_return_pct": round((curve[-1] / start - 1.0) * 100.0, 2) if start else 0.0,
            "max_drawdown_pct": round(max_dd * 100.0, 2),
            "marks": len(curve),
            "fills": len(self.portfolio.fills),
            "halted": self.limits.halted,
        }

    def report(self) -> str:
        prices = self.prices()
        snap = self.portfolio.snapshot(prices)
        perf = self.performance()
        lines = [
            f"equity {snap['equity']:,.2f}   cash {snap['cash']:,.2f}   "
            f"return {perf['total_return_pct']:+.2f}%   maxDD {perf['max_drawdown_pct']:.2f}%"
        ]
        if snap["positions"]:
            lines.append("positions:")
            for p in snap["positions"]:
                weight = p["market_value"] / snap["equity"] * 100.0 if snap["equity"] else 0.0
                lines.append(
                    f"  {p['symbol']:<9} {p['qty']:>13.6g} @ {p['avg_price']:>10,.2f}  "
                    f"mv {p['market_value']:>11,.2f} ({weight:4.1f}%)  "
                    f"upnl {p['unrealised_pnl']:+,.2f}"
                )
        else:
            lines.append("positions: none")
        if self.limits.halted:
            lines.append("HALTED — kill switch is on, no orders will fill")
        pend = self.pending()
        if pend:
            lines.append(f"pending: {len(pend)} proposal(s) awaiting approval")
        lines.append(DISCLAIMER)
        return "\n".join(lines)

    # -- tools -------------------------------------------------------------

    def tool_registry(self) -> ToolRegistry:
        """Tools bound to *this* session, in a private registry.

        Deliberately not the global REGISTRY. A default agent should not
        discover it can place orders because something imported this module.
        """
        reg = ToolRegistry()
        session = self

        @tool(registry=reg, description="Get the current bid/ask quote for a symbol.")
        def quote(symbol: str) -> dict:
            return session.feed.quote(lookup(symbol).symbol).as_dict()

        @tool(
            registry=reg,
            description="Recent daily closes and summary statistics for a symbol.",
        )
        def price_history(symbol: str, days: int = 30) -> dict:
            days = max(2, min(int(days), 365))
            stats = session.feed.stats(symbol, days)
            stats["closes"] = session.feed.closes(symbol, min(days, 60))
            return stats

        @tool(registry=reg, description="List the tradeable universe and whether each is open.")
        def universe() -> dict:
            return {
                "symbols": [
                    {
                        "symbol": s,
                        "asset_class": UNIVERSE[s].asset_class,
                        "name": UNIVERSE[s].name,
                        "market_open": market_is_open(UNIVERSE[s]),
                    }
                    for s in session.symbols
                    if s in UNIVERSE
                ]
            }

        @tool(registry=reg, description="Current portfolio: cash, equity, positions, P&L.")
        def portfolio() -> dict:
            return session.portfolio.snapshot(session.prices())

        @tool(registry=reg, description="The risk limits every order is checked against.")
        def risk_limits() -> dict:
            lim = session.limits
            return {
                "max_order_notional": lim.max_order_notional,
                "min_order_notional": lim.min_order_notional,
                "max_position_weight_pct": lim.max_position_weight * 100,
                "max_gross_exposure_pct": lim.max_gross_exposure * 100,
                "min_cash_buffer_pct": lim.min_cash_buffer * 100,
                "daily_loss_limit_pct": lim.daily_loss_limit_pct,
                "orders_remaining_today": max(0, lim.max_orders_per_day - session._orders_today),
                "shorting_allowed": lim.allow_short,
                "leverage_allowed": lim.allow_leverage,
                "halted": lim.halted,
                "autonomy": session.autonomy,
            }

        @tool(
            registry=reg,
            dangerous=True,
            description=(
                "Propose a trade. The rationale must state the thesis, the evidence, and "
                "what would falsify it. Under 'confirm' autonomy this does NOT execute — a "
                "human approves it afterwards."
            ),
        )
        def propose_trade(
            symbol: str,
            side: Literal["buy", "sell"],
            quantity: float,
            rationale: str,
            limit_price: float | None = None,
        ) -> dict:
            try:
                proposal = session.propose(
                    symbol,
                    side,
                    quantity,
                    rationale,
                    order_type="limit" if limit_price else "market",
                    limit_price=limit_price,
                )
            except RiskViolation as exc:
                # A refusal is information the model should reason about, not a
                # dead turn. Raising here just makes the loop retry blind.
                return {"status": "refused", "error": str(exc), "reasons": exc.reasons}
            return proposal.as_dict()

        return reg


# --------------------------------------------------------------------------
# persona + bot
# --------------------------------------------------------------------------


def investor_persona(session: TradingSession) -> Persona:
    """The analyst persona, with the session's actual constraints inlined.

    Stating the limits in the prompt is not the enforcement — RiskLimits is —
    but a model that knows the cap stops proposing orders that get refused, and
    the refusal loop was the largest source of wasted iterations in testing.
    """
    lim = session.limits
    autonomy_note = (
        " You propose; a human approves. Nothing you say moves money by itself."
        if session.autonomy != "auto"
        else " Your proposals execute immediately. Be correspondingly careful."
    )
    return Persona(
        name="investor",
        instructions=(
            "You manage a simulated portfolio. Prices come from a synthetic feed: they are "
            "not real market data, and nothing you conclude here transfers to a real "
            "market.\n\n"
            "How you work:\n"
            "- Look before you act. Quote the symbol and check its history before "
            "proposing anything.\n"
            "- Every proposal states a thesis, the evidence behind it, the size and why "
            "that size, and what would prove the thesis wrong.\n"
            "- Size by risk, not by conviction. A view you cannot size is a view you do "
            "not have.\n"
            "- 'No trade' is a valid and frequently correct answer. Do not manufacture "
            "activity.\n"
            "- Never state a prediction as fact. Talk about asymmetry and uncertainty.\n"
            "- Do not advise the user on real money or real securities. You are operating "
            "a simulator, and you say so plainly if asked.\n\n"
            f"Your constraints: at most {lim.max_order_notional:,.0f} per order, "
            f"{lim.max_position_weight:.0%} of equity in any one symbol, "
            f"{lim.max_gross_exposure:.0%} gross invested, "
            f"{'shorting allowed' if lim.allow_short else 'no shorting'}, "
            f"{'leverage allowed' if lim.allow_leverage else 'no leverage'}. "
            f"Autonomy is '{session.autonomy}'.{autonomy_note}"
        ),
        traits=[
            "quantitative and specific; numbers rather than adjectives",
            "explicit about uncertainty and about what would change your mind",
            "no hype, no price targets stated as fact",
        ],
        verbosity="balanced",
    )


class InvestorBot:
    """Binds an Engine to a TradingSession.

    `brief()` is computed, not generated — it is exactly the state the model
    would otherwise invent. The agent sees it before every turn.
    """

    def __init__(
        self,
        engine,
        session: TradingSession,
        *,
        preset: str = "research",
        max_iterations: int = 10,
    ):
        _require_enabled()
        self.engine = engine
        self.session = session
        self.preset = preset
        self.max_iterations = max_iterations
        self._agent = None

    def agent(self):
        if self._agent is None:
            from ..agent.presets import get_preset

            self._agent = get_preset(self.preset).build_agent(
                self.engine,
                tools=self.session.tool_registry(),
                persona=investor_persona(self.session),
                max_iterations=self.max_iterations,
            )
        return self._agent

    def brief(self) -> str:
        """Market and portfolio state as text. Deterministic, no model call."""
        lines = ["market:"]
        for symbol in self.session.symbols:
            if symbol not in UNIVERSE:
                continue
            stats = self.session.feed.stats(symbol, 30)
            trend = "above" if stats["last"] >= stats["sma_10"] else "below"
            closed = "" if market_is_open(UNIVERSE[symbol]) else "   [closed]"
            lines.append(
                f"  {symbol:<9} {stats['last']:>11,.2f}  30d {stats['change_pct']:+6.2f}%  "
                f"vol {stats['annualised_vol_pct']:>5.1f}%  {trend} 10d SMA{closed}"
            )
        lines.append("")
        lines.append(self.session.report())
        return "\n".join(lines)

    def ask(self, question: str):
        """One agent turn with the trading tools available."""
        return self.agent().run(f"{self.brief()}\n\n{question}")

    def propose(self, objective: str = "review the book and act only if warranted"):
        """Run the loop against an objective; return the proposals it produced."""
        before = set(self.session.proposals)
        self.ask(
            f"{objective}\n\n"
            "Use the tools to check prices before deciding. To trade, call propose_trade "
            "with a full rationale. If nothing is worth doing, say so and propose nothing."
        )
        return [p for pid, p in self.session.proposals.items() if pid not in before]

    # Convenience passthroughs so a UI only needs the bot object.
    def approve(self, proposal_id: str) -> Fill:
        return self.session.approve(proposal_id)

    def reject(self, proposal_id: str, reason: str = "") -> Proposal:
        return self.session.reject(proposal_id, reason)

    def report(self) -> str:
        return self.session.report()
