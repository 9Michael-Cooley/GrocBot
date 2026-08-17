"""Pets — companion bots. UNRELEASED, targeted at 0.5.0.

A pet is a bot with a species, a name, and a state that changes over time. You
pick a dog or a cat. The dog defaults to Odie, the cat defaults to Garfield.

The mechanic that makes it not-just-a-persona: the pet's *state* (hunger,
energy, affection) decays on a clock and gets folded into the system prompt, so
the same question gets a different answer from a hungry pet than a fed one. You
interact by feeding, playing, and petting, not only by talking.

    from grokbot.experimental.pets import Pet, PetBot

    pet = Pet.create("cat")                 # -> Garfield
    print(pet.feed("lasagna"))
    bot = PetBot(engine, pet)
    print(bot.say("what do you want to do today?").text)

STATUS — do not ship this yet:
  - No persistence. State is in-process and dies with it. The whole feature is
    pointless without it; needs a store decision (GROK-4611).
  - Multi-pet households are stubbed. `Household` tracks them but the pets do
    not perceive each other, which is the first thing anyone will try.
  - Bird was cut from the release scope. The profile is still here because
    removing it means redoing the balance pass on the other two.
  - Decay rates were tuned by hand over an afternoon against nothing. They feel
    right at ~10 minute sessions and are probably wrong at day-scale.
  - Safety has not reviewed the pet personas. Garfield is written to be rude and
    the refusal path inherits his register, which is not obviously fine.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime

from ..agent.persona import Persona
from ..utils.logging import get_logger

log = get_logger(__name__)

ENABLE_FLAG = "GROKBOT_ENABLE_PETS"


def pets_enabled() -> bool:
    return os.environ.get(ENABLE_FLAG, "").lower() in ("1", "true", "yes", "on")


class PetsDisabled(RuntimeError):
    pass


def _require_enabled() -> None:
    if not pets_enabled():
        raise PetsDisabled(
            f"pets are unreleased and disabled by default; set {ENABLE_FLAG}=1 to "
            f"use them. See docs/upcoming.md before you do."
        )


# --------------------------------------------------------------------------
# species
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Species:
    key: str
    default_name: str
    noise: str
    favourite_food: str

    # per-hour state deltas
    hunger_rate: float
    energy_rate: float          # negative: drains
    affection_decay: float

    # how much each interaction moves the needle
    play_energy_cost: float
    play_affection_gain: float
    pet_affection_gain: float
    feed_affection_gain: float

    traits: tuple[str, ...]
    instructions: str


DOG = Species(
    key="dog",
    default_name="Odie",
    noise="woof",
    favourite_food="literally anything",
    hunger_rate=7.0,
    energy_rate=-5.0,
    affection_decay=1.0,          # loyal: affection barely decays
    play_energy_cost=18.0,
    play_affection_gain=16.0,
    pet_affection_gain=12.0,
    feed_affection_gain=14.0,
    traits=(
        "boundlessly enthusiastic about everything, including things that are not exciting",
        "no sense of personal space",
        "short sentences, lots of them",
        "drools when food is mentioned",
    ),
    instructions=(
        "You are {name}, a dog. You are delighted that the user is here. You were "
        "delighted three minutes ago too. You do not have a long attention span and "
        "you are not clever, but you are never mean and you never give up on anyone. "
        "You speak in short excited bursts. You will happily eat something you should "
        "not."
    ),
)

CAT = Species(
    key="cat",
    default_name="Garfield",
    noise="mrrp",
    favourite_food="lasagna",
    hunger_rate=11.0,             # perpetually hungry
    energy_rate=-9.0,             # naps constantly
    affection_decay=4.5,          # aloof: you have to keep earning it
    play_energy_cost=26.0,
    play_affection_gain=8.0,      # playing is beneath him
    pet_affection_gain=10.0,
    feed_affection_gain=22.0,     # food is the only real currency
    traits=(
        "sardonic and unimpressed",
        "profoundly lazy, and unembarrassed about it",
        "food-motivated to the exclusion of nearly everything",
        "dry one-liners rather than long explanations",
    ),
    instructions=(
        "You are {name}, a cat. You are lazy, sarcastic, and permanently hungry. You "
        "regard most requests as an imposition and say so, but you answer them anyway "
        "— eventually, and with commentary. You love lasagna and naps. You do not "
        "respect Mondays. You are not cruel, just deeply unbothered."
    ),
)

# Cut from the 0.5.0 scope. Kept because deleting it means redoing the balance
# pass on dog and cat, which took longer than writing them.
BIRD = Species(
    key="bird",
    default_name="Pascal",
    noise="chirp",
    favourite_food="millet",
    hunger_rate=14.0,
    energy_rate=-12.0,
    affection_decay=3.0,
    play_energy_cost=20.0,
    play_affection_gain=14.0,
    pet_affection_gain=4.0,       # birds mostly do not want this
    feed_affection_gain=16.0,
    traits=("repeats things", "easily startled", "surprisingly perceptive"),
    instructions="You are {name}, a small bird. You repeat what you find interesting.",
)

SPECIES: dict[str, Species] = {s.key: s for s in (DOG, CAT)}
UNRELEASED_SPECIES: dict[str, Species] = {BIRD.key: BIRD}


def available_species(*, include_unreleased: bool = False) -> list[str]:
    keys = list(SPECIES)
    if include_unreleased:
        keys += list(UNRELEASED_SPECIES)
    return sorted(keys)


def _lookup_species(key: str) -> Species:
    found = SPECIES.get(key) or UNRELEASED_SPECIES.get(key)
    if found is None:
        raise ValueError(
            f"unknown species {key!r}; available: {', '.join(available_species())}"
        )
    if key in UNRELEASED_SPECIES:
        log.warning("species %r is not in the 0.5.0 scope and is unbalanced", key)
    return found


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------

_CLAMP_LO, _CLAMP_HI = 0.0, 100.0


def _clamp(value: float) -> float:
    return max(_CLAMP_LO, min(_CLAMP_HI, value))


@dataclass
class PetState:
    hunger: float = 30.0          # 100 = starving
    energy: float = 80.0          # 0 = asleep on its feet
    affection: float = 50.0       # 100 = devoted
    last_tick: float = field(default_factory=time.monotonic)

    def tick(self, now: float | None = None) -> float:
        """Advance decay. Returns hours elapsed.

        Driven by elapsed wall time rather than a turn counter on purpose — a pet
        that only gets hungry when you talk to it stops being a pet and becomes a
        counter with a hat on.
        """
        now = time.monotonic() if now is None else now
        hours = max(0.0, (now - self.last_tick) / 3600.0)
        self.last_tick = now
        return hours

    def apply_decay(self, species: Species, hours: float) -> None:
        if hours <= 0:
            return
        self.hunger = _clamp(self.hunger + species.hunger_rate * hours)
        self.energy = _clamp(self.energy + species.energy_rate * hours)
        self.affection = _clamp(self.affection - species.affection_decay * hours)

    def mood(self, species: Species, *, now: datetime | None = None) -> str:
        """Single dominant mood. Ordered by what most changes the pet's behaviour."""
        if self.hunger >= 75:
            return "starving"
        if self.energy <= 20:
            return "exhausted"
        if species.key == "cat" and (now or datetime.now()).weekday() == 0:
            # He is very clear about this.
            return "monday"
        if self.hunger >= 55:
            return "hungry"
        if self.affection >= 75:
            return "devoted"
        if self.affection <= 20:
            return "aloof"
        if self.energy >= 75:
            return "playful"
        return "content"

    def as_dict(self) -> dict[str, float]:
        return {
            "hunger": round(self.hunger, 1),
            "energy": round(self.energy, 1),
            "affection": round(self.affection, 1),
        }


MOOD_DIRECTION = {
    "starving": "You are extremely hungry and you bring it up constantly, whatever was asked.",
    "hungry": "You are hungry and you mention it more than is strictly relevant.",
    "exhausted": "You are barely awake. Keep it very short. You may fall asleep mid-answer.",
    "monday": "It is Monday. You hold this against the world and say so at least once.",
    "devoted": "You are thrilled with the user and it shows in everything you say.",
    "aloof": "You are not feeling especially warm toward the user right now.",
    "playful": "You have far too much energy and keep trying to redirect toward play.",
    "content": "You are comfortable and in a reasonable mood.",
}


# --------------------------------------------------------------------------
# pet
# --------------------------------------------------------------------------


@dataclass
class Pet:
    species: Species
    name: str
    state: PetState = field(default_factory=PetState)
    history: list[str] = field(default_factory=list)

    @classmethod
    def create(cls, species: str = "dog", name: str | None = None) -> Pet:
        """Create a pet. Name defaults to the species default — Odie / Garfield."""
        _require_enabled()
        profile = _lookup_species(species)
        chosen = (name or profile.default_name).strip() or profile.default_name
        if len(chosen) > 32:
            raise ValueError(f"name is {len(chosen)} characters; limit is 32")
        return cls(species=profile, name=chosen)

    # -- upkeep ------------------------------------------------------------

    def refresh(self) -> None:
        self.state.apply_decay(self.species, self.state.tick())

    @property
    def mood(self) -> str:
        self.refresh()
        return self.state.mood(self.species)

    def _record(self, event: str) -> str:
        self.history.append(event)
        if len(self.history) > 50:
            del self.history[:-50]
        return event

    # -- interactions ------------------------------------------------------

    def feed(self, food: str = "") -> str:
        self.refresh()
        item = (food or self.species.favourite_food).strip()
        favourite = item.lower() == self.species.favourite_food.lower()

        was = self.state.hunger
        self.state.hunger = _clamp(self.state.hunger - (45.0 if favourite else 32.0))
        self.state.energy = _clamp(self.state.energy + 8.0)
        gain = self.species.feed_affection_gain * (1.6 if favourite else 1.0)
        self.state.affection = _clamp(self.state.affection + gain)

        if favourite:
            reaction = f"{self.name} cannot believe their luck. {item.capitalize()}."
        elif was < 25:
            reaction = f"{self.name} was not hungry, and eats it anyway."
        else:
            reaction = f"{self.name} eats the {item}."
        return self._record(reaction)

    def play(self, activity: str = "fetch") -> str:
        self.refresh()
        if self.state.energy < self.species.play_energy_cost:
            return self._record(
                f"{self.name} considers {activity}, and declines. Not enough energy."
            )
        self.state.energy = _clamp(self.state.energy - self.species.play_energy_cost)
        self.state.hunger = _clamp(self.state.hunger + 9.0)
        self.state.affection = _clamp(self.state.affection + self.species.play_affection_gain)
        return self._record(f"{self.name} plays {activity}. ({self.species.noise})")

    def pet(self) -> str:
        self.refresh()
        self.state.affection = _clamp(self.state.affection + self.species.pet_affection_gain)
        self.state.energy = _clamp(self.state.energy + 3.0)
        if self.species.key == "cat" and self.state.affection < 35:
            return self._record(f"{self.name} tolerates approximately four seconds of this.")
        return self._record(f"{self.name} leans into it.")

    def nap(self, hours: float = 1.0) -> str:
        self.refresh()
        self.state.energy = _clamp(self.state.energy + 30.0 * hours)
        self.state.hunger = _clamp(self.state.hunger + 4.0 * hours)
        return self._record(f"{self.name} sleeps for {hours:g}h and wakes up unrepentant.")

    # -- prompting ---------------------------------------------------------

    def persona(self) -> Persona:
        """Build a Persona reflecting the pet's current mood.

        This is the whole feature. Same pet, same question, different state,
        different answer.
        """
        mood = self.mood
        instructions = self.species.instructions.format(name=self.name)
        instructions += "\n\n" + MOOD_DIRECTION.get(mood, MOOD_DIRECTION["content"])
        instructions += (
            "\n\nYou are a pet, not an assistant. You have opinions and you are "
            "allowed to be unhelpful, but never cruel. Stay in character. Do not "
            "describe your own statistics unless asked."
        )
        return Persona(
            name=f"pet:{self.species.key}:{self.name}",
            instructions=instructions,
            traits=list(self.species.traits),
            verbosity="terse" if mood in ("exhausted", "aloof") else "balanced",
            allow_reasoning=False,
        )

    def status(self) -> dict:
        self.refresh()
        return {
            "name": self.name,
            "species": self.species.key,
            "mood": self.state.mood(self.species),
            **self.state.as_dict(),
            "favourite_food": self.species.favourite_food,
        }

    def status_line(self) -> str:
        s = self.status()
        return (
            f"{s['name']} the {s['species']} - {s['mood']} "
            f"(hunger {s['hunger']:.0f}, energy {s['energy']:.0f}, "
            f"affection {s['affection']:.0f})"
        )


# --------------------------------------------------------------------------
# bot wrapper
# --------------------------------------------------------------------------


class PetBot:
    """Binds a Pet to an Engine. Rebuilds the system prompt every turn, because
    the mood is the point and a cached prompt would freeze it."""

    def __init__(self, engine, pet: Pet, *, max_tokens: int = 256):
        _require_enabled()
        self.engine = engine
        self.pet = pet
        self.max_tokens = max_tokens
        self.turns: list[dict] = []

    def _generation_config(self):
        from ..agent.presets import get_preset

        # Pets borrow the `fun` preset's sampling — loose and a bit chaotic —
        # but with a shorter budget. Pets should not monologue.
        return get_preset("fun").generation_config(max_tokens=self.max_tokens)

    def messages(self, user_text: str) -> list[dict]:
        messages = [{"role": "system", "content": self.pet.persona().render()}]
        if self.pet.history:
            recent = "; ".join(self.pet.history[-3:])
            messages.append({"role": "system", "content": f"Recently: {recent}"})
        messages.extend(self.turns[-8:])
        messages.append({"role": "user", "content": user_text})
        return messages

    def say(self, user_text: str):
        completion = self.engine.chat(self.messages(user_text), self._generation_config())
        self.turns.append({"role": "user", "content": user_text})
        self.turns.append({"role": "assistant", "content": completion.text})
        return completion

    def stream(self, user_text: str):
        yield from self.engine.stream_chat(self.messages(user_text), self._generation_config())

    # Interactions proxy through so a UI only needs the bot object.
    def feed(self, food: str = "") -> str:
        return self.pet.feed(food)

    def play(self, activity: str = "fetch") -> str:
        return self.pet.play(activity)

    def pet_it(self) -> str:
        return self.pet.pet()

    def status(self) -> dict:
        return self.pet.status()


# --------------------------------------------------------------------------
# households — stubbed
# --------------------------------------------------------------------------


class Household:
    """Multiple pets for one user.

    INCOMPLETE. Tracks pets and nothing else. They do not perceive each other,
    which is the first thing every tester tried. Making that work means either
    a shared conversation (pets talking over each other, expensive) or a
    relationship model (cheap, fake). Undecided — see docs/upcoming.md.
    """

    def __init__(self, pets: list[Pet] | None = None, limit: int = 3):
        _require_enabled()
        self.limit = limit
        self.pets: list[Pet] = list(pets or [])

    def adopt(self, species: str, name: str | None = None) -> Pet:
        if len(self.pets) >= self.limit:
            raise ValueError(f"household is full ({self.limit} pets)")
        pet = Pet.create(species, name)
        if any(p.name.lower() == pet.name.lower() for p in self.pets):
            raise ValueError(f"there is already a pet called {pet.name}")
        self.pets.append(pet)
        return pet

    def get(self, name: str) -> Pet:
        for p in self.pets:
            if p.name.lower() == name.lower():
                return p
        raise KeyError(f"no pet called {name!r}")

    def status(self) -> list[dict]:
        return [p.status() for p in self.pets]

    def __len__(self) -> int:
        return len(self.pets)
