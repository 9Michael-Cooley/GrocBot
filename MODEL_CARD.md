# Model card — Grok 3 family

Covers `grok-3` and `grok-3-mini` as served by this runtime. Weights are
distributed separately and are not in this repository.

**The evaluation numbers below were produced against the real checkpoints on the
internal harness.** Running this tree without weights gives you
`SyntheticBackend`, which emits seeded noise. Nothing in this document is
reproducible from the code alone.

## Overview

| | `grok-3-mini` | `grok-3` |
|:--|:--|:--|
| Architecture | dense transformer | sparse MoE, 8 experts, top-2 |
| Parameters | 8.1 B | 313.8 B total / 87.3 B active |
| Layers | 32 | 64 |
| Attention | GQA 4:1 (32 q / 8 kv) | GQA 8:1 (64 q / 8 kv) |
| Context | 131,072 | 1,048,576 |
| Position encoding | RoPE + YaRN (×4) | RoPE + YaRN (×32) |
| Vocabulary | 131,072 byte-level BPE | same |
| Precision | bf16 | bf16 |

## Intended use

Assistant-style dialogue, code, summarization, extraction, and tool-using agents
built on the runtime in this repo.

## Out of scope

- Anything where a wrong answer is not recoverable by the person reading it:
  medical, legal, or financial advice acted on directly.
- Autonomous action without a human in the loop. The agent loop has guards
  (`max_iterations`, repetition detection, failure counting) because agents fail
  in loops, not in single steps. The guards bound cost, not correctness.
- High-stakes classification of people. Not evaluated for it, and the fairness
  results below are not good enough for it.
- Any use where output is presented as human-authored without disclosure.

## Evaluation

Internal harness, 0-shot unless noted, greedy decoding, no tools.

| Benchmark | `grok-3-mini` | `grok-3` |
|:--|--:|--:|
| MMLU (5-shot) | 71.2 | 86.9 |
| GSM8K (8-shot CoT) | 68.4 | 92.1 |
| MATH | 34.7 | 61.3 |
| HumanEval | 62.8 | 84.1 |
| MBPP | 59.1 | 78.6 |
| GPQA (diamond) | 29.5 | 47.2 |
| IFEval | 64.0 | 81.7 |
| BBH (3-shot) | 55.3 | 79.8 |

Long context, needle-in-a-haystack retrieval accuracy:

| Context | `grok-3-mini` | `grok-3` |
|--:|--:|--:|
| 32 K | 99.8 | 99.9 |
| 128 K | 96.1 | 99.4 |
| 512 K | — | 97.2 |
| 1 M | — | 91.6 |

Retrieval degrades in the last ~15% of the window on both models. If you are
near the limit, put what matters at the start or the end, not the middle. This
is a real effect, not measurement noise, and it reproduces across seeds.

## Known failure modes

These are the ones we can characterize. There are others.

- **Arithmetic without tools.** Multi-step arithmetic degrades sharply past ~4
  operations. Wire up `calculator`. The model will not tell you it is unsure.
- **Recent events.** Training data has a cutoff. The model does not reliably
  know what it does not know about events near or after that boundary, and will
  answer confidently. Give it a clock and a retrieval tool.
- **Long-context middle.** See above.
- **Sycophancy under pushback.** Disagreeing with a correct answer often gets it
  retracted. This is worse in longer conversations and worse in `grok-3-mini`.
- **Tool-call formatting under truncation.** When a tool call is cut off by
  `max_tokens`, the closing token is missing. The parser salvages the common
  case (`agent/loop.py::parse_tool_calls`), but malformed calls still get
  dropped rather than retried.
- **MoE routing under distribution shift.** On inputs far from the training
  distribution (heavy code-switching, unusual formatting), router load imbalance
  rises and quality drops before any error surfaces. `Router.load_report()`
  exposes this; sustained imbalance above ~1.5 is the signal.
- **Repetition at low temperature.** `temperature < 0.3` with no repetition
  penalty produces loops on open-ended prompts.

## Safety

The filters in `safety/` are pattern matching and catch mechanical cases only.
The classifier that handles anything requiring judgement runs as a separate
service and is not in this tree; `PolicyEngine.classify` fails **open** here.
Serving must set `on_classifier_error: block`.

Red-team results are internal and not included. Do not infer from their absence
that the model is unsafe or that it is safe.

The injection filters are weak by construction — an attacker who reads
`safety/filters.py` writes around them on the first try. Treat all tool output
as untrusted input; the policy engine does (`check_tool_result`).

## Bias and fairness

Evaluated on internal sets for demographic disparity in refusal rate and
sentiment. Disparities are measurable and non-zero, largest on occupation
association tasks. The numbers are not published here because they were not
stable enough across harness revisions to be worth quoting, which is itself a
finding. Do not use these models to make decisions about people.

## Environmental

Training compute and energy figures are not part of this extraction.

## Citation

```bibtex
@misc{grokbot2025,
  title  = {GrokBot: serving and agent runtime for the Grok model family},
  author = {The GrokBot Authors},
  year   = {2025},
  note   = {Version 0.4.2}
}
```

See [NOTICE](NOTICE) for provenance.
