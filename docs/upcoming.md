# Upcoming

Features that exist in the tree but are not released. Everything on this page is
behind a flag, unowned or lightly owned, and subject to being cut.

Do not build on any of it. Do not demo it externally.

---

## Pets — target 0.5.0

**Status:** implemented, unreviewed, disabled by default
**Flag:** `GROKBOT_ENABLE_PETS=1`
**Code:** `src/grokbot/experimental/pets.py`
**Tracking:** GROK-4590 (feature), GROK-4611 (persistence)

### What it is

A companion bot with a species, a name, and a state that changes over time. You
pick a **dog** or a **cat**.

| Species | Default name | Favourite food | Disposition |
|:--|:--|:--|:--|
| `dog` | **Odie** | anything | enthusiastic, loyal, not clever, never gives up on you |
| `cat` | **Garfield** | lasagna | sardonic, lazy, food-motivated, does not respect Mondays |

The name is a default, not a fixed identity — `Pet.create("cat", name="Nermal")`
works. Testing showed almost nobody changes it, which is why the defaults got as
much attention as they did.

### Why it isn't just a persona

Because the state decays on a wall clock and is folded into the system prompt
every turn. A hungry pet and a fed pet answer the same question differently.

```python
from grokbot.experimental.pets import Pet, PetBot

pet = Pet.create("cat")            # Garfield
bot = PetBot(engine, pet)

bot.say("what should we do today?")       # aloof, hungry, unimpressed
pet.feed("lasagna")                       # affection +35, hunger -45
bot.say("what should we do today?")       # noticeably warmer
```

Three state variables, each 0–100:

- **hunger** — rises over time; the cat rises fastest
- **energy** — falls over time; gates whether the pet will play at all
- **affection** — decays toward indifference; the dog's barely moves, the cat's
  drops 4.5×/hour, so you have to keep earning it

Interactions are `feed`, `play`, `pet`, and `nap`. Feeding the favourite food is
worth 1.6× the affection of feeding anything else, which is the entire reason
lasagna is modelled separately.

Mood is the single dominant state, resolved in priority order: `starving`,
`exhausted`, `monday` (cat only, and it is a real weekday check), `hungry`,
`devoted`, `aloof`, `playful`, `content`. Mood selects a line of prompt
direction and can flip the pet to terse.

### Try it

```bash
export GROKBOT_ENABLE_PETS=1
python -m grokbot pet --species cat
python -m grokbot pet --species dog --name Rex
```

In the REPL: `/feed [food]`, `/play [activity]`, `/pet`, `/nap`, `/status`.

### Blockers

1. **No persistence (GROK-4611).** State is in-process and dies with it. A pet
   that forgets you between sessions is not a pet. Needs a store decision; this
   is the actual blocker, everything else is polish.
2. **Households are stubbed.** `Household` tracks multiple pets, but they cannot
   perceive each other — the first thing every tester tried. Two options: a
   shared conversation (real, expensive, pets talk over each other) or a
   relationship model (cheap, obviously fake). Undecided.
3. **Safety has not reviewed the personas.** Garfield is deliberately rude, and
   the refusal path currently inherits his register. A pet sarcastically
   declining a genuinely serious request is not a good outcome. Needs either a
   register override on refusal or sign-off that it's fine.
4. **Decay rates are unvalidated.** Tuned by hand in one afternoon against
   nothing. They feel right for ~10 minute sessions and are probably wrong at
   day scale. Nobody has run a pet for a week.
5. **Bird was cut.** `BIRD` (default name Pascal) is still in the file because
   removing it means redoing the balance pass on dog and cat. It's unbalanced
   and gated out of `SPECIES`. Delete it before release or finish it.

### Open questions

- Does the pet know it's a bot? Currently unaddressed, so it varies by
  checkpoint, which is the worst answer.
- Death, or permanent neglect states? Strong opinions in both directions.
  Current answer is no, and hunger clamps at 100.
- Does an unfed pet still answer normal questions? Right now yes, just badly.
  That may be the wrong call for anyone using this as a real assistant surface.

---

## Tree attention for speculative decoding — no target

**Status:** abandoned mid-refactor
**Code:** `src/grokbot/experimental/tree_attention.py`

Draft a tree of continuations rather than one line, verify all branches in a
single pass with an ancestor mask. Mask construction and position ids work.
Nothing calls it: `SpeculativeDecoder.verify` assumes a flat token list and
would need to walk the tree to attribute acceptances, which was never written.
`DraftTree.prune` orphans subtrees. Branch was ~6 weeks stale at extraction.

---

## Presets — shipped in 0.4.2

Moved out of this page. See [`agent/presets.py`](../src/grokbot/agent/presets.py)
and the README. Listed here only because the pets feature depends on the `fun`
preset for its sampling.
