from datetime import datetime

import pytest

from grokbot.agent.presets import (
    PRESETS,
    BotPreset,
    describe_all,
    get_preset,
    list_presets,
    register_preset,
)

# -- presets ----------------------------------------------------------------


def test_all_presets_build_valid_sampling_configs():
    for name in list_presets(include_hidden=True):
        gen = PRESETS[name].generation_config()
        assert gen.max_tokens > 0
        assert 0.0 <= gen.temperature
        assert 0.0 < gen.top_p <= 1.0


def test_preset_overrides_win():
    gen = get_preset("code").generation_config(temperature=0.0, max_tokens=64)
    assert gen.temperature == 0.0
    assert gen.max_tokens == 64
    assert gen.top_p == get_preset("code").top_p     # untouched fields survive


def test_none_overrides_are_ignored():
    """The CLI passes unset flags as None; they must not clobber the preset."""
    preset = get_preset("deep")
    gen = preset.generation_config(temperature=None, max_tokens=None)
    assert gen.temperature == preset.temperature
    assert gen.max_tokens == preset.max_tokens


def test_eval_preset_is_greedy_and_hidden():
    assert get_preset("eval").temperature == 0.0
    assert "eval" not in list_presets()
    assert "eval" in list_presets(include_hidden=True)


def test_code_preset_disables_repetition_penalty():
    """Penalising repetition mangles boilerplate."""
    assert get_preset("code").repetition_penalty == 1.0


def test_fast_preset_is_actually_smaller():
    fast, deep = get_preset("fast"), get_preset("deep")
    assert fast.max_tokens < deep.max_tokens
    assert fast.memory_tokens < deep.memory_tokens
    assert fast.max_iterations < deep.max_iterations


def test_unknown_preset_lists_alternatives():
    with pytest.raises(ValueError, match="unknown preset"):
        get_preset("nope")


def test_presets_resolve_to_real_personas():
    for name in list_presets(include_hidden=True):
        assert PRESETS[name].get_persona() is not None


def test_system_prompt_renders():
    prompt = get_preset("research").system_prompt()
    assert "Grok" in prompt


def test_register_preset_requires_replace():
    custom = BotPreset(name="unit-test-preset", description="x")
    register_preset(custom)
    try:
        with pytest.raises(ValueError, match="already exists"):
            register_preset(BotPreset(name="unit-test-preset", description="y"))
        register_preset(BotPreset(name="unit-test-preset", description="y"), replace=True)
        assert get_preset("unit-test-preset").description == "y"
    finally:
        PRESETS.pop("unit-test-preset", None)


def test_build_agent_applies_preset(engine):
    agent = get_preset("fast").build_agent(engine)
    assert agent.max_iterations == get_preset("fast").max_iterations
    assert agent.memory.max_tokens == get_preset("fast").memory_tokens


def test_describe_all_covers_visible_presets():
    text = describe_all()
    for name in list_presets():
        assert name in text


# -- pets -------------------------------------------------------------------


@pytest.fixture
def pets_enabled(monkeypatch):
    monkeypatch.setenv("GROKBOT_ENABLE_PETS", "1")
    import grokbot.experimental.pets as pets_module

    return pets_module


def test_pets_disabled_by_default(monkeypatch):
    monkeypatch.delenv("GROKBOT_ENABLE_PETS", raising=False)
    from grokbot.experimental.pets import Pet, PetsDisabled

    with pytest.raises(PetsDisabled, match="unreleased"):
        Pet.create("dog")


def test_default_names(pets_enabled):
    assert pets_enabled.Pet.create("dog").name == "Odie"
    assert pets_enabled.Pet.create("cat").name == "Garfield"


def test_custom_name_overrides_default(pets_enabled):
    assert pets_enabled.Pet.create("cat", "Nermal").name == "Nermal"


def test_blank_name_falls_back_to_default(pets_enabled):
    assert pets_enabled.Pet.create("dog", "   ").name == "Odie"


def test_overlong_name_rejected(pets_enabled):
    with pytest.raises(ValueError, match="limit is 32"):
        pets_enabled.Pet.create("dog", "x" * 33)


def test_unknown_species_rejected(pets_enabled):
    with pytest.raises(ValueError, match="unknown species"):
        pets_enabled.Pet.create("dragon")


def test_bird_is_gated_out_of_the_release(pets_enabled):
    assert pets_enabled.available_species() == ["cat", "dog"]
    assert "bird" in pets_enabled.available_species(include_unreleased=True)


def test_feeding_reduces_hunger_and_raises_affection(pets_enabled):
    pet = pets_enabled.Pet.create("dog")
    pet.state.hunger, pet.state.affection = 80.0, 40.0
    pet.feed("kibble")
    assert pet.state.hunger < 80.0
    assert pet.state.affection > 40.0


def test_favourite_food_is_worth_more(pets_enabled):
    plain = pets_enabled.Pet.create("cat")
    fancy = pets_enabled.Pet.create("cat")
    for p in (plain, fancy):
        p.state.hunger, p.state.affection = 60.0, 30.0
    plain.feed("dry food")
    fancy.feed("lasagna")
    assert fancy.state.affection > plain.state.affection
    assert fancy.state.hunger < plain.state.hunger


def test_play_refused_when_too_tired(pets_enabled):
    pet = pets_enabled.Pet.create("dog")
    pet.state.energy = 1.0
    assert "declines" in pet.play("fetch")
    # Refusing must not spend the play cost. Approximate because refresh() still
    # applies elapsed-time decay, which is correct — the clock doesn't stop.
    assert pet.state.energy == pytest.approx(1.0, abs=0.05)


def test_play_costs_energy_and_earns_affection(pets_enabled):
    pet = pets_enabled.Pet.create("dog")
    pet.state.energy, pet.state.affection = 90.0, 40.0
    pet.play("fetch")
    assert pet.state.energy < 90.0
    assert pet.state.affection > 40.0


def test_state_is_clamped(pets_enabled):
    pet = pets_enabled.Pet.create("dog")
    for _ in range(20):
        pet.feed()
        pet.pet()
    assert 0.0 <= pet.state.hunger <= 100.0
    assert 0.0 <= pet.state.affection <= 100.0
    assert 0.0 <= pet.state.energy <= 100.0


def test_decay_moves_state_in_the_right_direction(pets_enabled):
    pet = pets_enabled.Pet.create("cat")
    before = pet.state.as_dict()
    pet.state.apply_decay(pet.species, hours=4.0)
    assert pet.state.hunger > before["hunger"]
    assert pet.state.energy < before["energy"]
    assert pet.state.affection < before["affection"]


def test_cat_is_more_demanding_than_dog(pets_enabled):
    """The species differ in more than prompt text."""
    cat, dog = pets_enabled.CAT, pets_enabled.DOG
    assert cat.hunger_rate > dog.hunger_rate
    assert cat.affection_decay > dog.affection_decay
    assert cat.feed_affection_gain > dog.feed_affection_gain
    assert cat.play_affection_gain < dog.play_affection_gain


@pytest.mark.parametrize(
    "hunger,energy,affection,expected",
    [
        (90.0, 80.0, 50.0, "starving"),
        (30.0, 10.0, 50.0, "exhausted"),
        (60.0, 60.0, 50.0, "hungry"),
        (30.0, 60.0, 90.0, "devoted"),
        (30.0, 60.0, 10.0, "aloof"),
        (30.0, 90.0, 50.0, "playful"),
        (30.0, 50.0, 50.0, "content"),
    ],
)
def test_mood_resolution_order(pets_enabled, hunger, energy, affection, expected):
    pet = pets_enabled.Pet.create("dog")
    pet.state.hunger, pet.state.energy, pet.state.affection = hunger, energy, affection
    assert pet.state.mood(pet.species) == expected


def test_cat_hates_mondays(pets_enabled):
    monday = datetime(2026, 8, 17)      # a Monday
    tuesday = datetime(2026, 8, 18)
    cat = pets_enabled.Pet.create("cat")
    dog = pets_enabled.Pet.create("dog")
    for p in (cat, dog):
        p.state.hunger, p.state.energy, p.state.affection = 30.0, 50.0, 50.0

    assert cat.state.mood(cat.species, now=monday) == "monday"
    assert cat.state.mood(cat.species, now=tuesday) == "content"
    assert dog.state.mood(dog.species, now=monday) == "content"   # dog does not care


def test_mood_changes_the_system_prompt(pets_enabled):
    pet = pets_enabled.Pet.create("dog")
    pet.state.hunger, pet.state.energy, pet.state.affection = 95.0, 80.0, 50.0
    starving = pet.persona().render()
    pet.state.hunger = 10.0
    fed = pet.persona().render()
    assert starving != fed
    assert "hungry" in starving.lower()


def test_persona_carries_the_pets_name(pets_enabled):
    assert "Garfield" in pets_enabled.Pet.create("cat").persona().render()


def test_status_line_is_renderable(pets_enabled):
    line = pets_enabled.Pet.create("cat").status_line()
    assert "Garfield" in line and "cat" in line


def test_history_is_bounded(pets_enabled):
    pet = pets_enabled.Pet.create("dog")
    for _ in range(80):
        pet.pet()
    assert len(pet.history) <= 50


def test_petbot_prompt_includes_persona_and_history(pets_enabled, engine):
    pet = pets_enabled.Pet.create("cat")
    bot = pets_enabled.PetBot(engine, pet)
    pet.feed("lasagna")
    messages = bot.messages("hello")
    assert messages[0]["role"] == "system"
    assert "Garfield" in messages[0]["content"]
    assert any("Recently:" in m["content"] for m in messages if m["role"] == "system")
    assert messages[-1] == {"role": "user", "content": "hello"}


def test_petbot_generates(pets_enabled, engine):
    bot = pets_enabled.PetBot(engine, pets_enabled.Pet.create("dog"), max_tokens=16)
    completion = bot.say("who's a good boy?")
    assert completion.text
    assert len(bot.turns) == 2


def test_household_enforces_limit_and_unique_names(pets_enabled):
    house = pets_enabled.Household(limit=2)
    house.adopt("dog")
    with pytest.raises(ValueError, match="already a pet called"):
        house.adopt("dog")
    house.adopt("cat")
    with pytest.raises(ValueError, match="household is full"):
        house.adopt("cat", "Nermal")
    assert len(house) == 2
    assert house.get("garfield").species.key == "cat"    # lookup is case-insensitive
