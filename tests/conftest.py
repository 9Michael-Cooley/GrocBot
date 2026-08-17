from pathlib import Path

import pytest

from grokbot.config import Config
from grokbot.inference.engine import Engine
from grokbot.tokenizer import Tokenizer

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"


@pytest.fixture(scope="session")
def mini_config_path() -> Path:
    return CONFIGS / "grok-3-mini.yaml"


@pytest.fixture(scope="session")
def mini_config(mini_config_path) -> Config:
    return Config.load(mini_config_path)


@pytest.fixture(scope="session")
def tokenizer() -> Tokenizer:
    # Small vocab: building 32k merges per session is dead time in CI.
    return Tokenizer.synthetic(vocab_size=4096, seed=0)


@pytest.fixture
def engine(mini_config_path) -> Engine:
    return Engine.from_config(mini_config_path)
