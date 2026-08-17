.PHONY: setup test lint fmt bench serve chat clean docker

PY ?= python

setup:
	$(PY) -m pip install -e ".[all]"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests
	$(PY) -m mypy src/grokbot

fmt:
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

bench:
	$(PY) benchmarks/bench_decode.py --config configs/grok-3-mini.yaml --requests 128

serve:
	$(PY) -m grokbot serve --config configs/serving.yaml

chat:
	$(PY) -m grokbot chat --config configs/grok-3-mini.yaml

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +

docker:
	docker build -t grokbot:0.4.2 .
