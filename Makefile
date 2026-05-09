.PHONY: install test run clean

install:
	uv venv .venv
	uv pip install -r requirements.txt

test:
	.venv/bin/python -m pytest tests/ -v

run:
	bash run.sh

clean:
	rm -rf .venv results/raw/* results/reports/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
