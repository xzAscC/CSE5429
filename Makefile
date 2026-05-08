.PHONY: install test run clean

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

run:
	bash scripts/run_all.sh

clean:
	rm -rf results/raw/* results/reports/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
