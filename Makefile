install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb src/__main__.py

clean:
	rm -rf __pycache__ src/__pycache__ .mypy_cache .pytest_cache

lint:
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 .
	mypy . --strict

.PHONY: install run debug clean lint lint-strict