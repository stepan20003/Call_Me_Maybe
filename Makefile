install:
	pip install pydantic numpy flake8 mypy

run:
	python3 -m src

debug:
	python3 -m pdb src/__main__.py

clean:
	rm -rf __pycache__ src/__pycache__ .mypy_cache

lint:
	flake8 src/
	mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs src/

.PHONY: install run debug clean lint