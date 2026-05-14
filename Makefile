.PHONY: install test lint demo api

install:
	python3.12 -m venv .venv
	.venv/bin/pip install -r requirements.txt

test:
	.venv/bin/python -m pytest -q

lint:
	.venv/bin/ruff check .

demo:
	.venv/bin/python main.py demo

api:
	.venv/bin/python main.py api --host 127.0.0.1 --port 8000
