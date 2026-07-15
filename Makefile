.PHONY: install api ui test lint fmt record

install:
	uv sync --extra dev --extra ui

api:
	uv run uvicorn app.main:app --reload

ui:
	uv run streamlit run streamlit_app/app.py

test:
	uv run pytest --cov=app --cov-report=term-missing

lint:
	uv run ruff check app tests && uv run black --check app tests

fmt:
	uv run black app tests && uv run ruff check --fix app tests

record:
	uv sync --extra record && uv run python samples/audio_creation.py
