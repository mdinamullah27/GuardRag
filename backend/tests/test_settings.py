from app.config import persist_env


def test_persist_env_updates_and_preserves(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("DATABASE_URL=postgresql+psycopg://x\nLLM_API_KEY=old\nLLM_MODEL=gpt-4o-mini\n")
    monkeypatch.setattr("app.config.env_file_path", lambda: env_file)

    persist_env({"llm_api_key": "new-key", "llm_base_url": "https://api.x.ai/v1"})

    content = env_file.read_text()
    assert "LLM_API_KEY=new-key" in content
    assert "LLM_BASE_URL=https://api.x.ai/v1" in content
    assert "LLM_MODEL=gpt-4o-mini" in content
    assert "DATABASE_URL=postgresql+psycopg://x" in content
    assert "old" not in content


def test_persist_env_creates_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    monkeypatch.setattr("app.config.env_file_path", lambda: env_file)

    persist_env({"llm_model": "grok-4"})

    assert env_file.read_text().strip() == "LLM_MODEL=grok-4"
