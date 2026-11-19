import pytest
from typer.testing import CliRunner

from bioinfo_agent.cli import app


@pytest.fixture(autouse=True)
def ignore_dotenv(monkeypatch):
    # A developer's .env must not change results (load_dotenv also writes into os.environ).
    monkeypatch.setattr("bioinfo_agent.cli.load_dotenv", lambda *args, **kwargs: False)


def test_info_command_runs_and_is_ascii():
    result = CliRunner().invoke(app, ["info"])
    assert result.exit_code == 0
    assert "15831 genes x 90 samples" in result.output
    assert "control" in result.output and "treated" in result.output
    # Plain ASCII output avoids encoding errors on legacy Windows consoles.
    result.output.encode("ascii")


def test_ask_offline_prints_answer_and_tool_calls():
    result = CliRunner().invoke(
        app, ["ask", "Is STAT1 differentially expressed?", "--backend", "offline"]
    )
    assert result.exit_code == 0
    assert "STAT1: not differentially expressed" in result.output
    assert "get_genes(names=['STAT1'])" in result.output


def test_ask_with_missing_settings_fails_cleanly(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = CliRunner().invoke(app, ["ask", "hi", "--backend", "openai"])
    assert result.exit_code == 1
