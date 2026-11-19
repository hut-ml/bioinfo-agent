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


def test_precompute_writes_results_that_the_other_commands_read(
    synthetic_data, tmp_path, monkeypatch
):
    monkeypatch.setenv("BIOINFO_DATA_DIR", str(synthetic_data))
    monkeypatch.setenv("BIOINFO_RESULTS_DIR", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(app, ["precompute"])
    assert result.exit_code == 0, result.output
    assert "16 up, 4 down" in result.output
    assert (tmp_path / "de_results.csv").exists()
    assert "UP0: differentially expressed: higher in treated" in (
        runner.invoke(app, ["gene", "UP0"]).output
    )
    assert "gene" in runner.invoke(app, ["lane"]).output


def test_dotenv_settings_reach_every_command(synthetic_data, monkeypatch):
    def fake_load_dotenv(*args, **kwargs):
        monkeypatch.setenv("BIOINFO_DATA_DIR", str(synthetic_data))

    monkeypatch.setattr("bioinfo_agent.cli.load_dotenv", fake_load_dotenv)
    result = CliRunner().invoke(app, ["info"])
    assert "150 genes x 20 samples" in result.output
