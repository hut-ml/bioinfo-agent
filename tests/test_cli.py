from typer.testing import CliRunner

from bioinfo_agent.cli import app


def test_info_command_runs_and_is_ascii():
    result = CliRunner().invoke(app, ["info"])
    assert result.exit_code == 0
    assert "15831 genes x 90 samples" in result.output
    assert "control" in result.output and "treated" in result.output
    # Plain ASCII output avoids encoding errors on legacy Windows consoles.
    result.output.encode("ascii")
