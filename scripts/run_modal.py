from cs336_alignment.modal_utils import app, submit_commands


@app.local_entrypoint()
def main(*argv: str) -> None:
    command = [
        "python",
        "-u",
        "cs336_alignment/prompting_baselines.py",
        *argv,
    ]

    submit_commands([command])