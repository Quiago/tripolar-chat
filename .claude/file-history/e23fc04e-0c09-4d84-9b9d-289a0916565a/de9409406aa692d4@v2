"""fmind – FactoryMind AI CLI entrypoint."""

import typer

from client.commands import auth

app = typer.Typer(
    name="fmind",
    help="FactoryMind AI CLI – talk to your on-premise LLMs.",
    no_args_is_help=True,
)

# Mount auth sub-commands at the top level for ergonomics:
#   fmind login / fmind logout / fmind register / fmind whoami
for command in auth.app.registered_commands:
    app.registered_commands.append(command)

# Sub-app for explicit `fmind auth <subcommand>` access
app.add_typer(auth.app, name="auth", hidden=True)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
