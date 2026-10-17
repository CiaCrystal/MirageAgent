"""Build the current target's knowledge index through the public CLI."""

from migration_agent.cli import app

if __name__ == "__main__":
    app(["index", *__import__("sys").argv[1:]])
