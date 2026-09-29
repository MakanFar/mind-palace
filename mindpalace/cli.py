"""The `mindpalace` command (docs/decisions/0005 §Part 4).

`serve` runs the MCP server; `review`, `adopt`, `merge`, `retire` put
structure-adding decisions to the person at the terminal, through the same
gate the chat surface uses. Every interactive command refuses a stdin that
is not a terminal: an agent with a shell could otherwise pipe in its own yes.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import sys
from pathlib import Path

from mindpalace import gate, tools
from mindpalace.config import ConfigError
from mindpalace.embed import EmbedderError
from mindpalace.session import Session, VaultLockedError

SUBCOMMANDS = ("serve", "review", "adopt", "merge", "retire")
CHOICES = {"c": "confirm", "d": "dismiss", "s": "skip", "q": "stop"}
NOT_A_TERMINAL = (
    "mindpalace: this command needs a person at a terminal, and its input is not "
    "one; nothing was written"
)


class Terminal:
    def __init__(self, stdin=None, stdout=None) -> None:
        self.stdin = stdin or sys.stdin
        self.stdout = stdout or sys.stdout

    def interactive(self) -> bool:
        return self.stdin.isatty()

    def say(self, text: str) -> None:
        self.stdout.write(text + "\n")
        self.stdout.flush()

    def ask(self, prompt: str) -> str | None:
        """One line of input, stripped; None at end of input (Ctrl-D)."""
        self.stdout.write(prompt)
        self.stdout.flush()
        line = self.stdin.readline()
        return None if line == "" else line.strip()


class TerminalAsker:
    """An `Asker` at the terminal."""

    def __init__(self, terminal: Terminal) -> None:
        self.terminal = terminal

    def can_ask(self) -> bool:
        return self.terminal.interactive()

    async def confirm(self, message: str) -> bool:
        self.terminal.say(message)
        answer = self.terminal.ask("Proceed? [y/N] ")
        return answer is not None and answer.lower() in {"y", "yes"}

    async def choose(self, message: str) -> tuple[str, str | None]:
        self.terminal.say("")
        self.terminal.say(message)
        while True:
            answer = self.terminal.ask("[c]onfirm [d]ismiss [s]kip [q]uit: ")
            if answer is None:
                return "stop", None
            decision = CHOICES.get(answer.lower()[:1])
            if decision is not None:
                break
        if decision in {"confirm", "dismiss"}:
            reason = self.terminal.ask("Reason (optional): ")
            return decision, reason or None
        return decision, None


def _direct(session: Session):
    async def call(fn):
        return fn(session)
    return call


def _refuse_without_terminal(terminal: Terminal) -> bool:
    if terminal.interactive():
        return False
    terminal.say(NOT_A_TERMINAL)
    return True


def review_command(session: Session, terminal: Terminal, limit: int = 20) -> int:
    if _refuse_without_terminal(terminal):
        return 2
    result = asyncio.run(gate.run_review(
        TerminalAsker(terminal), _direct(session), via="cli", limit=limit
    ))
    if result["pending"] == 0:
        terminal.say("Nothing is waiting for review.")
    elif result["reviewed"]:
        terminal.say(
            f"{result['confirmed']} confirmed, {result['dismissed']} dismissed, "
            f"{result['skipped']} skipped."
        )
        if result["already_decided"]:
            terminal.say(f"Already decided elsewhere: {', '.join(result['already_decided'])}")
    return 0


def _approve_command(session: Session, terminal: Terminal, describe, act) -> int:
    if _refuse_without_terminal(terminal):
        return 2
    try:
        asyncio.run(gate.approve(TerminalAsker(terminal), _direct(session), describe, act))
    except tools.ToolError as exc:
        terminal.say(f"mindpalace: {exc}")
        return 1
    terminal.say("Done.")
    return 0


def adopt_command(session: Session, terminal: Terminal, **args) -> int:
    return _approve_command(
        session, terminal,
        functools.partial(tools.describe_adopt_type, **args),
        functools.partial(tools.adopt_type, **args, via="cli"),
    )


def merge_command(session: Session, terminal: Terminal, **args) -> int:
    return _approve_command(
        session, terminal,
        functools.partial(tools.describe_merge_entities, **args),
        functools.partial(tools.merge_entities, **args, via="cli"),
    )


def retire_command(session: Session, terminal: Terminal, **args) -> int:
    return _approve_command(
        session, terminal,
        functools.partial(tools.describe_retire_entity, **args),
        functools.partial(tools.retire_entity, **args, via="cli"),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mindpalace")
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, help: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help)
        sub.add_argument("--vault", required=True, help="path to the vault directory")
        return sub

    serve = command("serve", "run the MCP server")
    serve.add_argument("--init", action="store_true", help="scaffold a new vault")
    serve.add_argument("--check", action="store_true", help="open and validate, then exit")

    review = command("review", "confirm or dismiss pending proposals")
    review.add_argument("--limit", type=int, default=20)

    adopt = command("adopt", "adopt a proposed wording as a type")
    adopt.add_argument("kind", choices=["edge", "entity"])
    adopt.add_argument("proposed")
    adopt.add_argument("name")
    shape = adopt.add_mutually_exclusive_group()
    shape.add_argument("--directed", dest="directed", action="store_true", default=None)
    shape.add_argument("--symmetric", dest="directed", action="store_false")
    adopt.add_argument("--cluster-weight", type=float, default=1.0)
    adopt.add_argument("--domain", nargs="*")
    adopt.add_argument("--range", nargs="*")
    adopt.add_argument("--revoke", action="store_const", const="revoke", default="adopt", dest="action")

    merge = command("merge", "merge a duplicate entity into a canonical one")
    merge.add_argument("duplicate")
    merge.add_argument("canonical")
    how = merge.add_mutually_exclusive_group()
    how.add_argument("--unmerge", action="store_const", const="unmerge", dest="action")
    how.add_argument("--keep", action="store_const", const="keep", dest="action")
    merge.set_defaults(action="merge")
    merge.add_argument("--reason")

    retire = command("retire", "retire a slug that is not an entity")
    retire.add_argument("slug")
    retire.add_argument("--restore", action="store_const", const="restore", default="retire", dest="action")
    retire.add_argument("--reason")
    return parser


def _serve(args) -> int:
    from mindpalace.server import build_server

    try:
        session = Session(Path(args.vault), init=args.init).open()
    except (ConfigError, VaultLockedError, EmbedderError) as exc:
        # EmbedderError included so a misconfigured embedder is a legible message
        # rather than a traceback out of the MCP entry point.
        print(str(exc), file=sys.stderr)
        return 2
    try:
        if args.check:
            return 0
        build_server(session).run()
        return 0
    finally:
        session.close()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # The bare `mindpalace --vault X` form predates subcommands and is in
    # every existing MCP config: it keeps meaning `serve`.
    if not argv or argv[0] not in SUBCOMMANDS:
        argv = ["serve", *argv]
    args = _parser().parse_args(argv)
    if args.command == "serve":
        return _serve(args)

    try:
        session = Session(Path(args.vault)).open()
    except (ConfigError, VaultLockedError, EmbedderError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    terminal = Terminal()
    try:
        if args.command == "review":
            return review_command(session, terminal, limit=args.limit)
        if args.command == "adopt":
            return adopt_command(
                session, terminal, kind=args.kind, proposed=args.proposed, name=args.name,
                directed=args.directed, cluster_weight=args.cluster_weight,
                domain=args.domain, range=args.range, action=args.action,
            )
        if args.command == "merge":
            return merge_command(
                session, terminal, duplicate=args.duplicate, canonical=args.canonical,
                action=args.action, reason=args.reason,
            )
        return retire_command(session, terminal, slug=args.slug, action=args.action, reason=args.reason)
    finally:
        session.close()
