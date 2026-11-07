"""Command groups that import a command's module only when it is used.

``bench --version`` and ``bench --help`` used to import every command module,
and through them rollouts, sandboxes, the ACP schema and the review stack,
about a second before printing anything. A :class:`LazyGroup` lists each
command it holds as a :class:`LazyCommand`: the module, its ``register_*``
function, and what the parent's listing shows (name, first help paragraph,
panel, hidden, deprecated). The module is imported, and the real command
built, the first time anything asks the group for that command: running it,
``bench <command> --help``, shell completion, or a test reading
``group.commands[name]``. Only the group's own help listing reads the listed
facts without importing.

``tests/test_cli_lazy_commands.py`` keeps every listed fact equal to the real
command's, and the help golden (``tools/cli_help_golden.py``) keeps every
``--help`` text byte-identical.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import click
import typer
from typer.core import TyperCommand, TyperGroup

if TYPE_CHECKING:
    from click.exceptions import NoSuchOption, UsageError
    from click.globals import get_current_context
else:
    # typer 0.26 and later run on their own copy of click (``typer._click``):
    # the usage errors their commands raise and the contexts they push are that
    # copy's, so ``except click.UsageError`` misses them and
    # ``click.get_current_context()`` finds none. Take the classes from the
    # click typer runs on; type checking reads click's, which they mirror.
    try:
        _exceptions = importlib.import_module("typer._click.exceptions")
        _globals = importlib.import_module("typer._click.globals")
    except ImportError:  # typer before 0.26 runs on click itself
        _exceptions = importlib.import_module("click.exceptions")
        _globals = importlib.import_module("click.globals")
    NoSuchOption = _exceptions.NoSuchOption
    UsageError = _exceptions.UsageError
    get_current_context = _globals.get_current_context


@dataclass(frozen=True)
class LazyCommand:
    """A command a group lists without importing its module.

    ``register`` is the module's ``register_<group>`` function, called on a
    fresh ``typer.Typer`` the way ``cli/main.py`` used to call it on ``app``;
    the command named ``name`` is taken from what it registered. ``alias``
    registers through ``register(<eval group>, alias_app=<this group>)``, the
    signature of :func:`benchflow.cli.continue_cmd.register_continue`.

    ``source`` is the command's path in what ``register`` built when it is not
    ``name``: ``bench new`` is ``("tasks", "init")`` of ``register_tasks``. The
    built command takes this entry's name, panel and hidden flag, so the
    parent lists and runs it as its layout says (``bench doctor`` is under
    "This machine" whatever panel ``register_doctor`` names).
    """

    name: str
    module: str
    register: str
    help: str
    rich_help_panel: str | None = None
    hidden: bool = False
    deprecated: bool = False
    alias: bool = False
    source: tuple[str, ...] = ()

    def load(self, *, rich_markup_mode: Any, suggest_commands: bool) -> click.Command:
        """Import the module and build the real command, as eager registration did."""
        register = getattr(importlib.import_module(self.module), self.register)
        options: dict[str, Any] = {"rich_markup_mode": rich_markup_mode}
        # ``suggest_commands`` first exists in typer 0.20; pyproject allows older.
        if "suggest_commands" in inspect.signature(typer.Typer).parameters:
            options["suggest_commands"] = suggest_commands
        host = typer.Typer(**options)
        if self.alias:
            register(typer.Typer(), alias_app=host)
        else:
            register(host)
        command: click.Command = typer.main.get_group(host)
        for part in self.source or (self.name,):
            # TyperGroup, not click.Group: from typer 0.26 a typer group is no
            # click.Group (typer runs on its own click).
            assert isinstance(command, TyperGroup), (self.module, self.source)
            command = command.commands[part]
        assert isinstance(command, (TyperCommand, TyperGroup)), (self.module, self.name)
        command.name = self.name
        command.hidden = self.hidden
        command.rich_help_panel = self.rich_help_panel
        return command

    def listed(self) -> click.Command:
        """The stand-in a help listing reads: the listed facts, no callback."""
        return TyperCommand(
            self.name,
            help=self.help,
            hidden=self.hidden,
            deprecated=self.deprecated,
            rich_help_panel=self.rich_help_panel,
        )


def listing_help(command: click.Command) -> str:
    """What a parent's help listing shows for ``command``: its first paragraph.

    The same cut typer's listing makes (``rich_utils._make_command_help``), so a
    :class:`LazyCommand` whose ``help`` equals this lists exactly as the real
    command would.
    """
    return inspect.cleandoc(command.short_help or command.help or "").split("\n\n")[0]


class _Commands(dict[str, click.Command]):
    """A group's commands; a lazy entry becomes the real command when read.

    Name lookups and iteration over names never import. Reading an entry
    (``[name]``, ``get``, ``values``, ``items``) builds it, so code that walks
    ``group.commands`` sees the same objects eager registration gave it. Only
    ``listed`` (the help listing) and raw dict copies see the stand-ins.
    """

    def __init__(
        self,
        commands: Mapping[str, click.Command],
        lazy: Mapping[str, LazyCommand],
        group: LazyGroup,
    ) -> None:
        super().__init__(commands)
        self._lazy = dict(lazy)
        self._group = group

    def _load(self, name: str) -> None:
        spec = self._lazy.get(name)
        if spec is not None:
            command = spec.load(
                rich_markup_mode=self._group.rich_markup_mode,
                suggest_commands=getattr(self._group, "suggest_commands", True),
            )
            # Forget the spec only once the real command is in: a load that
            # raised (a missing dependency) raises again on the next lookup,
            # instead of leaving the stand-in, which has no callback and
            # would run as a silent success.
            super().__setitem__(name, command)
            del self._lazy[name]

    def listed(self, name: str) -> click.Command | None:
        """The entry as a listing shows it (a stand-in until read), importing nothing."""
        return super().get(name)

    def __getitem__(self, name: str) -> click.Command:
        self._load(name)
        return super().__getitem__(name)

    def __setitem__(self, name: str, command: click.Command) -> None:
        self._lazy.pop(name, None)
        super().__setitem__(name, command)

    def get(self, name: str, default: Any = None) -> Any:
        if name in self:
            self._load(name)
        return super().get(name, default)

    def _load_all(self) -> None:
        for name in list(self._lazy):
            self._load(name)

    def values(self):
        self._load_all()
        return super().values()

    def items(self):
        self._load_all()
        return super().items()


class LazyGroup(TyperGroup):
    """A typer group whose ``layout`` lists commands in the order help shows them.

    Each layout entry is a :class:`LazyCommand` or the name of a command
    registered on this group's ``typer.Typer`` the usual way (``@app.command``
    or ``add_typer``). Eager commands the layout does not name follow, in
    registration order.
    """

    layout: ClassVar[tuple[LazyCommand | str, ...]] = ()

    def __init__(self, *, commands: Any = None, **attrs: Any) -> None:
        super().__init__(**attrs)
        eager: dict[str, click.Command] = dict(self.commands)
        if isinstance(commands, Mapping):
            eager.update(commands)
        elif commands:
            eager.update({c.name: c for c in commands if c.name})
        ordered: dict[str, click.Command] = {}
        lazy: dict[str, LazyCommand] = {}
        for entry in self.layout:
            if isinstance(entry, LazyCommand) and entry.name in eager:
                # Registered on the group's Typer under a lazy entry's name (an
                # application replacing ``bench doctor``): it wins, as it did
                # when the built-in registered first and it registered after.
                # Left pending, the first lookup would load the built-in over
                # it.
                ordered[entry.name] = eager.pop(entry.name)
            elif isinstance(entry, LazyCommand):
                lazy[entry.name] = entry
                ordered[entry.name] = entry.listed()
            elif entry in eager:
                ordered[entry] = eager.pop(entry)
        ordered.update(eager)
        self.commands = _Commands(ordered, lazy, self)
        self._listing = False

    def list_commands(self, ctx: click.Context) -> list[str]:
        # Names only: typer's version reads ``commands.items()``, which would
        # import every command to list them.
        return list(self.commands)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        if self._listing:
            commands = self.commands
            assert isinstance(commands, _Commands)
            return commands.listed(cmd_name)
        return super().get_command(ctx, cmd_name)

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        # The listing of this group's commands reads the listed facts; nothing
        # else in its help asks for a command.
        self._listing = True
        try:
            super().format_help(ctx, formatter)
        finally:
            self._listing = False
