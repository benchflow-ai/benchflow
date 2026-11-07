"""Parse verifier installer argv without importing or executing its tools.

Option arity and spelling live here, shared by shell, Python AST and recovery
callers. pip uses optparse (including unique long prefixes); uv uses clap
(unique long prefixes and optional '=' on short values). Python stops parsing
interpreter flags at -c, -m, a script, or --. A value is never parsed again as
an option. These tables describe pip's install/download/wheel and uv's
pip/add/run grammars; unknown options cannot grant additional path precision.
"""

import re
from dataclasses import dataclass


class UnknownDependencyInputs(ValueError):
    """The command is outside the grammar used to grant install provenance."""


def registry_requirement(value: str) -> bool:
    """A closed registry-only subset, never an archive filename (V2-R12-CONF-1).

    Unversioned names have no dots. Versioned names may have dots, but the
    comparison must end in a numeric release/prerelease or wildcard, never
    an archive suffix. This also rules out future dot-extension formats by
    construction, rather than maintaining a list of today's archive suffixes.
    The same subset applies to argv and protected requirement-file contents.
    """
    extras = r"(?:\[[A-Za-z0-9_,.-]+\])?"
    return bool(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*" + extras, value)
        or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*"
            + extras
            + r"(?:==|!=|>=|<=|~=|>|<)(?:[0-9]+!)?v?[0-9]+(?:\.[0-9]+)*"
            r"(?:(?:a|b|rc|\.?post|\.?dev)[0-9]+)*(?:\.\*)?",
            value,
        )
    )


@dataclass(frozen=True)
class PythonInvocation:
    kind: str
    value: str
    index: int
    rest: int


def python_invocation(argv: list[str]) -> PythonInvocation | None:
    """Parse CPython argv after its executable, including bundled short flags."""
    index = 0
    while index < len(argv):
        word = argv[index]
        if word == "--" or not word.startswith("-") or word == "-":
            return None
        if word == "--check-hash-based-pycs":
            index += 2
            continue
        if word.startswith("--"):
            return None
        offset = 1
        while offset < len(word):
            flag = word[offset]
            if flag in "cmWX":
                value = word[offset + 1 :]
                value_index = index
                if not value:
                    index += 1
                    if index >= len(argv):
                        return None
                    value = argv[index]
                    value_index = index
                if flag in "cm":
                    return PythonInvocation(flag, value, value_index, index + 1)
                break
            if flag not in "bBdEiIOPqRsSuvx":
                return None
            offset += 1
        index += 1
    return None


@dataclass(frozen=True)
class Grammar:
    values: frozenset[str]
    flags: frozenset[str]
    short: dict[str, str]
    inputs: frozenset[str]
    short_equals: bool = False


def _names(text: str) -> frozenset[str]:
    return frozenset("--" + name for name in text.split())


_PIP = Grammar(
    values=_names("""
        requirement constraint build-constraint editable target platform
        python-version implementation abi root prefix src source source-dir
        source-directory upgrade-strategy config-settings global-option
        install-option no-binary only-binary progress-bar root-user-action
        report index-url pypi-url extra-index-url find-links python log log-file
        local-log keyring-provider proxy retries timeout default-timeout
        exists-action trusted-host cert client-cert cache-dir use-feature
        use-deprecated destination dest wheel-dir group resume-retries
    """),
    flags=_names("""
        no-deps no-dependencies pre dry-run user no-user upgrade force-reinstall
        ignore-installed ignore-requires-python no-build-isolation use-pep517
        no-use-pep517 check-build-dependencies break-system-packages compile
        no-compile no-warn-script-location no-warn-conflicts prefer-binary
        require-hashes no-clean no-index help debug isolated require-virtualenv
        require-venv verbose version quiet no-input no-cache-dir
        disable-pip-version-check no-color no-python-version-warning
    """),
    short=dict(
        r="--requirement",
        c="--constraint",
        e="--editable",
        t="--target",
        U="--upgrade",
        I="--ignore-installed",
        C="--config-settings",
        i="--index-url",
        f="--find-links",
        h="--help",
        v="--verbose",
        V="--version",
        q="--quiet",
        d="--dest",
        w="--wheel-dir",
    ),
    inputs=_names("requirement constraint build-constraint"),
)

# uv 0.13.1 option arities, from its command help. Each subcommand has
# its own schema: add/run flags differ from pip flags of the same name.
_UV = {
    "": Grammar(
        values=_names("""
            allow-insecure-host cache-dir color config-file directory project
        """),
        flags=_names("""
            help managed-python no-cache no-config no-managed-python no-progress
            no-python-downloads offline quiet system-certs verbose version
        """),
        short={
            "n": "--no-cache",
            "q": "--quiet",
            "v": "--verbose",
            "h": "--help",
            "V": "--version",
        },
        inputs=_names("""
        """),
        short_equals=True,
    ),
    "pip install": Grammar(
        values=_names("""
            allow-insecure-host build-constraints cache-dir cert color config-file
            config-setting config-settings-package constraints default-index
            directory editable exclude-newer exclude-newer-package excludes extra
            extra-index-url find-links fork-strategy group index index-strategy
            index-url keyring-provider link-mode no-binary no-build-isolation-package
            no-editable-package no-sources-package only-binary output-format
            overrides prefix prerelease prerelease-package project python
            python-platform python-version refresh-package reinstall-package
            requirements resolution target torch-backend upgrade-group
            upgrade-package
        """),
        flags=_names("""
            all-extras break-system-packages check compile-bytecode dry-run exact
            help managed-python no-break-system-packages no-build no-build-isolation
            no-cache no-config no-deps no-editable no-index no-managed-python
            no-progress no-python-downloads no-sources no-verify-hashes offline quiet
            refresh reinstall require-build-hashes require-hashes strict system
            system-certs upgrade user verbose
        """),
        short={
            "r": "--requirements",
            "e": "--editable",
            "c": "--constraints",
            "b": "--build-constraints",
            "t": "--target",
            "i": "--index-url",
            "f": "--find-links",
            "U": "--upgrade",
            "P": "--upgrade-package",
            "C": "--config-setting",
            "n": "--no-cache",
            "p": "--python",
            "q": "--quiet",
            "v": "--verbose",
            "h": "--help",
        },
        inputs=_names("""
            build-constraints constraints excludes overrides requirements
        """),
        short_equals=True,
    ),
    "pip sync": Grammar(
        values=_names("""
            allow-insecure-host build-constraints cache-dir cert color config-file
            config-setting config-settings-package constraints default-index
            directory exclude-newer exclude-newer-package extra extra-index-url
            find-links group index index-strategy index-url keyring-provider
            link-mode no-binary no-sources-package only-binary output-format prefix
            project python python-platform python-version refresh-package
            reinstall-package target torch-backend
        """),
        flags=_names("""
            all-extras allow-empty-requirements break-system-packages check
            compile-bytecode dry-run help managed-python no-allow-empty-requirements
            no-break-system-packages no-build no-build-isolation no-cache no-index
            no-managed-python no-progress no-python-downloads no-sources
            no-verify-hashes offline quiet refresh reinstall require-build-hashes
            require-hashes strict system system-certs verbose
        """),
        short={
            "c": "--constraints",
            "b": "--build-constraints",
            "t": "--target",
            "i": "--index-url",
            "f": "--find-links",
            "C": "--config-setting",
            "n": "--no-cache",
            "p": "--python",
            "q": "--quiet",
            "v": "--verbose",
            "h": "--help",
        },
        inputs=_names("""
            build-constraints constraints
        """),
        short_equals=True,
    ),
    "pip compile": Grammar(
        values=_names("""
            allow-insecure-host annotation-style build-constraints cache-dir cert
            color config-file config-setting config-settings-package constraints
            custom-compile-command default-index directory exclude-newer
            exclude-newer-package excludes extra extra-index-url find-links
            fork-strategy format group index index-strategy index-url
            keyring-provider link-mode no-binary no-build-isolation-package
            no-emit-package no-sources-package only-binary output-file overrides
            prerelease prerelease-package project python python-platform
            python-version refresh-package resolution torch-backend upgrade-group
            upgrade-package
        """),
        flags=_names("""
            all-extras emit-build-options emit-find-links emit-index-annotation
            emit-index-url generate-hashes help managed-python no-annotate no-build
            no-build-isolation no-cache no-deps no-header no-index no-managed-python
            no-progress no-python-downloads no-sources no-strip-extras
            no-strip-markers offline quiet refresh require-build-hashes system
            system-certs universal upgrade verbose
        """),
        short={
            "c": "--constraints",
            "b": "--build-constraints",
            "o": "--output-file",
            "i": "--index-url",
            "f": "--find-links",
            "U": "--upgrade",
            "P": "--upgrade-package",
            "C": "--config-setting",
            "n": "--no-cache",
            "p": "--python",
            "q": "--quiet",
            "v": "--verbose",
            "h": "--help",
        },
        inputs=_names("""
            build-constraints constraints excludes overrides
        """),
        short_equals=True,
    ),
    "add": Grammar(
        values=_names("""
            allow-insecure-host bounds branch cache-dir color config-file
            config-setting config-settings-package constraints default-index
            directory exclude-newer exclude-newer-package extra extra-index-url
            find-links fork-strategy group index index-strategy index-url
            keyring-provider link-mode marker no-binary-package
            no-build-isolation-package no-build-package no-install-package
            no-sources-package optional package prerelease prerelease-package project
            python refresh-package reinstall-package requirements resolution rev
            script tag upgrade-group upgrade-package
        """),
        flags=_names("""
            active compile-bytecode dev editable frozen help lfs locked
            managed-python no-binary no-build no-build-isolation no-cache no-config
            no-index no-install-local no-install-project no-install-workspace
            no-managed-python no-progress no-python-downloads no-require-build-hashes
            no-sources no-sync no-workspace offline quiet raw refresh reinstall
            require-build-hashes system-certs upgrade verbose workspace
        """),
        short={
            "r": "--requirements",
            "c": "--constraints",
            "m": "--marker",
            "i": "--index-url",
            "f": "--find-links",
            "U": "--upgrade",
            "P": "--upgrade-package",
            "C": "--config-setting",
            "n": "--no-cache",
            "p": "--python",
            "q": "--quiet",
            "v": "--verbose",
            "h": "--help",
        },
        inputs=_names("""
            constraints requirements
        """),
        short_equals=True,
    ),
    "run": Grammar(
        values=_names("""
            allow-insecure-host cache-dir color config-file config-setting
            config-settings-package default-index directory env-file exclude-newer
            exclude-newer-package extra extra-index-url find-links fork-strategy
            group index index-strategy index-url keyring-provider link-mode
            no-binary-package no-build-isolation-package no-build-package
            no-editable-package no-extra no-group no-sources-package only-group
            package prerelease prerelease-package project python python-platform
            refresh-package reinstall-package resolution upgrade-group
            upgrade-package with with-editable with-requirements
        """),
        flags=_names("""
            active all-extras all-groups all-packages compile-bytecode exact frozen
            gui-script help isolated locked managed-python module no-binary no-build
            no-build-isolation no-cache no-config no-default-groups no-dev
            no-editable no-env-file no-index no-managed-python no-progress no-project
            no-python-downloads no-require-build-hashes no-sources no-sync offline
            only-dev quiet refresh reinstall require-build-hashes script system-certs
            upgrade verbose
        """),
        short={
            "m": "--module",
            "w": "--with",
            "s": "--script",
            "i": "--index-url",
            "f": "--find-links",
            "U": "--upgrade",
            "P": "--upgrade-package",
            "C": "--config-setting",
            "n": "--no-cache",
            "p": "--python",
            "q": "--quiet",
            "v": "--verbose",
            "h": "--help",
        },
        inputs=_names("""
            with-requirements
        """),
        short_equals=True,
    ),
}


def _options(
    argv: list[str],
    grammar: Grammar,
    *,
    stop_at_positional: bool = False,
    strict: bool = False,
) -> tuple[list[tuple[str, str, int]], list[tuple[str, int]]]:
    """Return semantic option values and positionals, respecting arity and --."""
    values = []
    positional = []
    index = 0
    choices = grammar.values | grammar.flags
    while index < len(argv):
        word = argv[index]
        index += 1
        if word == "--":
            positional.extend((value, i) for i, value in enumerate(argv[index:], index))
            break
        if word == "-" or not word.startswith("-"):
            if stop_at_positional:
                positional.extend(
                    (value, i) for i, value in enumerate(argv[index - 1 :], index - 1)
                )
                break
            positional.append((word, index - 1))
            continue
        if word.startswith("--"):
            option, equals, value = word.partition("=")
            if option not in choices:
                matches = [name for name in choices if name.startswith(option)]
                if len(matches) != 1:
                    if strict:
                        raise UnknownDependencyInputs("unknown or ambiguous option")
                    continue
                option = matches[0]
            if option in grammar.values:
                value_index = index - 1
                if not equals:
                    if index >= len(argv):
                        if strict:
                            raise UnknownDependencyInputs("missing option value")
                        continue
                    value = argv[index]
                    value_index = index
                    index += 1
                values.append((option, value, value_index))
            elif equals and strict:
                raise UnknownDependencyInputs("flag has a value")
            continue
        offset = 1
        while offset < len(word):
            option = grammar.short.get(word[offset])
            offset += 1
            if option is None:
                if strict:
                    raise UnknownDependencyInputs("unknown short option")
                break
            if option in grammar.values:
                value = word[offset:]
                value_index = index - 1
                if not value:
                    if index >= len(argv):
                        if strict:
                            raise UnknownDependencyInputs("missing option value")
                        break
                    value = argv[index]
                    value_index = index
                    index += 1
                elif grammar.short_equals:
                    value = value.removeprefix("=")
                values.append((option, value, value_index))
                break
    if strict:
        # These operands can select local configuration, credentials, source
        # directories or another project's files. Their transitive reads are
        # not represented by requirement/constraint paths.
        for option, _value, _ in values:
            if option not in grammar.inputs and option not in {"--python"}:
                raise UnknownDependencyInputs("unmodeled option inputs")
    return values, positional


_CONVENTIONAL = re.compile(
    r"requirements[\w.+-]*\.(?:txt|in)|pylock(?:\.[\w.-]+)?\.toml|setup\.(?:py|cfg)"
)


# Words that run the command after them (a test.sh line's wrappers).
_WRAPPERS = frozenset({"command", "exec", "env"})


def dependency_inputs(
    argv: list[str],
    *,
    unknown_python: bool = False,
    precise: list[bool] | None = None,
    strict: bool = False,
) -> list[str] | None:
    """Return installer input paths, or None when argv is not an installer.

    The caller tracks whether each value is literal; no path gets authority
    merely because an expression fragment resembles an absolute pathname.
    """
    if not argv:
        return None
    original_length = len(argv)
    argv = list(argv)
    # Shell wrappers and assignments select argv; later tool names are data.
    while argv:
        executable = argv.pop(0)
        if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*=.*", executable, re.DOTALL):
            continue
        executable = executable.rsplit("/", 1)[-1]
        if executable in _WRAPPERS:
            while argv and argv[0].startswith("-"):
                option = argv.pop(0)
                if option == "--":
                    break
                if executable == "env" and option in {"-u", "--unset", "-C", "--chdir"}:
                    if argv:
                        argv.pop(0)
                elif option not in {
                    "-i",
                    "-p",
                    "-c",
                    "-l",
                    "--ignore-environment",
                } and not option.startswith(("--unset=", "--chdir=")):
                    return None
            continue
        break
    else:
        return None
    if re.fullmatch(r"python[\d.]*", executable) or unknown_python:
        invocation = python_invocation(argv)
        if invocation is None or invocation.kind != "m":
            return None
        executable = invocation.value
        argv = argv[invocation.rest :]
    if re.fullmatch(
        r"pip(?:[\d.]*|\.__main__|-(?:sync|compile))", executable
    ) or executable in {"pipenv", "pipx"}:
        if strict:
            # pip-tools and wrappers have implicit files/configuration of
            # their own. Even direct pip may execute in another environment:
            # bootstrap sys.prefix does not bound that interpreter's pip.conf.
            # Until that environment is modeled, every workspace file is an
            # input. Recovery callers may still obtain non-authoritative hints.
            raise UnknownDependencyInputs("unmodeled installer execution environment")
        grammar = _PIP
    elif executable in {"uv", "uv.__main__"}:
        _, commands = _options(argv, _UV[""], stop_at_positional=True, strict=strict)
        if not commands:
            return None
        command, command_index = commands[0]
        start = command_index + 1
        if command == "pip":
            _, subcommands = _options(
                argv[start:], _UV[""], stop_at_positional=True, strict=strict
            )
            if not subcommands:
                return None
            subcommand, subcommand_index = subcommands[0]
            command += " " + subcommand
            start += subcommand_index + 1
        if command not in _UV or not command:
            return None
        grammar = _UV[command]
        argv = argv[start:]
    else:
        return None
    values, positional = _options(
        argv,
        grammar,
        stop_at_positional=executable in {"uv", "uv.__main__"} and command == "run",
        strict=strict,
    )
    inputs = [
        (value, index) for option, value, index in values if option in grammar.inputs
    ]
    # sync/compile consume positional requirement files, including custom names.
    if executable in {"pip-sync", "pip-compile"}:
        # Retain discovery for historical pip-sync/compile "install -r"
        # forms as well as their real positional-file grammar.
        inputs.extend(
            positional[1:]
            if inputs and positional[:1] and positional[0][0] == "install"
            else positional
        )
    if executable in {"uv", "uv.__main__"} and command in {"pip sync", "pip compile"}:
        inputs.extend(positional)
    if strict:
        packages = positional
        if executable in {"uv", "uv.__main__"} and command in {"run", "add"}:
            # Project commands resolve the project's dependency graph and
            # workspace sources, beyond their explicit command operands.
            raise UnknownDependencyInputs("unmodeled project command")
        file_indices = {index for _, index in inputs}
        for value, index in packages:
            if index not in file_indices and not registry_requirement(value):
                raise UnknownDependencyInputs("unmodeled package source")
    # Existing discovery also tracks conventional setup/requirements/lock files
    # named outside dependency options. Inspect semantic values, not flag text.
    for value, index in [*positional, *((value, index) for _, value, index in values)]:
        if _CONVENTIONAL.fullmatch(value.rstrip("/").rsplit("/", 1)[-1]):
            inputs.append((value, index))
    offset = original_length - len(argv)
    result = [
        value
        if precise is None or precise[offset + index]
        else value.rstrip("/").rsplit("/", 1)[-1]
        for value, index in inputs
    ]
    if executable in {"uv", "uv.__main__"} and command == "run" and positional:
        child_index = positional[0][1]
        child_precise = precise[offset + child_index :] if precise is not None else None
        result.extend(
            dependency_inputs(argv[child_index:], precise=child_precise, strict=strict)
            or []
        )
    return result
