"""The redactor's word shortcut and lazy patterns mask what the plain loop masks.

Review of PR #148, 2026-11-13. The shared left boundary (``_KEY_START``) is
9.9 KB of pattern text. The seven carriers that try it at every position of a
text made redaction 2.5 times slower, their ``\\1`` templates made ``re`` hash
a 250 KB pattern's program on every call, and compiling them added two seconds
to ``import benchflow.trajectories.types``; in the combined suite
``test_a_long_proxied_run_never_holds_its_log_in_memory`` went over its budget.

The fix must not change what is masked. So, here:

- each carrier's words are proved from its parsed pattern, not guessed;
- the shortcut is compared with every carrier run on every text, over each
  carrier name as written, encoded, split by a colour code, quoted and in the
  four letters ``re`` folds onto ASCII, over every string in the redaction
  test files, and over seeded random mixes;
- the function that replaces a ``\\1`` template is compared with the template;
- a text with none of a carrier's words runs none of the seven, and importing
  the module compiles no pattern that holds the boundary.

Every secret here is made up.
"""

from __future__ import annotations

import ast
import itertools
import random
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from re import _constants, _parser

import pytest

from benchflow.trajectories import types

_SECRET = "Zx" + "9Qv7Lm2R" + "t8Kp4Wn6" + "Yb"
_TABLE = types._REDACTION_PATTERNS
_CARRIERS = [pattern for pattern, _, _ in _TABLE if pattern.words]


def _subn(pattern, replacement, text: str) -> tuple[str, int]:
    """*pattern*'s replacements; a function's (the secret-name carrier's,
    which asks the classifier: part B) counted only where it changes
    the text, as the redactor's loop counts them. Structural only: what is
    masked is the function's either way."""
    if callable(replacement):
        return types._subn_changed(pattern.compiled(), replacement, text)
    return pattern.compiled().subn(replacement, text)


def _every_carrier(text: str) -> tuple[str, dict[str, int]]:
    """The loop with no shortcut: every pattern runs on every text; then the
    escaped text's decoded view, read on the text as written, as the
    redactor ends."""
    categories: dict[str, int] = {}
    original = text
    for pattern, replacement, category in _TABLE:
        text, count = _subn(pattern, types._PREFIX_KEEPERS[replacement], text)
        if count:
            categories[category] = categories.get(category, 0) + count
    text, count = types._view_after(original, text)
    if count:
        field = types.REDACTION_CATEGORY_CREDENTIAL_FIELD
        categories[field] = categories.get(field, 0) + count
    return text, categories


def _templates(text: str) -> tuple[str, dict[str, int]]:
    """The loop as it was before this change: every pattern, and its
     template as the table writes it; then the escaped text's decoded view
    , read on the text as written, as the redactor ends."""
    categories: dict[str, int] = {}
    original = text
    for pattern, replacement, category in _TABLE:
        text, count = _subn(pattern, replacement, text)
        if count:
            categories[category] = categories.get(category, 0) + count
    text, count = types._view_after(original, text)
    if count:
        field = types.REDACTION_CATEGORY_CREDENTIAL_FIELD
        categories[field] = categories.get(field, 0) + count
    return text, categories


# The words, proved from the pattern.

_WORD = "\1"  # a spelling that holds one of the words
_GAP = "\0"  # anything that is not a letter the pattern spells out


def _spellings(items, words: tuple[str, ...], fold: bool) -> set[str]:
    """Every way a parsed pattern's own letters can be spelled along a match.

    A literal is its letter, a group is its content, an alternation is each
    of its branches, a lookaround is nothing (it consumes nothing), and all
    else (a class, a repeat, a reference) is a gap, so nothing is assumed of
    it. A spelling that holds a word is cut to ``_WORD``; a long one keeps
    its two ends, which is all a neighbour can join.
    """
    cap = max(map(len, words))

    def squash(spelled: str) -> str:
        if _WORD in spelled or any(word in spelled for word in words):
            return _WORD
        if len(spelled) <= 2 * cap:
            return spelled
        return spelled[:cap] + _GAP + spelled[-cap:]

    spelled = {""}
    for op, av in items:
        if op is _constants.LITERAL:
            letter = chr(av)
            choices = {letter.lower() if fold else letter}
        elif op is _constants.SUBPATTERN:
            _group, add_flags, _del_flags, content = av
            choices = _spellings(content, words, fold or bool(add_flags & re.I))
        elif op is _constants.ATOMIC_GROUP:
            choices = _spellings(av, words, fold)
        elif op is _constants.BRANCH:
            choices = set().union(
                *(_spellings(branch, words, fold) for branch in av[1])
            )
        elif op in (_constants.ASSERT, _constants.ASSERT_NOT, _constants.AT):
            choices = {""}
        else:
            choices = {_GAP}
        spelled = {squash(a + b) for a in spelled for b in choices}
    return spelled


def _needs_a_word(source: str, flags: int, words: tuple[str, ...]) -> bool:
    parsed = _parser.parse(source, flags)
    return _spellings(parsed, words, bool(flags & re.I)) == {_WORD}


def test_the_proof_tells_a_needed_word_from_one_a_branch_can_do_without():
    assert _needs_a_word(
        r"x[a-z]{0,9}(?:TOKEN(?<!per_TOKEN)|_?KEY)\s*=", re.I, ("token", "key")
    )
    assert _needs_a_word(r"[A-Z]{1,9}_(?:PASS|PWD)=", 0, ("_PASS", "_PWD"))
    # A branch with no word, a word split by a class, an optional word, and
    # a word in the wrong case each leave a match that holds none.
    assert not _needs_a_word(r"(?:TOKEN|SECRET|PIN)=", re.I, ("token", "secret"))
    assert not _needs_a_word(r"TOK[E3]N=", re.I, ("token",))
    assert not _needs_a_word(r"(?:TOKEN)?=", re.I, ("token",))
    assert not _needs_a_word(r"[A-Z]{1,9}_(?i:PASS)=", 0, ("_PASS",))


def test_nine_carriers_have_words_and_they_are_the_ones_tried_everywhere():
    """A carrier that starts with the boundary is tried at every position of
    a text. Each has words; no other pattern does."""
    leading = [
        pattern
        for pattern, _, _ in _TABLE
        if re.sub(r"^\(\?=\[[^\]]*\]\)", "", pattern.pattern).startswith(
            types._KEY_START
        )
    ]
    # Eight since a price's name keeps a number only (the lead's ruling, 2026-11-16 09:58Z);
    # the ninth: every name that ends in `authorization` (the lead, 2026-11-16 10:53Z).
    assert len(leading) == 9
    assert leading == _CARRIERS


@pytest.mark.parametrize("index", range(9))
def test_no_match_of_a_carrier_is_without_one_of_its_words(index):
    """Read from the carrier's own pattern: every spelling of the letters
    along a match holds one of its words. A name added to a carrier without
    its word fails here."""
    carrier = _CARRIERS[index]
    words = carrier.words
    if carrier.flags & re.I:
        assert all(word == word.lower() and word.isascii() for word in words)
    assert _needs_a_word(carrier.pattern, carrier.flags, words)
    # Each word is needed: without it some match holds none of the rest.
    for word in words:
        rest = tuple(w for w in words if w != word)
        assert not rest or not _needs_a_word(carrier.pattern, carrier.flags, rest)


# The shortcut against every carrier run.

_NAMES = (
    "token",
    "api_token",
    "GITHUB_TOKEN",
    "secret",
    "client_secret",
    "password",
    "passwd",
    "access_key",
    "secretkey",
    "account_key",
    "private_key",
    "encryption_key",
    "credential",
    "credentials",
    "bearer_token",
    "AWS_BEARER_TOKEN_BEDROCK",
    "DB_PASS",
    "MYSQL_PWD",
    "cookie",
    "set-cookie",
    "HTTP_COOKIE",
    "authorization",
    "proxy-authorization",
    "HTTP_AUTHORIZATION",
)
# The letters `re` matches to an ASCII letter when case is ignored.
_FOLDS = {"i": "\u0131\u0130", "s": "\u017f", "k": "\u212a"}


def _forms(name: str) -> list[str]:
    """*name* as written, in each case, with a letter of its word encoded
    (``\\x74oken``, ``%74oken``, ``%2574oken``), split by a colour code as a
    byte and as text, and in each letter `re` folds onto one of its own."""
    forms = [name, name.upper(), name.lower(), name.capitalize()]
    for at in (0, len(name) // 2, len(name) - 1):
        code = ord(name[at])
        for encoded in (f"\\x{code:02x}", f"%{code:02x}", f"%25{code:02x}"):
            forms.append(name[:at] + encoded + name[at + 1 :])
    middle = len(name) // 2
    for colour in ("\x1b[0m", "\\x1b[0m", "\\u001b[0m", "%1B[0m"):
        forms.append(name[:middle] + colour + name[middle:])
    for at, letter in enumerate(name):
        for folded in _FOLDS.get(letter.lower(), ""):
            forms.append(name[:at] + folded + name[at + 1 :])
    return forms


_QUOTES = ("{}", '"{}"', '\\"{}\\"', '\\\\\\"{}\\\\\\"')
_BEFORE = ("", " ", "x", "\\n", "\x1b[32m", "\\u001b[1;32m", "%0A", '{"env": {')
_VALUES = (
    f"={_SECRET}",
    f": {_SECRET}",
    f": Bearer {_SECRET}",
    f": sid={_SECRET}; theme=dark",
    f":\n  {_SECRET}",
    f': \\"{_SECRET}\\"',
    "=12",
    ": required",
)


def _name_cases() -> list[str]:
    return [
        before + quote.format(form) + value
        for name in _NAMES
        for form in _forms(name)
        for quote in _QUOTES
        for before in _BEFORE
        for value in _VALUES
    ]


def _strings_of_the_redaction_tests() -> list[str]:
    """Every string the redaction test files spell out: the review's matrix
    of secrets, escapes, encodings and look-alikes."""
    found: set[str] = set()
    tests = Path(__file__).parent
    for path in sorted(
        {*tests.glob("test_redact*.py"), *tests.glob("test_*redaction*.py")}
    ):
        for node in ast.walk(ast.parse(path.read_text())):
            value = getattr(node, "value", None)
            if isinstance(node, ast.Constant) and isinstance(value, str):
                found.add(value)
    return sorted(found)


def _random_mixes(count: int) -> list[str]:
    rng = random.Random(148)
    pieces = [
        *(form for name in _NAMES for form in _forms(name)[:8]),
        *("=", ": ", ":\n  ", " ", "\n", '"', '\\"', ";", "&", "?", ",", "{", "}"),
        *(_SECRET, f"Bearer {_SECRET}", f"sid={_SECRET}", "12", "true", "required"),
        *("\u0131", "\u0130", "\u017f", "\u212a", "\xb5", "\x1b[32m", "\\n", "%20"),
        *("y" * 40, "task-implementation", "FAQ.", "n_input_tokens", "cost_per_token"),
    ]
    return ["".join(rng.choices(pieces, k=rng.randint(1, 12))) for _ in range(count)]


@pytest.fixture(scope="module")
def name_cases() -> list[str]:
    return _name_cases()


def test_the_shortcut_masks_what_every_carrier_run_masks(name_cases):
    """Text for text, and count for count. Nothing a carrier masks is lost
    to its words: not an encoded name, not one a colour code splits, not a
    quoted one, not one in a letter `re` folds onto ASCII."""
    corpus = [*name_cases, *_strings_of_the_redaction_tests(), *_random_mixes(4000)]
    assert len(corpus) > 30000
    masked = skipped = 0
    for text in corpus:
        expected = _every_carrier(text)
        assert types.redact_trajectory_text_with_categories(text) == expected, text
        masked += bool(expected[1])
        folded = types._folded(text)
        skipped += any(not carrier.may_match(text, folded) for carrier in _CARRIERS)
    # Both sides are met: texts that are masked, and texts a carrier skips.
    assert masked > 5000 and skipped > 5000


def test_each_carrier_masks_its_own_names_through_the_shortcut():
    """The corpus above is no row of texts nothing masks: every name, as
    written and in upper case, loses its value."""
    for name in _NAMES:
        value = f": sid={_SECRET}" if "cookie" in name.lower() else f"={_SECRET}"
        for form in {name, name.upper()}:
            text = form + value
            redacted = types.redact_trajectory_text(text)
            assert _SECRET not in redacted, text
            assert redacted == _every_carrier(text)[0]


@pytest.mark.parametrize(
    "text",
    [
        f"GITHUB_TO\u212aEN={_SECRET}",
        f"client_\u017fecret={_SECRET}",
        f"Author\u0131zation: Bearer {_SECRET}",
        f"COOK\u0130E: sid={_SECRET}",
        f"pa\u017f\u017fword={_SECRET}",
    ],
)
def test_a_name_in_a_letter_re_folds_onto_ascii_is_still_masked(text):
    """`re` matches these four letters to `i`, `s` and `k` when case is
    ignored, and ``str.lower`` does not write them so: lower case would not
    show the word, so such a text is given to every carrier."""
    assert _SECRET not in _templates(text)[0]
    assert types._folded(text) is None
    assert types.redact_trajectory_text_with_categories(text) == _templates(text)
    assert _SECRET not in types.redact_trajectory_text(text)


def test_the_function_writes_what_the_template_writes(name_cases):
    """A ``\\1`` template is now a function (``_prefix_keeper``): the same
    text, and the same counts."""
    corpus = [
        *name_cases[::53],
        *_strings_of_the_redaction_tests(),
        *_random_mixes(600),
        f"https://user:{_SECRET}@example.test/x?sig={_SECRET}{_SECRET}",
        f"x-api-key: {_SECRET}\nmaster_key={_SECRET}",
    ]
    changed = 0
    for text in corpus:
        expected = _templates(text)
        assert _every_carrier(text) == expected, text
        changed += expected[0] != text
    assert changed > 500
    # The secret-name carrier's replacement is a function that asks the
    # classifier (part B, the lead's OK: structural, no masking
    # assertion changed); every other is one of the three templates.
    templates = {
        replacement for _, replacement, _ in _TABLE if isinstance(replacement, str)
    }
    assert templates == {"***REDACTED***", r"\1***REDACTED***", r"\1***REDACTED***@"}
    functions = {replacement for _, replacement, _ in _TABLE if callable(replacement)}
    # And the prose pass under a code Authorization name (the lead's ruling
    # (b), 2026-11-15 01:3xZ), and a YAML block's body.
    assert functions == {
        types._secret_name_value,
        types._authorization_value,
        types._code_name_prose,
        types._yaml_block,
        types._price_value,  # the lead's ruling, 2026-11-16 09:58Z
        types._next_line_entry_value,  # a next line's entry,
    }
    assert types._prefix_keeper("***REDACTED***") == "***REDACTED***"
    # A group that took no part is the empty string, as in a template.
    optional = re.compile(r"(a)?b")
    keeper = types._prefix_keeper(r"\1***")
    assert callable(keeper)
    assert optional.sub(keeper, "b ab") == optional.sub(r"\1***", "b ab") == "*** a***"
    # A template with anything more than `\1` and plain text stays one.
    assert types._prefix_keeper(r"\1***\2") == r"\1***\2"
    assert types._prefix_keeper(r"\g<1>***") == r"\g<1>***"


# What the shortcut saves.


def _carriers_run(monkeypatch, text: str) -> list[types._LazyPattern]:
    ran: list[types._LazyPattern] = []
    real = types._LazyPattern.subn

    def subn(self, repl, string):
        ran.append(self)
        return real(self, repl, string)

    with monkeypatch.context() as patch:
        patch.setattr(types._LazyPattern, "subn", subn)
        types.redact_trajectory_text(text)
    return [pattern for pattern in ran if pattern.words]


def test_a_text_with_none_of_their_words_runs_none_of_the_seven(monkeypatch):
    """What a proxied log mostly holds: tool output, encoded reasoning."""
    for text in ("y" * 14000, "gA" * 7000, "step 3 \xb5 " + "y" * 400, "PASS 12 of 12"):
        assert _carriers_run(monkeypatch, text) == []


def test_a_text_with_a_word_runs_the_carriers_that_need_it(monkeypatch):
    bearer, secret, price, passes, cookie, *authorization = _CARRIERS
    assert _carriers_run(monkeypatch, "max_TOKENS: 4096") == [secret, price]
    assert _carriers_run(monkeypatch, "sort_key=3") == [secret]
    assert _carriers_run(monkeypatch, "TESTS_PASS=12") == [passes]
    assert _carriers_run(monkeypatch, "Set-Cookie: header too large") == [cookie]
    assert _carriers_run(monkeypatch, "Authorization: required") == authorization
    assert _carriers_run(monkeypatch, "a BEARER_token b") == [bearer, secret, price]
    # A letter `re` folds onto ASCII: lower case cannot say, so all but the
    # carrier that reads upper case as written.
    assert _carriers_run(monkeypatch, "\u017f") == [
        bearer,
        secret,
        price,
        cookie,
        *authorization,
    ]


def test_the_words_are_looked_for_in_the_text_as_it_then_stands(monkeypatch):
    """After the patterns before a carrier have masked what they mask: the
    only `token` here is a URL's user name, gone when the carriers come."""
    text = f"https://token:{_SECRET}@example.test/x"
    assert types.redact_trajectory_text(text) == "https://***REDACTED***@example.test/x"
    assert types.redact_trajectory_text_with_categories(text) == _every_carrier(text)
    assert _carriers_run(monkeypatch, text) == []


def test_importing_the_module_compiles_no_pattern_that_holds_the_boundary():
    """The boundary's patterns took two seconds to compile at import. They
    are compiled when first used, and a text with no carrier word compiles
    none of the seven."""
    script = textwrap.dedent(
        """
        import re
        real, seen = re.compile, []
        def spy(pattern, flags=0):
            seen.append(pattern)
            return real(pattern, flags)
        re.compile = spy
        from benchflow.trajectories import types
        re.compile = real
        holders = [pattern for pattern, _, _ in types._REDACTION_PATTERNS]
        holders += [pattern for pattern, _ in types._JUDGE_FAMILY_PATTERNS]
        holders.append(types._SECRET_FIELD)
        # 34: the camel-case code Authorization name's carrier and the prose
        # pass after it (the lead's ruling (b), 2026-11-15 01:3xZ); 35: a YAML
        # block under a credential's name; 36: a price's name, which
        # keeps a number only (the lead's ruling, 2026-11-16 09:58Z);
        # 37: every name that ends in `authorization` (the lead, 2026-11-16 10:53Z);
        # 38: a next line's entry under a credential's name.
        assert len(holders) == 38 + 18 + 1, len(holders)
        assert all(holder._compiled is None for holder in holders)
        sources = [s for s in seen if isinstance(s, str)]
        assert not any(types._NOT_AFTER_CSI in s for s in sources)
        text = "step 3 \\xb5 " + "y" * 400
        assert types.redact_trajectory_text(text) == text
        seven = [holder for holder in holders if holder.words]
        assert len(seven) == 9 and all(h._compiled is None for h in seven)
        # and the YAML block row, which needs a `|` or `>` (a gated pattern, no carrier)
        assert sum(h._compiled is not None for h in holders) == 38 - 9 - 1
        masked = types.redact_trajectory_text("GITHUB_TOKEN=Zx"""
        + """9Qv7Lm2R"""
        + """t8Kp4Wn6"""
        + """Yb")
        assert masked == "GITHUB_TOKEN=***REDACTED***", masked
        assert types.is_secret_field("client_secret")
        print("ok")
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr[-3000:]
    assert done.stdout.strip() == "ok"


def test_a_lazy_pattern_is_the_compiled_pattern_in_all_else():
    holder = types._LazyPattern(r"(a)(?P<b>b)?", re.IGNORECASE)
    assert (holder.pattern, holder.flags, holder.words) == (r"(a)(?P<b>b)?", re.I, ())
    assert holder._compiled is None
    assert holder.groups == 2 and holder.groupindex["b"] == 2
    assert holder.fullmatch("AB") and holder.search("xab").start() == 1
    assert holder.sub("-", "ab a") == "- -" and holder.subn("-", "ab") == ("-", 1)
    assert holder.compiled() is holder.compiled()
    for missing in ("no_such_attribute", "_private"):
        assert not hasattr(holder, missing)


def test_the_test_names_cover_every_form_the_review_listed():
    forms = _forms("token")
    for listed in ("token", "\\x74oken", "%74oken", "%2574oken", "to\x1b[0mken"):
        assert listed in forms
    assert '\\"token\\"' in {quote.format("token") for quote in _QUOTES}
    assert len(list(itertools.chain.from_iterable(_forms(n) for n in _NAMES))) > 400
