"""The shared name classifier, benchflow._utils.credential_names.

Carried as one file, with this test, by the config override record and the
trajectory redactors, so the two cannot drift. The table is the lead's
rulings of 2026-11-13 (PR #145): which credential words a companion
word turns into a setting, the tokenizer and id rules, and the count rule
for ``credentials``. The review of PR #145 (L4) made a key run together an
ending of a word; a corpus of ordinary words measures what else that rule
catches.
"""

from __future__ import annotations

import ast
import builtins
import functools
import importlib.util
import json
import keyword
import re
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

from benchflow._utils import credential_names
from benchflow._utils.credential_names import _words, entry_kind, name_kind, name_words
from benchflow.trajectories.types import is_secret_number, redact_trajectory_obj

#: Names whose value is a setting's: stored and compared.
SETTINGS = [
    # yield, fail and through just after pass (the lead's ruling, 2026-11-14)
    "first_pass_yield", "firstPassYield", "FIRST_PASS_YIELD", "pass_fail",
    "PASS_FAIL", "passFail", "pass_through", "PASS_THROUGH", "passThrough",
    "pass_through_endpoints", "pass_fail_count",
    # tokens beside any companion word
    "max_tokens", "max_output_tokens", "max_thinking_tokens", "prompt_tokens", "context_tokens", "completion_tokens", "total_tokens",
    "num_tokens", "n_input_tokens", "n_output_tokens", "maxTokens", "max-tokens",
    # token or session beside a limit, count or time word
    "token_budget", "token_limit", "max_token", "token_ttl",
    # credentials beside a count word
    "num_credentials", "max_credentials", "total_credentials",
    "count_credentials", "min_credentials", "credentials_limit",
    "credentials_count",
    # a tokenizer's special tokens, and a token's id
    "eos_token", "bos_token", "pad_token", "unk_token", "sep_token",
    "cls_token", "mask_token", "eos_tokens", "eos_token_id", "eos_token_ids",
    "pad_token_id", "bos_token_id", "add_bos_token", "add_eos_token",
    "token_id", "token_ids", "input_token_ids",
    # run together
    "maxtokens", "maxoutputtokens", "maxtoken", "eostoken", "bostoken",
    "padtoken", "eostokenid", "masktoken", "septoken", "numtokens",
    # singular token beside max, a limit word: a setting (output beside it
    # changes nothing; the lead's review of 2c568af4c pins it)
    "max_output_token",
    # the lead's list (review of a2f87822a): these must stay settings
    "input_tokens", "refresh_token_count", "api_token_limit", "session_timeout",
    "session_ttl", "INPUT_TOKENS",
    # tokens beside a usage word: cache_read, cache_write, cache_creation and
    # n join the companions (ruled in review, 2026-11-13: token counts are
    # never withheld); cache_read_tokens was a credential's row until then
    "cache_read_tokens", "cache_write_tokens", "cache_creation_tokens",
    "n_cache_read_tokens", "n_cache_tokens", "cacheReadTokens",
    # and the usage and statistics words of a count (ruled in review,
    # 2026-11-13: token counts are never withheld)
    "cached_tokens", "cachedReadTokens", "native_tokens_cached", "audio_tokens",
    "image_tokens", "text_tokens", "thought_tokens", "thoughtsTokens",
    "accepted_prediction_tokens", "rejected_prediction_tokens",
    "candidatesTokensDetails", "cacheTokensDetails", "final_tokens",
    "mean_tokens_to_converge", "median_tokens", "avg_tokens", "sum_tokens",
    "p50_tokens", "p90_tokens", "p95_tokens", "p99_tokens", "subagent_tokens",
    "stream_tokens", "pre_tokens", "preTokens", "original_tokens",
    "live_usage_tokens",
    # a provider's word just before tokens, with a pure count word in the
    # name: a count (ruling A, review of 2026-11-13)
    "total_access_tokens", "n_api_tokens", "max_session_tokens",
    "RESPONSES_API_MIN_OUTPUT_TOKENS", "OPENAI_RESPONSES_API_MIN_MAX_OUTPUT_TOKENS",
    # a TTL, an expiry or a count beside a token decides before the price
    # rule (ruled in review, 2026-11-13)
    "MCP_PER_USER_TOKEN_DEFAULT_TTL", "MCP_PER_USER_TOKEN_EXPIRY_BUFFER_SECONDS",
    "_compute_per_user_token_ttl", "_get_per_item_prompt_tokens",
    "_get_per_item_prompt_tokens_details", "per_sample_token_count",
    # a price word with per and any unit: cost fields are never withheld
    # (ruled in review, 2026-11-13); LiteLLM's price table had the first
    "code_interpreter_cost_per_session", "cost_per_session", "cost_per_request",
    "usd_per_1k", "input_cost_per_image", "file_search_cost_per_1k_calls",
    # pass just before a metric's word, as the last word after a test's, an
    # outcome's or a count's, or before a qualifier (ruling C, review of
    # 2026-11-13); SWE-bench's FAIL_TO_PASS and PASS_TO_PASS by name
    "pass_rate", "pass_at_k", "pass@k", "pass_count", "pass_score", "pass_at_1",
    "PASS_RATE", "passRate", "pass_ratio", "pass_fraction", "pass_pct",
    "mean_pass_rate", "tests_pass", "test_pass", "ALL_TESTS_PASS",
    "all_blockers_pass", "deterministic_pass", "first_pass", "second_pass",
    "final_pass", "is_pass", "expected_pass", "did_pass", "check_pass",
    "checks_pass", "gate_pass", "smoke_pass", "fail_to_pass", "FAIL_TO_PASS",
    "PASS_TO_PASS", "n_pass", "num_pass", "total_pass", "PASS_FILE",
    "DB_PASS_PATH",
    # pass@<digits> reads as pass_at_<n>, a metric like pass@k (the lead's
    # ruling on RPUB's lows, review of PR #173, 2026-11-13)
    "pass@1", "pass@10", "pass@100", "PASS@1", "pass_at_10",
    # and hat, threshold, total, deltas after pass; gates and blocker before
    # it; three of our own record fields by name (the lead's ruling on the
    # two counterexamples, 2026-11-13)
    "pass_hat_k", "pass_hat_k_scored_only", "pass_threshold",
    "scoring.pass_threshold", "pass_total", "pass_deltas", "gates_pass",
    "blocker_pass", "pass_to_fail", "first_pass_iteration", "tokens_to_pass",
    # a provider's word before tokens is the kind of token a count counts
    "max_auth_tokens",
    # the eleven fields of ours the guard found (the lead's ruling of
    # 2026-11-13): ratio after token; var and staged as last words; resume
    # before a last session; env before a last keys; three by name
    "token_ratio_vs_baseline", "token_ratio", "api_key_var",
    "_muse_credentials_staged", "credentials_staged", "resume_session",
    "retry_resume_session", "resume-session", "agent_env_keys", "env_keys",
    "_attempt_key", "idempotency_key", "atif_session_id",
]  # fmt: skip

#: Names whose value is a credential's: withheld, never compared.
CREDENTIALS = [
    # token or session without a limit, count or time word
    "input_token", "prompt_token", "context_token", "output_token",
    "MS_GRAPH_TOKEN", "session", "sessionid", "session_id", "session_token",
    "sid", "session_output", "use_token", "GITHUB_TOKEN", "time_to_first_token",
    "token_mask", "inputtoken",
    # bare tokens with no companion: TOKENS=a,b can be a list of secrets
    # (ruled in review, 2026-11-13); toolathlon's list of token env names
    "tokens", "_GLOBAL_TOKENS",
    # the usage words are companions of plural tokens only
    "final_token", "original_token", "stream_token", "native_token",
    # four usage names no ruled word reaches stay credentials, by ruling
    # (tokens_to_pass is a setting by name since the lead's later ruling)
    "tokens_unknown_trials", "tokens_trajectory", "trajectory.tokens",
    "of_tokens",
    # the id rule does not hold beside a provider's word
    "access_token_id", "api_token_id", "accesstokenid",
    # always a credential, whatever is beside it
    "max_api_key", "api_key_timeout", "password_length", "max_password",
    "secret_size", "key_size", "use_auth_token",
    "has_password", "is_secret", "DB_PASS", "api_key", "password", "x-api-key",
    "MYSQL_PWD", "masterKey", "api_password", "API_KEY", "authToken", "Cookie",
    "db_passwd", "PWD", "passphrase", "bearer", "client_secret", "X-Auth-Token",
    "githubToken", "APIKey", "apikey", "Authorization", "connect.sid",
    "passwd", "pass", "no-auth",
    # credential, and credentials without a count word
    "credential", "credentials", "num_credential", "credentials_timeout",
    "credentials_size", "aws_credentials", "max_credential", "withCredentials",
    # run together
    "authtoken", "accesstoken", "refreshtoken", "apitoken", "ghtoken",
    "GHTOKEN", "privatekey", "secretkey", "clientsecret", "xapikey",
    "sessiontoken", "JSESSIONID", "jsessionid", "mstoken", "idtoken",
    "oauthtoken", "bearertoken",
    # keys run together (the lead's review of 2c568af4c), alone, upper-case
    # and in a longer name
    "accesskey", "masterkey", "signingkey", "sessionkey", "encryptionkey",
    "hmackey", "ACCESSKEY", "MASTERKEY", "aws_accesskey", "X-MasterKey",
    "authkey", "publickey", "licensekey", "storagekey",
    "sharedkey", "clientkey", "rootkey",
    # key is a credential's word: a schema's foreign key too. primary_key
    # (and secondary_key) moved to ENTRY_KINDS (PRIMARY_KEY_ROWS) by the
    # lead's ruling of 2026-11-14: their kind depends on the value
    "foreign_key",
    # the lead's list (review of a2f87822a): real credentials, word for word
    "ms_token", "msToken", "INPUT_TOKEN", "INPUT_PASSWORD", "INPUT_API_KEY",
    "context7_api_key", "completion_api_key", "reasoning_api_key",
    "prompt_api_key", "output_secret", "total_secret", "api_key_input",
    "access_token_ms", "cookie_ttl", "bearer_ttl", "auth_timeout",
    # two credential words
    "session_token_ttl",
    # per and a word that is no token's kind, in a name with no price word:
    # a token per user, server, project, org or request, not a price (the
    # lead's narrowing of the price rule, review of 2026-11-13)
    "PER_USER_TOKEN", "MCP_PER_USER_TOKEN", "PER_SERVER_TOKEN",
    "GITLAB_PER_PROJECT_TOKEN", "PER_ORG_TOKEN", "per_request_token",
    # a provider's word just before tokens lists them; cached is no pure
    # count word (ruling A, review of 2026-11-13)
    "cached_refresh_tokens", "get_cached_refresh_tokens", "access_tokens",
    "refresh_tokens", "api_tokens", "oauth_tokens", "bearer_tokens",
    "session_tokens", "client_tokens", "id_tokens", "cached_id_tokens",
    "cached_oauth_tokens",
    # another credential word, or the name's first token, decides before the
    # price rule (ruled in review, 2026-11-13)
    "per_token_secret", "_BATCH_TOKEN_ESTIMATE_BYTES_PER_TOKEN",
    # pass anywhere else is a password's word (ruling C, review of 2026-11-13)
    "SMTP_PASS", "db_pass", "admin_pass", "PASS_WORD", "pass_wd", "pass_phrase",
    "DB_PASSWORD", "DB_PASS_2", "PASS_VALUE", "user_pass", "score_pass",
    "pg_pass",
    # passvalue, run together, is a password's word (the lead's ruling on
    # the redactor agent's agreement test, 2026-11-13)
    "PASSVALUE", "passvalue", "DB_PASSVALUE",
    # the new exits reach no further than ruled
    "API_KEY_VALUE", "resume_session_token", "env_key", "keys", "api_keys",
    "secret_keys", "token_ratio_secret", "staged_credentials",
    # a bare PASS_<digit> is a password, and so is pass@ before anything
    # but digits or k (RPUB's lows, review of PR #173, 2026-11-13)
    "PASS_2", "PASS_10", "pass_2", "pass@word", "pass@secret",
    # the price rule yields when a provider's or a credential's word sits
    # between per and the token (the lead's ruling, 2026-11-13)
    "price_per_refresh_token", "usd_per_access_token", "cost_per_auth_token",
    # and before a key, and a credential word as the unit before a token
    # (the agent's readings, accepted by the lead, 2026-11-13)
    "cost_per_session_token", "price_per_api_key",
    # resume_session_id holds the session id a rollout resumes, as
    # session_id does: a credential until the lead rules otherwise
    "resume_session_id",
    # a key run together after a glued prefix (review of PR #145,
    # 2026-11-13, L4)
    "myaccesskey", "mymasterkey", "mysigningkey", "mysessionkey",
    "myencryptionkey", "myhmackey", "mylicensekey", "mystoragekey",
    "mysharedkey", "myclientkey", "myrootkey", "mypublickey",
    # and after the older keys run together (the lead's ruling on L4)
    "myapikey", "myauthkey", "myprivatekey", "mysecretkey", "myaccesskeyid",
]  # fmt: skip

#: Names with no credential word.
NEITHER = [
    "author", "hotkeys", "keyboard", "sidebar", "tokenizer", "passenger",
    "hotkey", "monkey", "turkey",
    # one word that only contains pass, and with a camel hump inside it
    # (the lead's ruling, 2026-11-14)
    "bypass", "ByPass", "isByPass", "ByPASS", "BYPASS_CACHE",
    # passvalue only as a whole word: COMPASS_VALUE is no password
    "compassvalue", "dbpassvalue",
]  # fmt: skip


@pytest.mark.parametrize("name", SETTINGS)
def test_a_setting_name(name):
    assert name_kind(name) == "setting"


@pytest.mark.parametrize("name", CREDENTIALS)
def test_a_credential_name(name):
    assert name_kind(name) == "credential"


@pytest.mark.parametrize("name", NEITHER)
def test_a_name_with_no_credential_word(name):
    assert name_kind(name) is None


def test_the_module_imports_only_the_standard_library():
    """Another branch carries a byte-identical copy: no benchflow import."""
    from benchflow._utils import credential_names

    tree = ast.parse(Path(credential_names.__file__).read_text())
    imported = {
        (node.module or "").split(".")[0]
        if isinstance(node, ast.ImportFrom)
        else alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    }
    assert imported <= {"__future__", "re", "typing"}


def test_the_table_names_each_name_once():
    names = [*SETTINGS, *CREDENTIALS, *NEITHER]
    assert len(names) == len(set(names))


#: The keys run together that end a credential's word (review of PR #145,
#: 2026-11-13, L4; the lead's ruling on it added the five older ones).
GLUED_KEYS = [
    "apikey", "authkey", "privatekey", "secretkey", "accesskeyid",
    "accesskey", "masterkey", "signingkey", "sessionkey", "encryptionkey",
    "hmackey", "licensekey", "storagekey", "sharedkey", "clientkey",
    "rootkey", "publickey",
]  # fmt: skip
#: The review's six forms of each: alone, upper-case, after ``aws_``, in a
#: header, before ``_value`` and after a glued ``my`` (the ones it found
#: stored).
REVIEW_FORMS = [
    form.format(key=key, KEY=key.upper())
    for key in GLUED_KEYS
    for form in ("{key}", "{KEY}", "aws_{key}", "X-{KEY}", "{key}_value", "my{key}")
]
#: Glued prefixes beyond ``my``, as written and upper-case, and camelCase.
GLUED_PREFIXES = [
    *(prefix + key for prefix in ("aws", "gh", "the", "user") for key in GLUED_KEYS),
    *(
        (prefix + key).upper()
        for prefix in ("my", "aws", "gh", "the", "user")
        for key in GLUED_KEYS
    ),
    "myAccessKey", "MyAccesskey", "awssecretaccesskey", "AWSSECRETACCESSKEY",
]  # fmt: skip
#: Names the ending rule leaves as they were: ``key`` is no ending, and the
#: rulings beside it hold.
KEPT = {
    "hotkey": None, "monkey": None, "turkey": None, "keyboard": None,
    "passkey": None, "primary_key": "credential", "foreign_key": "credential",
    "sort_key": "credential", "partition_key": "credential",
    "cache_key": "credential", "max_tokens": "setting",
    "token_budget": "setting", "eos_token": "setting",
}  # fmt: skip

_REPO = Path(__file__).resolve().parents[1]
_FENCE = re.compile(r"^```(json|toml|ya?ml)[ \t]*\n(.*?)^```", re.MULTILINE | re.DOTALL)


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def _changed(names):
    """The names whose kind the ending rule changes, each with its kind
    before and after; before is name_kind with the keys run together
    matched as whole words only."""
    after = {name: name_kind(name) for name in names}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            credential_names,
            "_CREDENTIAL_WORDS",
            credential_names._CREDENTIAL_WORDS | set(GLUED_KEYS),
        )
        patch.setattr(
            credential_names,
            "_CREDENTIAL_ENDINGS",
            tuple(
                ending
                for ending in credential_names._CREDENTIAL_ENDINGS
                if ending not in GLUED_KEYS
            ),
        )
        before = {name: name_kind(name) for name in names}
    return {
        name: (before[name], after[name])
        for name in names
        if before[name] != after[name]
    }


@functools.cache
def _source_paths() -> tuple[Path, ...]:
    """Every Python file of benchflow's src/, in the order of _source_index."""
    return tuple(sorted((_REPO / "src").rglob("*.py")))


@functools.cache
def _source_index() -> tuple[tuple[list[ast.AST], dict[int, ast.AST]], ...]:
    """Every Python file of benchflow's src/, parsed and walked once: its
    nodes (the module first) and each node's parent, by id."""
    index = []
    for path in _source_paths():
        todo: list[ast.AST] = [ast.parse(path.read_text(encoding="utf-8"))]
        nodes: list[ast.AST] = []
        parents: dict[int, ast.AST] = {}
        while todo:
            node = todo.pop()
            nodes.append(node)
            for child in ast.iter_child_nodes(node):
                parents[id(child)] = node
                todo.append(child)
        index.append((nodes, parents))
    return tuple(index)


def _identifiers(nodes: list[ast.AST]) -> set[str]:
    """Every name, attribute, argument, keyword, function, class and string
    dict key of one Python file."""
    found: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.arg | ast.keyword) and node.arg:
            found.add(node.arg)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            found.add(node.name)
        elif isinstance(node, ast.Dict):
            found.update(
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            )
    return found


def _keys(value: object) -> set[str]:
    """Every string key of a parsed config, at any depth."""
    if isinstance(value, dict):
        nested = [_keys(item) for item in value.values()]
        return {key for key in value if isinstance(key, str)}.union(*nested)
    if isinstance(value, list):
        return set().union(*(_keys(item) for item in value))
    return set()


def _config_keys() -> set[str]:
    """The keys of the YAML, TOML and JSON config examples under docs/ and
    tests/fixtures/ that parse: files, and fenced blocks in a page."""
    keys: set[str] = set()
    for folder in ("docs", "tests/fixtures"):
        for path in sorted((_REPO / folder).rglob("*")):
            kind = path.suffix.lstrip(".")
            if kind == "md":
                blocks = _FENCE.findall(path.read_text(encoding="utf-8"))
            elif kind in ("json", "toml", "yaml", "yml"):
                blocks = [(kind, path.read_text(encoding="utf-8"))]
            else:
                continue
            for language, text in blocks:
                try:
                    if language == "json":
                        keys |= _keys(json.loads(text))
                    elif language == "toml":
                        keys |= _keys(tomllib.loads(text))
                    else:
                        keys |= _keys(list(yaml.safe_load_all(text)))
                except (ValueError, yaml.YAMLError):
                    continue
    return keys


#: A word of a token count's, a price's, a cost's or a usage record's name.
_ACCOUNTING_WORDS = frozenset(
    {"token", "tokens", "cost", "costs", "usd", "price", "prices", "pricing",
     "usage", "billing", "budget", "cache", "cached"}
)  # fmt: skip


def _field_names(nodes: list[ast.AST]) -> set[str]:
    """The names one Python file reads and writes as fields: string dict
    keys and subscripts, the key of a ``.get``, ``.pop`` or ``.setdefault``,
    keyword arguments, attributes and annotated fields."""
    found: set[str] = set()
    for node in nodes:
        keys: list[ast.expr | None] = []
        if isinstance(node, ast.Dict):
            keys = node.keys
        elif isinstance(node, ast.Subscript):
            keys = [node.slice]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("get", "pop", "setdefault"):
                keys = list(node.args[:1])
        elif isinstance(node, ast.keyword) and node.arg:
            found.add(node.arg)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            found.add(node.target.id)
        found.update(
            key.value
            for key in keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        )
    return found


def _schema_properties(value: object) -> set[str]:
    """Every property name of a JSON schema, at any depth."""
    if isinstance(value, dict):
        own = value.get("properties")
        names = set(own) if isinstance(own, dict) else set()
        return names.union(*(_schema_properties(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_schema_properties(item) for item in value))
    return set()


def _accounting_names() -> set[str]:
    """The token, cost and usage field names benchflow reads and writes:
    each field name in src/ with an accounting word, every property of the
    published schemas (docs/reference/schemas/) and every key of the usage
    blocks the redaction guard builds (its USAGE_BLOCKS)."""
    from tests.test_redaction_escape_boundary import USAGE_BLOCKS

    fields = set().union(*(_field_names(nodes) for nodes, _ in _source_index()))
    names = {name for name in fields if _ACCOUNTING_WORDS & set(_words(name))}
    for path in sorted((_REPO / "docs/reference/schemas").glob("*.json")):
        names |= _schema_properties(json.loads(path.read_text(encoding="utf-8")))
    return names | _keys(list(USAGE_BLOCKS.values()))


_ENV_NAME = re.compile(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+")
_ENV_READ_NAME = re.compile(r"[A-Z_][A-Z0-9_]*")
#: A word of a collection's name that says it holds env var names.
_ENV_COLLECTION_WORDS = frozenset(
    {"env", "envs", "environ", "var", "vars", "keys", "names"}
)
_DOC_ENV_NAMES = (
    re.compile(r"\bBENCHFLOW_[A-Z0-9_]*[A-Z0-9]\b"),
    re.compile(r"^\|\s*`([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)`", re.MULTILINE),
)
_DOC_FENCE = re.compile(r"^```[^\n]*\n(.*?)^```", re.MULTILINE | re.DOTALL)
_DOC_FENCE_ENV = re.compile(
    r"^\s*(?:export\s+|-\s+)?([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\s*[=:]"
    r"|--env\s+([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)=",
    re.MULTILINE,
)


def _is_environ(node: ast.expr) -> bool:
    if isinstance(node, ast.Attribute):
        return node.attr == "environ"
    return isinstance(node, ast.Name) and node.id == "environ"


def _env_read(node: ast.AST) -> ast.expr | None:
    """The name *node* reads from the environment (``os.environ[...]``,
    ``environ.get``, ``.pop`` or ``.setdefault``, ``getenv``, ``... in
    os.environ``), or None."""
    if isinstance(node, ast.Subscript) and _is_environ(node.value):
        return node.slice
    if isinstance(node, ast.Call) and node.args:
        func = node.func
        if isinstance(func, ast.Attribute) and _is_environ(func.value):
            return node.args[0] if func.attr in ("get", "pop", "setdefault") else None
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        return node.args[0] if name == "getenv" else None
    if isinstance(node, ast.Compare) and isinstance(node.ops[0], ast.In | ast.NotIn):
        return node.left if _is_environ(node.comparators[0]) else None
    return None


def _target_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    return node.attr if isinstance(node, ast.Attribute) else None


def _env_names() -> dict[str, set[str]]:
    """Every environment variable name BenchFlow reads or documents, by
    where it was found. From src/, with ``ast``: a read of os.environ or
    getenv by a constant or by a named string constant; the argument of a
    function that reads the variable its parameter names (an env helper);
    the names of a collection a loop or comprehension reads one by one; a
    string assigned to a name ending in ``_ENV`` or ``_ENV_VAR``; and the
    env-shaped strings of a collection whose name says env, var, keys or
    names, or that holds a name read above. From docs/: every BENCHFLOW_*
    name, the first cell of an env table and the variables a fenced block
    sets. Upper-case names only."""
    index = _source_index()
    strings: dict[str, set[str]] = {}
    collections: dict[str, set[str]] = {}
    for nodes, _ in index:
        body = nodes[0].body if isinstance(nodes[0], ast.Module) else []
        classes = [node for node in body if isinstance(node, ast.ClassDef)]
        top = {id(node) for node in [*body, *(n for c in classes for n in c.body)]}
        for node in nodes:
            if not isinstance(node, ast.Assign | ast.AnnAssign) or node.value is None:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            for name in filter(None, map(_target_name, targets)):
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    if id(node) in top:  # a module's or a class's constant
                        strings.setdefault(name, set()).add(value.value)
                elif not isinstance(value, ast.Constant):
                    collections.setdefault(name, set()).update(
                        item.value
                        for item in ast.walk(value)
                        if isinstance(item, ast.Constant)
                        and isinstance(item.value, str)
                        and _ENV_NAME.fullmatch(item.value)
                    )

    def named(expr: ast.expr | None) -> set[str]:
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return {expr.value}
        name = _target_name(expr) if expr is not None else None
        values = strings.get(name, set()) if name else set()
        return values if len(values) == 1 else set()

    def listed(expr: ast.expr) -> set[str]:
        if isinstance(expr, ast.Tuple | ast.List | ast.Set):
            return set().union(*map(named, expr.elts))
        if isinstance(expr, ast.Call):
            inner = expr.args[0] if expr.args else expr.func
            return listed(inner.value if isinstance(inner, ast.Attribute) else inner)
        name = _target_name(expr)
        return collections.get(name, set()) if name else set()

    found: dict[str, set[str]] = {}
    helpers: dict[str, set[tuple[int, str]]] = {}
    comprehensions = ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp
    for nodes, parents in index:
        for node in nodes:
            key = _env_read(node)
            if key is None:
                continue
            found.setdefault("os.environ / getenv reads", set()).update(named(key))
            outer = parents.get(id(node)) if isinstance(key, ast.Name) else None
            while outer is not None and isinstance(key, ast.Name):
                if isinstance(outer, ast.FunctionDef | ast.AsyncFunctionDef):
                    args = (
                        outer.args.posonlyargs + outer.args.args + outer.args.kwonlyargs
                    )
                    params = [arg.arg for arg in args if arg.arg not in ("self", "cls")]
                    if key.id in params:
                        place = (params.index(key.id), key.id)
                        helpers.setdefault(outer.name, set()).add(place)
                loops = [outer] if isinstance(outer, ast.For) else []
                if isinstance(outer, comprehensions):
                    loops = outer.generators
                for loop in loops:
                    if isinstance(loop.target, ast.Name) and loop.target.id == key.id:
                        found.setdefault("collections read one by one", set()).update(
                            listed(loop.iter)
                        )
                outer = parents.get(id(outer))
    for nodes, _ in index:
        for node in nodes:
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            for index, param in helpers.get(node.func.id, set()):
                given = [kw.value for kw in node.keywords if kw.arg == param]
                if index < len(node.args):
                    given.append(node.args[index])
                found.setdefault("env helper arguments", set()).update(
                    *map(named, given)
                )
    found["*_ENV and *_ENV_VAR strings"] = set().union(
        *(
            values
            for name, values in strings.items()
            if name.endswith(("_ENV", "_ENV_VAR"))
        )
    )
    known = set().union(*found.values())
    found["env-name collections"] = set().union(
        *(
            values
            for name, values in collections.items()
            if values & known or _ENV_COLLECTION_WORDS & set(_words(name))
        )
    )
    for path in sorted((_REPO / "docs").rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        found.setdefault("docs: BENCHFLOW_* and env tables", set()).update(
            *(pattern.findall(text) for pattern in _DOC_ENV_NAMES)
        )
        for block in _DOC_FENCE.findall(text):
            found.setdefault("docs: fenced blocks", set()).update(
                name
                for match in _DOC_FENCE_ENV.findall(block)
                for name in match
                if name
            )
    return {
        source: {name for name in names if _ENV_READ_NAME.fullmatch(name)}
        for source, names in found.items()
    }


@pytest.mark.parametrize("flag", [False, True], ids=["key", "flag"])
@pytest.mark.parametrize("name", REVIEW_FORMS)
def test_a_key_run_together_is_a_credential_in_every_form(name, flag):
    """Review of PR #145, 2026-11-13, L4: the keys run together were whole
    words only, so after a glued prefix (``myaccesskey``, as a key and as
    the flag ``--myaccesskey``) the name had no credential word and the
    override record stored its value. A word that ends in one of them names
    a credential."""
    assert name_kind(_flag(name) if flag else name) == "credential"


@pytest.mark.parametrize("name", REVIEW_FORMS)
def test_a_number_under_a_key_run_together_is_masked(name):
    """Review of PR #145, 2026-11-13, L4, the redactor's side: an integer
    of four digits or more under such a name is masked."""
    assert is_secret_number(name, 12345678)


@pytest.mark.parametrize("name", GLUED_PREFIXES)
def test_a_key_run_together_after_another_glued_prefix(name):
    """Review of PR #145, 2026-11-13, L4: prefixes beyond ``my``."""
    assert name_kind(name) == "credential"


def test_the_review_forms_cover_every_key_run_together():
    """Review of PR #145, 2026-11-13, L4: a key the classifier adds as an
    ending is added to GLUED_KEYS, and so to the review's forms."""
    assert sorted(credential_names._GLUED_KEYS) == sorted(GLUED_KEYS)


def test_the_ending_rule_changes_only_the_names_it_is_for():
    """Review of PR #145, 2026-11-13, L4: ``key`` is no ending (hotkey,
    monkey, turkey, keyboard, passkey), a schema's or a cache's key and the
    token settings keep their kind, and of every name the table pins only
    the ``my<key>`` names change, and xapikey, which no list names whole
    any more: it ends in apikey."""
    assert {name: name_kind(name) for name in KEPT} == KEPT
    table = [*SETTINGS, *CREDENTIALS, *NEITHER, *KEPT]
    changed = {f"my{key}": (None, "credential") for key in GLUED_KEYS}
    assert _changed(table) == {**changed, "xapikey": (None, "credential")}


@pytest.mark.parametrize("key", credential_names._GLUED_KEYS)
def test_every_listed_key_run_together_is_an_ending(key):
    """The lead's ruling on the review of PR #145, 2026-11-13, L4: every
    word of the one list of keys run together works as an ending, so none
    is a whole word only: alone, after a glued prefix, upper-case and with
    a plural s."""
    upper = key.upper()
    forms = [key, upper, f"my{key}", f"aws{key}", f"MY{upper}", f"{key}s", f"my{key}s"]
    assert {form: name_kind(form) for form in forms} == dict.fromkeys(
        forms, "credential"
    )


def test_a_key_run_together_is_listed_as_an_ending_not_as_a_whole_word():
    """The lead's ruling on the review of PR #145, 2026-11-13, L4: a word
    that ends in ``key`` or ``keyid`` put among the whole words would be
    matched whole only (``mytenantkey`` stored); it belongs in _GLUED_KEYS.
    Bare ``key`` is a whole word and no ending."""
    whole = credential_names._CREDENTIAL_WORDS
    assert {word for word in whole if word.endswith(("key", "keyid"))} == {"key"}
    assert "key" not in credential_names._CREDENTIAL_ENDINGS
    assert not set(credential_names._GLUED_KEYS) & whole
    assert all(
        key.endswith(("key", "keyid")) and key not in ("key", "keyid")
        for key in credential_names._GLUED_KEYS
    )


@pytest.mark.xdist_group("credential-name-corpus")
def test_the_ending_rule_makes_no_ordinary_word_a_credential():
    """Review of PR #145, 2026-11-13, L4, measured: no name and no word of
    a corpus of ordinary ones changes kind under the ending rule. The
    corpus is Python's keywords, builtins and standard modules, every
    identifier and string dict key in benchflow's src/, every key of the
    config examples under docs/ and tests/fixtures/, each also as its
    words, and /usr/share/dict/words where the machine has one."""
    identifiers = set().union(*(_identifiers(nodes) for nodes, _ in _source_index()))
    config_keys = _config_keys()
    assert len(identifiers) > 10_000 and len(config_keys) > 500
    names = identifiers | config_keys | set(keyword.kwlist) | set(dir(builtins))
    names |= set(sys.stdlib_module_names)
    dictionary = Path("/usr/share/dict/words")
    if dictionary.is_file():
        names |= set(dictionary.read_text(encoding="utf-8", errors="replace").split())
    names |= {word for name in names for word in _words(name)}
    assert _changed(names) == {}


@pytest.mark.xdist_group("credential-name-corpus")
def test_no_token_count_or_cost_field_ends_in_a_key_run_together():
    """The lead's ruling on the review of PR #145, 2026-11-13, L4: the
    number rule masks an integer under a credential's name, so no field that
    token and cost accounting reads may end in a key run together. For every
    usage, cost and price field name benchflow has (_accounting_names): no
    word of it ends in one of _GLUED_KEYS, and its kind is the same with the
    ending rule off, as the rc had it, so none is a credential by the rule."""
    names = _accounting_names()
    assert len(names) > 500
    glued = {
        name: word
        for name in sorted(names)
        for word in _words(name)
        if word.removesuffix("s").endswith(credential_names._GLUED_KEYS)
        or word.endswith(credential_names._GLUED_KEYS)
    }
    assert not glued, f"usage or cost fields ending in a key run together: {glued}"
    changed = _changed(names)
    assert not changed, f"usage or cost fields the ending rule changes: {changed}"


#: The lead's rule for ``token`` (review of PR #145, 2026-11-13, L4, and
#: the lows that followed): it names no credential after ``per`` and at most
#: one word (a price), just before a qualifier, or plural beside a companion.
TOKEN_SETTINGS = [
    # the two the lead named, lower-case and camelCase too
    "MODEL_COST_PER_TOKEN", "model_cost_per_token", "modelCostPerToken",
    "BENCHFLOW_CAPTURE_TOKEN_LOGPROBS", "benchflow_capture_token_logprobs",
    "benchflowCaptureTokenLogprobs",
    # per just before token or tokens, or per and a token's kind
    "per_token", "usd_per_token", "PRICE_PER_TOKEN", "input_cost_per_token",
    "inputCostPerToken", "output_cost_per_reasoning_token",
    "price_per_million_tokens", "cost_per_token_usd",
    # a priced token is no credential word: a second token word is a count
    "input_cost_per_token_above_100k_tokens",
    "cache_read_input_token_cost_above_100k_tokens",
    # each qualifier, and a plural one
    "token_capture", "cache_read_input_token_cost", "token_costs",
    "ROBO_ORACLE_TOKEN_FILE", "token_logprobs", "last_token_usage",
    "TOKEN_USAGE",
    # plural tokens beside a companion
    "MAX_TOKENS", "max_output_tokens", "n_input_tokens", "total_tokens",
]  # fmt: skip
#: Names whose ``token`` stays a credential's: alone, a provider's, beside
#: another credential word, ``per`` two words away, or before a word that is
#: no qualifier.
TOKEN_CREDENTIALS = [
    "GITHUB_TOKEN", "HF_TOKEN", "X_API_TOKEN", "OPENAI_API_TOKEN",
    "ACCESS_TOKEN", "REFRESH_TOKEN", "SESSION_TOKEN", "BEARER_TOKEN", "TOKEN",
    "token_secret", "per_token_secret", "TOKEN_LIMIT_SECRET",
    "developer_token", "super_secret_token", "cost_per_api_access_token",
    "TOKEN_FILE_PATH", "TOKEN_URL", "TOKEN_ENV", "TOKEN_NAME", "TOKEN_VALUE",
    "TOKEN_KEY", "TOKEN_SECRET", "TOKEN_USER", "token_mask",
]  # fmt: skip
#: Every env var name of the survey (_env_names) the classifier calls a
#: credential, read one by one: each holds a secret (a key, a token, a
#: password, a login file, or a git config value that can be an
#: Authorization header).
CREDENTIAL_ENV_NAMES = [
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "AWS_ACCESS_KEY_ID",
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AZURE_API_KEY",
    "BENCHFLOW_API_KEY",
    "BENCHFLOW_HUB_KEY",
    "BENCHFLOW_HUB_TOKEN",
    "BENCHFLOW_LITELLM_MASTER_KEY",
    "BENCHFLOW_PROVIDER_API_KEY",
    "BENCHFLOW_TASKMD_JUDGE_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_OAUTH_TOKEN",
    "CODEX_ACCESS_TOKEN",
    "CODEX_API_KEY",
    "CODEX_AUTH_JSON",
    "DAYTONA_API_KEY",
    "DEEPSEEK_API_KEY",
    "DEFAULT_AUTH_REQUEST",
    "DOUBAO_SEED_2_LITE_API_KEY",
    "DOUBAO_SEED_2_PRO_API_KEY",
    "GEMINI_API_KEY",
    "GITHUB_TOKEN",
    "GIT_CONFIG_VALUE_0",  # actions/checkout's http.extraheader
    "GIT_CONFIG_VALUE_1",  # actions/checkout's http.extraheader
    "GIT_CONFIG_VALUE_2",  # actions/checkout's http.extraheader
    "GIT_CONFIG_VALUE_3",  # actions/checkout's http.extraheader
    "GIT_CONFIG_VALUE_4",  # actions/checkout's http.extraheader
    "GLM_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS_JSON",
    "GOOGLE_GENERATIVE_AI_API_KEY",
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUNYUAN_API_KEY",
    "KIMI_API_KEY",
    "LITELLM_API_KEY",
    "LITELLM_MASTER_KEY",
    "LLM_API_KEY",
    "META_API_KEY",
    "MINIMAX_API_KEY",
    "MODAL_TOKEN_SECRET",
    "MUSE_AUTH_JSON",
    "NCBI_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "PG_PASSWORD",
    "PRIMEINTELLECT_API_KEY",
    "PRIME_API_KEY",
    "QWEN_API_KEY",
    "ROBOUSE_ORACLE_TOKEN",
    "ROBO_ORACLE_TOKEN",
    "TOOLATHLON_GITHUB_TOKEN",
    "TOOLATHLON_GOOGLE_CLIENT_SECRET",
    "TOOLATHLON_GOOGLE_REFRESH_TOKEN",
    "TOOLATHLON_HF_TOKEN",
    "TOOLATHLON_MAPS_API_KEY",
    "TOOLATHLON_NOTION_KEY",
    "TOOLATHLON_NOTION_KEY_EVAL",
    "TOOLATHLON_SERPER_API_KEY",
    "TOOLATHLON_WANDB_API_KEY",
    "WANDB_API_KEY",
    "XIAOMI_API_KEY",
    "ZAI_API_KEY",
]
#: The survey's names with a credential's word that the classifier does not
#: call a credential, with their kind: none holds a secret. The ones the
#: lead's ruling on the 18 covers are settings by a last-word qualifier
#: (AUTH_ENABLED, BENCHFLOW_HUB_KEY_FILE) or by name (GIT_CONFIG_KEY_<n>,
#: TIKTOKEN_CACHE_DIR, GOOGLE_APPLICATION_CREDENTIALS).
NOT_CREDENTIAL_ENV_NAMES = {
    "AUTH_ENABLED": "setting",
    "AWS_SHARED_CREDENTIALS_FILE": "setting",
    "BENCHFLOW_CAPTURE_TOKEN_LOGPROBS": "setting",
    "BENCHFLOW_HUB_KEY_FILE": "setting",
    "BENCHFLOW_HUB_KEY_NO_CHECKSUM": "setting",
    "BENCHFLOW_KEEP_AGENT_SESSION_LOGS": "setting",
    "BENCHFLOW_KEY_CHECK": "setting",
    "BENCHFLOW_MAX_TOKENS": "setting",
    "BENCHFLOW_MODEL_MAX_TOKENS": "setting",
    "BENCHFLOW_TASKMD_MAX_JUDGE_SESSIONS": "setting",
    "BENCHFLOW_TASK_TOML_ALLOW_UNKNOWN_KEYS": "setting",
    "BENCHFLOW_TRIAL_TOKEN_BUDGET": "setting",
    "CLAUDE_CODE_CUSTOM_OAUTH_URL": None,
    "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "setting",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "setting",
    "CLAUDE_LOCAL_OAUTH_API_BASE": None,
    "CLAUDE_LOCAL_OAUTH_APPS_BASE": None,
    "CLAUDE_LOCAL_OAUTH_CONSOLE_BASE": None,
    "CUSTOM_TIKTOKEN_CACHE_DIR": "setting",
    "DBUS_SESSION_BUS_ADDRESS": "setting",
    "GIT_CONFIG_KEY_0": "setting",
    "GIT_CONFIG_KEY_1": "setting",
    "GIT_CONFIG_KEY_2": "setting",
    "GIT_CONFIG_KEY_3": "setting",
    "GIT_CONFIG_KEY_4": "setting",
    "GOOGLE_APPLICATION_CREDENTIALS": "setting",
    "MAX_THINKING_TOKENS": "setting",
    "MAX_TOKENS": "setting",
    "MODAL_TOKEN_ID": "setting",
    "OAUTH_POOL": None,
    "ROBO_ORACLE_TOKEN_FILE": "setting",
    "SESSION_TIMEOUT": "setting",
    "SSH_AUTH_SOCK": "setting",
    "TIKTOKEN_CACHE_DIR": "setting",
}
_SECRET_WORDS = frozenset(
    {"token", "tokens", "key", "keys", "secret", "secrets", "password",
     "passwd", "pass", "pwd", "auth", "oauth", "session", "sessions",
     "credential", "credentials", "cookie", "bearer", "sid"}
)  # fmt: skip


@pytest.mark.parametrize("name", TOKEN_SETTINGS)
def test_a_price_or_a_qualified_token_names_a_setting(name):
    """The lead's ruling on the lows of PRs #145 and #148, 2026-11-13: a
    price per token (MODEL_COST_PER_TOKEN) or a token's qualifier
    (BENCHFLOW_CAPTURE_TOKEN_LOGPROBS) names a setting, which the override
    record stores and compares: never mask a cost, and show the price a run
    used. ``per`` is a price's just before the token, before a token's kind
    or in a name with a price word (the lead's narrowing)."""
    assert name_kind(name) == "setting"


@pytest.mark.parametrize("name", TOKEN_CREDENTIALS)
def test_a_token_with_no_price_or_qualifier_stays_a_credential(name):
    """The lead's ruling on the lows of PRs #145 and #148, 2026-11-13: a
    bare or provider token, a second credential word (TOKEN_LIMIT_SECRET)
    and a word that is no qualifier (TOKEN_URL, TOKEN_FILE_PATH)
    keep the name a credential's."""
    assert name_kind(name) == "credential"


@pytest.mark.xdist_group("credential-name-corpus")
def test_every_env_name_the_classifier_calls_a_credential_is_listed():
    """The lead's ruling on the lows of PRs #145 and #148, 2026-11-13: the
    classifier over every environment variable name BenchFlow reads or
    documents (_env_names). The ones it calls a credential are
    CREDENTIAL_ENV_NAMES, so a new one fails here until it is added on
    purpose; the ones with a credential's word that it does not are
    NOT_CREDENTIAL_ENV_NAMES, with their kind."""
    names = set().union(*_env_names().values())
    assert len(names) > 400
    assert sorted(CREDENTIAL_ENV_NAMES) == CREDENTIAL_ENV_NAMES
    kinds = {name: name_kind(name) for name in names}
    credentials = {name for name, kind in kinds.items() if kind == "credential"}
    assert credentials == set(CREDENTIAL_ENV_NAMES)
    worded = {
        name: kind
        for name, kind in kinds.items()
        if kind != "credential"
        and (
            _SECRET_WORDS & set(_words(name))
            or any(map(credential_names._credential_word, _words(name)))
        )
    }
    assert worded == NOT_CREDENTIAL_ENV_NAMES


#: A last word, just after the credential word, that names the credential's
#: file, path, socket, directory, switch, check or logs makes a setting (the
#: lead's ruling on the 18 env names, review of PR #145, 2026-11-13).
LAST_WORD_SETTINGS = [
    "AWS_SHARED_CREDENTIALS_FILE", "BENCHFLOW_HUB_KEY_FILE", "ROBO_ORACLE_TOKEN_FILE",
    "password_file", "TOKEN_PATH", "GITHUB_TOKEN_PATH", "accesskey_path",
    "SSH_AUTH_SOCK", "secrets_dir", "AUTH_ENABLED", "api_key_enabled",
    "BENCHFLOW_KEY_CHECK", "BENCHFLOW_KEEP_AGENT_SESSION_LOGS",
]  # fmt: skip
#: Settings by name, where no qualifier reaches: none holds a secret.
NOT_SECRET_NAMES = [
    "GIT_CONFIG_KEY_0", "GIT_CONFIG_KEY_5", "GIT_CONFIG_KEY_12", "git_config_key_3",
    "DBUS_SESSION_BUS_ADDRESS", "GOOGLE_APPLICATION_CREDENTIALS",
    "google_application_credentials", "TIKTOKEN_CACHE_DIR",
    "CUSTOM_TIKTOKEN_CACHE_DIR", "BENCHFLOW_HUB_KEY_NO_CHECKSUM",
    "BENCHFLOW_TASK_TOML_ALLOW_UNKNOWN_KEYS",
    # a tokenizer's switches and counts (ruled in review, 2026-11-13)
    "add_special_tokens", "addSpecialTokens", "return_assistant_tokens_mask",
    "generate_tokens", "custom_trainer_pretokenized_trainable_tokens",
]  # fmt: skip
#: Credentials beside those rules: a git config value always, a name the
#: list does not hold, another credential word, or a word between.
UNQUALIFIED_CREDENTIALS = [
    "GIT_CONFIG_VALUE_0", "GIT_CONFIG_VALUE_5", "git_config_value_10",
    "GIT_CONFIG_KEY", "GOOGLE_APPLICATION_CREDENTIALS_JSON", "TIKTOKEN_API_KEY",
    "SESSION_TOKEN_FILE", "TOKEN_FILE_PATH", "API_KEY_HASH_CHECK",
    "KEY_FILE_CONTENTS",
    # a key's checksum is metadata about a secret (ruled in review, 2026-11-13)
    "key_checksum", "API_KEY_CHECKSUM",
]  # fmt: skip
_FAKE_KEY = "sk-ant-api03-" + "FAKEfakeFAKEfake0123456789" * 3 + "-FAKEfakeAA"


#: The lead's follow-up PR (2026-11-14): a last-word qualifier with a
#: number between it and the credential word, alone or with up to three
#: letters beside it (2fa, v2), and a password's hint, are settings.
QUALIFIED_SETTINGS = [
    "TOKEN_2FA_ENABLED", "PASSWORD_2_HINT", "PASSWORD_HINT", "passphrase_hint",
    "DB_PASS_HINT", "dbPasswordHint", "KEY_2_FILE", "TOKEN_V2_PATH",
    "api_key_10_var", "SECRET_1_DIR",
]  # fmt: skip
#: The same names' credentials: a key, a token or a session and a number
#: (KEY_2, TOKEN_12345, SESSION_2), a token with a version or a second
#: factor (TOKEN_V2: CircleCI's CIRCLE_OIDC_TOKEN_V2 holds an OIDC token;
#: TOKEN_2FA), bare SESSION, a hint after a word that is no password's
#: (id_token_hint holds an ID token, key_hint a key's first letters), a
#: longer word or two words between (TOKEN_2_FILE_PATH, KEY_X509C_FILE),
#: and the three the lead asked about for which no rule by name is safe
#: (SECRET_2_OF_3, PRIMARY_KEY_2, TOKENS_2; reported to the lead).
NUMBERED_CREDENTIALS = [
    "KEY_2", "KEY_10", "TOKEN_12345", "TOKEN_2FA", "TOKEN_V2", "SESSION",
    "SESSION_2", "CIRCLE_OIDC_TOKEN_V2", "id_token_hint", "key_hint",
    "TOKEN_2_FILE_PATH", "KEY_X509C_FILE", "SECRET_2_OF_3", "PRIMARY_KEY_2",
    "TOKENS_2",
]  # fmt: skip
#: by_pass in snake case is two words, by and a password's pass: the lead's
#: ruling (2026-11-14) names bypass and ByPass only, one word glued.
TWO_WORD_BYPASS = ["by_pass", "BY_PASS", "by-pass"]
#: Names the agreement test (tests/test_redaction_classifier_agreement.py)
#: compares besides its own sets (the lead's ruling, 2026-11-14): the four
#: of the smaller disagreements and the numbered names.
COMPARED_NAMES = sorted(
    {
        "authorization_url", "session_cookie_name", "cookie_policy",
        "http2cookie", *QUALIFIED_SETTINGS, *NUMBERED_CREDENTIALS,
        *TWO_WORD_BYPASS,
    }
)  # fmt: skip
#: The one word splitter (name_words), as the redactor reads it too: digits
#: are words of their own, camel humps and acronyms split, separators and
#: any other character between words drop, and a letter outside ASCII is
#: read as the ASCII letters of its case fold.
SPLITS = [
    ("http2cookie", ["http", "2", "cookie"]),
    ("oauth2token", ["oauth", "2", "token"]),
    ("base64key", ["base", "64", "key"]),
    ("TOKEN_V2", ["token", "v", "2"]),
    ("dbPassword2", ["db", "password", "2"]),
    ("Token2FA", ["token", "2", "fa"]),
    ("HTTPCookie", ["http", "cookie"]),
    ("AWSSecretKey", ["aws", "secret", "key"]),
    ("APIKey", ["api", "key"]),
    ("connect.sid", ["connect", "sid"]),
    ("x-api-key", ["x", "api", "key"]),
    ("a/b token", ["a", "b", "token"]),
    ("pass@k", ["pass", "k"]),
    ("PA\u1e9eWORD", ["password"]),
    ("pa\u00dfword", ["password"]),
    ("\ufb01le_token", ["file", "token"]),
    ("\u212aEY", ["key"]),
    ("ByPass", ["bypass"]),
    ("isByPassCache", ["is", "bypass", "cache"]),
    ("by_pass", ["by", "pass"]),
    ("caf\u00e9_key", ["caf", "key"]),
]


@pytest.mark.parametrize("name", TWO_WORD_BYPASS)
def test_by_pass_written_as_two_words_names_a_password(name):
    """The lead's ruling (2026-11-14) makes bypass one word in ByPass too;
    with a separator it is still two, and pass a password's word."""
    assert name_kind(name) == "credential"


@pytest.mark.parametrize(("name", "words"), SPLITS)
def test_the_one_word_splitter(name, words):
    """The lead's ruling (2026-11-14): one splitter for the classifier and
    the redactor; the classifier's own name for it is _words."""
    assert credential_names.name_words(name) == words
    assert _words(name) == words


def test_a_letter_whose_case_fold_is_ascii_names_what_its_fold_names():
    """The redactor's snake case read PA\u1e9eWORD as password (its case
    fold); the one splitter does too, so the switch masks no less."""
    assert name_kind("PA\u1e9eWORD") == "credential"
    assert name_kind("http2cookie") == "credential"
    assert entry_kind("http2cookie", "theme=dark; sid=FAKEsid0123") == "credential"


@pytest.mark.parametrize("name", QUALIFIED_SETTINGS)
def test_a_qualifier_after_a_number_or_a_passwords_hint_names_a_setting(name):
    """The lead's follow-up PR (2026-11-14): TOKEN_2FA_ENABLED and
    PASSWORD_2_HINT are settings by a rule, not a list: a number between
    the credential word and its last-word qualifier keeps the qualifier, and
    hint is one after a password's word."""
    assert name_kind(name) == "setting"
    assert entry_kind(name, "north") == "setting"


@pytest.mark.parametrize("name", NUMBERED_CREDENTIALS)
def test_a_numbered_key_token_or_session_stays_a_credential(name):
    """The lead's follow-up PR (2026-11-14): the redactor masks KEY_<n> and
    TOKEN_<5+ digits> as the classifier calls them credentials; these stay
    credentials whatever they hold."""
    assert name_kind(name) == "credential"
    assert entry_kind(name, "north") == "credential"
    assert entry_kind(name, "12") == "credential"


@pytest.mark.parametrize("name", LAST_WORD_SETTINGS)
def test_a_credentials_file_path_socket_or_switch_names_a_setting(name):
    """The lead's ruling on the 18 env names (review of PR #145,
    2026-11-13): a qualifier as the last word, just after the credential
    word, names a setting, which the override record stores and compares."""
    assert name_kind(name) == "setting"


@pytest.mark.parametrize("name", NOT_SECRET_NAMES)
def test_a_name_listed_as_no_secret_names_a_setting(name):
    """The lead's ruling on the 18 env names (review of PR #145,
    2026-11-13): the names no qualifier reaches are settings by name."""
    assert name_kind(name) == "setting"


@pytest.mark.parametrize("name", UNQUALIFIED_CREDENTIALS)
def test_a_git_config_value_or_an_unqualified_name_stays_a_credential(name):
    """The lead's ruling on the 18 env names (review of PR #145,
    2026-11-13): GIT_CONFIG_VALUE_<n> is always a credential, and the
    qualifier and the list reach no further than they say."""
    assert name_kind(name) == "credential"


def _overlay_record():
    """``config_override.overlay_record``, or a skip on a tree that has no
    override record yet (it came with PR #145)."""
    from benchflow._utils import config_override

    if not hasattr(config_override, "overlay_record"):
        pytest.skip("no override record on this tree (PR #145)")
    return config_override.overlay_record


def test_a_git_config_value_is_withheld_beside_its_stored_key():
    """The lead's ruling on the 18 env names (review of PR #145,
    2026-11-13): actions/checkout passes its token as GIT_CONFIG_VALUE_<n>
    (http.extraheader: Authorization: Bearer). Under an env mapping the
    value is withheld, neither stored nor digested; the key's name beside it
    is stored and compared."""
    overlay_record = _overlay_record()
    key = "http.https://example.test/.extraheader"
    value = "AUTHORIZATION: bearer FAKEextraheader0123456789"
    env = {"GIT_CONFIG_KEY_0": key, "GIT_CONFIG_VALUE_0": value}
    record = overlay_record({"sandbox": {"env": env}})
    assert record["withheld"] == ["sandbox.env.GIT_CONFIG_VALUE_0"]
    assert record["patch"] == {"sandbox": {"env": {"GIT_CONFIG_KEY_0": key}}}
    compare = {"GIT_CONFIG_KEY_0": key, "GIT_CONFIG_VALUE_0": "<withheld>"}
    assert record["compare"] == {"sandbox": {"env": compare}}
    assert "FAKEextraheader" not in json.dumps(record)


def test_a_key_shaped_value_under_a_setting_name_is_withheld_all_the_same():
    """The lead's ruling on the 18 env names (review of PR #145,
    2026-11-13): value redaction still applies to every value the record
    keeps. A key shape inside the path under BENCHFLOW_HUB_KEY_FILE is
    withheld as it is under a harmless name, under an env mapping and
    elsewhere; a plain path is stored."""
    overlay_record = _overlay_record()
    plain = "/home/me/.config/benchflow/hub.key"
    shaped = f"/home/me/{_FAKE_KEY}/hub.key"
    stored = overlay_record({"sandbox": {"env": {"BENCHFLOW_HUB_KEY_FILE": plain}}})
    assert stored["patch"] == {"sandbox": {"env": {"BENCHFLOW_HUB_KEY_FILE": plain}}}
    for name, harmless, section in (
        ("BENCHFLOW_HUB_KEY_FILE", "MY_NOTE", ("sandbox", "env")),
        ("benchflow_hub_key_file", "note", ("metadata",)),
    ):

        def record(key: str, section: tuple[str, ...] = section) -> dict:
            overlay: dict = {key: shaped}
            for part in reversed(section):
                overlay = {part: overlay}
            return overlay_record(overlay)

        kept, other = record(name), record(harmless)
        assert kept["withheld"] == [".".join((*section, name))]
        compared, unnamed = kept["compare"], other["compare"]
        for part in section:
            compared, unnamed = compared[part], unnamed[part]
        assert compared[name] == unnamed[harmless]
        assert _FAKE_KEY not in json.dumps(kept)


#: Prices by the narrowed rule: per just before the token, per and a token's
#: kind, or per and any one word (a number with its unit too) in a name with
#: a price word (cost, price, usd).
PER_PRICES = [
    "MODEL_COST_PER_TOKEN", "COST_PER_INPUT_TOKEN", "input_cost_per_token",
    "output_cost_per_reasoning_token", "usd_per_token", "price_per_output_token",
    "usd_per_request_token", "cost_per_user_token", "usd_per_million_tokens",
    "cost_per_1k_tokens", "input_cost_per_video_token",
]  # fmt: skip
#: Per and a word that is no kind, with no price word: a credential's name.
PER_CREDENTIALS = [
    "per_user_api_token", "PER_TENANT_TOKEN", "perUserToken",
    "X-Per-Request-Token", "per_user_tokens", "per_1k_tokens",
]  # fmt: skip
#: The six the review found stored, each with a made-up opaque value.
PER_TOKEN_PROBES = {
    "PER_USER_TOKEN": "pu-FAKEopaque0123456789abcdef",
    "MCP_PER_USER_TOKEN": "gAAAAABFAKEencrypted0123456789abcdef",
    "PER_SERVER_TOKEN": "ps-FAKEopaque0123456789abcdef",
    "GITLAB_PER_PROJECT_TOKEN": "FAKEgitlab0123456789abcd",
    "PER_ORG_TOKEN": "po-FAKEopaque0123456789abcdef",
    "per_request_token": "pr-FAKEopaque0123456789abcdef",
}
#: Numbers the redactor keeps: a price is no secret (ruled in review,
#: 2026-11-13). Until this rule, an integer of four digits or more under a
#: price with a credential word as its unit was masked.
KEPT_NUMBERS = [
    "code_interpreter_cost_per_session", "cost_per_request", "usd_per_1k",
    # and a pass metric or outcome (ruling C): until it, masked too
    "n_pass", "pass_count", "pass_score", "pass_at_1", "PASS_RATE", "pass_at_k",
    "tests_pass", "fail_to_pass",
]  # fmt: skip
#: Passwords the redactor still masks, with a made-up value.
MASKED_NUMBERS = {
    "DB_PASS": 123456,
    "SMTP_PASS": 12345678,
    "admin_pass": 123456,
    "DB_PASS_2": 123456,
}


@pytest.mark.parametrize("name", PER_PRICES)
def test_a_price_per_token_names_a_setting(name):
    """The lead's narrowing of the price rule (review of 2026-11-13): a
    cost per token stays a setting, stored and compared."""
    assert name_kind(name) == "setting"


@pytest.mark.parametrize("name", [*PER_CREDENTIALS, *PER_TOKEN_PROBES])
def test_a_token_per_user_or_request_names_a_credential(name):
    """The lead's narrowing of the price rule (review of 2026-11-13): per
    and a word that is no token's kind, in a name with no price word, is a
    token per something, a credential (LiteLLM's MCP_PER_USER_TOKEN)."""
    assert name_kind(name) == "credential"


@pytest.mark.parametrize("name", list(PER_TOKEN_PROBES))
def test_a_token_per_user_is_withheld_and_a_price_beside_it_stored(name):
    """The lead's narrowing of the price rule (review of 2026-11-13): under
    an env mapping a made-up opaque value under each of the six is withheld
    and never stored, and a price beside it is stored as given."""
    overlay_record = _overlay_record()
    value = PER_TOKEN_PROBES[name]
    prices = {price: "0.000003" for price in PER_PRICES}
    record = overlay_record({"sandbox": {"env": {name: value, **prices}}})
    assert record["withheld"] == [f"sandbox.env.{name}"]
    assert record["patch"] == {"sandbox": {"env": prices}}
    assert value not in json.dumps(record)


def _litellm_price_fields() -> set[str]:
    """The field names of LiteLLM's price table, read from the installed
    package's JSON without importing it, or a skip when it is absent."""
    spec = importlib.util.find_spec("litellm")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("litellm is not installed")
    root = Path(next(iter(spec.submodule_search_locations)))
    tables = sorted(root.glob("model_prices_and_context_window*.json"))
    if not tables:
        pytest.skip("litellm has no price table here")
    return {
        field
        for table in tables
        for model in json.loads(table.read_text(encoding="utf-8")).values()
        if isinstance(model, dict)
        for field in model
    }


def test_no_field_of_litellms_price_table_is_a_credential():
    """Ruled in review, 2026-11-13: cost fields are never withheld or
    masked. Every field of LiteLLM's price table is a setting or names
    nothing; code_interpreter_cost_per_session, the one that was a
    credential, is a settings row of the table above."""
    fields = _litellm_price_fields()
    assert len(fields) > 100
    credentials = sorted(field for field in fields if name_kind(field) == "credential")
    assert not credentials, f"price fields named as credentials: {credentials}"


@pytest.mark.parametrize("name", KEPT_NUMBERS)
def test_the_redactor_keeps_a_number_under_a_price_or_a_pass_metric(name):
    """Ruled in review, 2026-11-13: cost fields and our core metrics are
    never masked. The redactor's number rule asks name_kind, so an integer
    under a price or a pass metric stays in trajectories."""
    assert not is_secret_number(name, 1234)
    assert redact_trajectory_obj({name: 1234}) == {name: 1234}


#: The 21 names a provider's word anywhere in the name made credentials,
#: from the corpus (site-packages), with what each is by ruling A: only a
#: provider's word just before tokens lists tokens.
PROVIDER_CORPUS_NAMES = {
    "OPENAI_RESPONSES_API_MIN_MAX_OUTPUT_TOKENS": "setting",
    "RESPONSES_API_MIN_OUTPUT_TOKENS": "setting",
    "client.aio.models.count_tokens": "setting",
    "client.count_tokens": "setting",
    "client.models.count_tokens": "setting",
    "get_cached_refresh_tokens": "credential",
    "google.cloud.aiplatform_v1.LlmUtilityServiceAsyncClient.count_tokens": "setting",
    "google.cloud.aiplatform_v1.LlmUtilityServiceClient.CountTokens": "setting",
    "google.cloud.aiplatform_v1.LlmUtilityServiceClient.count_tokens": "setting",
    "google.cloud.aiplatform_v1beta1.PredictionServiceAsyncClient.count_tokens": "setting",
    "google.cloud.aiplatform_v1beta1.PredictionServiceClient.CountTokens": "setting",
    "google.cloud.aiplatform_v1beta1.PredictionServiceClient.count_tokens": "setting",
    "response_api_usage.input_tokens": "setting",
    "response_api_usage.input_tokens_details": "setting",
    "response_api_usage.output_tokens": "setting",
    "self._client._transport.count_tokens": "setting",
    "self._client.get_cached_refresh_tokens": "credential",
    "self._llm_utility_async_client.count_tokens": "setting",
    "self._llm_utility_client.count_tokens": "setting",
    "self._prediction_async_client.count_tokens": "setting",
    "self._prediction_client.count_tokens": "setting",
}


@pytest.mark.parametrize("name", list(PROVIDER_CORPUS_NAMES))
def test_a_provider_word_lists_tokens_only_just_before_them(name):
    """Ruling A (review of 2026-11-13): a provider's word just before
    tokens makes the name a credential unless a pure count word is in it;
    elsewhere in the name it leaves a count a count (LiteLLM's
    response_api_usage.input_tokens, a client's count_tokens)."""
    assert name_kind(name) == PROVIDER_CORPUS_NAMES[name]


def test_id_is_no_provider_for_the_tokenizer_and_id_rules():
    """Ruling B (review of 2026-11-13): id is a provider's word just before
    tokens (id_tokens), but not for the tokenizer and id rules, whose own
    word it is."""
    assert "id" in credential_names._PROVIDER_WORDS
    kinds = {
        name: name_kind(name) for name in ("token_id", "eos_token_id", "pad_token_id")
    }
    assert kinds == dict.fromkeys(kinds, "setting")


@pytest.mark.parametrize("name", list(MASKED_NUMBERS))
def test_the_redactor_still_masks_a_number_under_a_password(name):
    """Ruling C (review of 2026-11-13): pass is a password's word outside
    its exits, and its number is masked (DB_PASS_2 too)."""
    value = MASKED_NUMBERS[name]
    assert is_secret_number(name, value)
    assert redact_trajectory_obj({name: value}) != {name: value}


#: A record file BenchFlow writes; a module that names one and writes JSON
#: is one of its writers.
_RECORD_FILE = re.compile(
    r"(result|summary|evaluation|config|job|run_summary|trial)\.json"
)
#: Field names of ours that hold a secret by design, each with what it holds.
SECRET_BY_DESIGN = {
    # an env var's name, a key of the continue orchestrator's provider env:
    # it holds the replay run's API key
    "LLM_API_KEY": "src/benchflow/continue_run/orchestrator.py",
}
#: Documented exceptions (the lead's ruling of 2026-11-13): record fields
#: of ours that share their name with a credential. Bare tokens, session
#: and key stay credentials, as changing them would reopen real leaks; an
#: EN issue (the classifier learns whether a name is a record field or an
#: env or override key) settles them. A real training-signal key (a dict of
#: strings) and a branch row's session (a short string) are kept by every
#: shipped redactor; only an integer of four digits or more under them would
#: be masked, and no writer makes one.
DOCUMENTED_EXCEPTIONS = {
    # a trial's group key, and the training-signal schema's group key
    "key": "src/benchflow/jobs.py",
    # a branch row's agent_session state (fresh, resumed)
    "session": "docs/reference/schemas/benchflow-branch-child-row.v2.schema.json",
    # a branch run's source agent session id
    "session_id": "src/benchflow/branch_run.py",
    # a token count (report and result rows, the model-calls schema,
    # taskmd's judge budget)
    "tokens": "src/benchflow/report.py",
}


def _writes_json(nodes: list[ast.AST]) -> bool:
    """Whether a module writes JSON: json.dump, a write_text, or a writer
    whose name says json."""
    for node in nodes:
        if isinstance(node, ast.Call):
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", "")
            )
            if name in ("dump", "write_text") or ("write" in name and "json" in name):
                return True
    return False


def _record_fields() -> dict[str, set[str]]:
    """The field names BenchFlow's record writers build, with the files they
    are in: in each module of src/ (not a vendored one) that names a record
    file and writes JSON, the string keys of its dict literals and of its
    ``[...] =`` assignments, and its classes' annotated fields (the models
    and dataclasses the records are written from)."""
    fields: dict[str, set[str]] = {}
    for path, (nodes, _) in zip(_source_paths(), _source_index(), strict=True):
        named = any(
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and _RECORD_FILE.fullmatch(node.value)
            for node in nodes
        )
        if "_vendor" in path.parts or not named or not _writes_json(nodes):
            continue
        where = str(path.relative_to(_REPO))
        names: list[str] = []
        for node in nodes:
            keys: list[ast.expr | None] = []
            if isinstance(node, ast.Dict):
                keys = node.keys
            elif isinstance(node, ast.Assign):
                keys = [t.slice for t in node.targets if isinstance(t, ast.Subscript)]
            elif isinstance(node, ast.ClassDef):
                names += [
                    item.target.id
                    for item in node.body
                    if isinstance(item, ast.AnnAssign)
                    and isinstance(item.target, ast.Name)
                ]
            names += [
                key.value
                for key in keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            ]
        for name in names:
            fields.setdefault(name, set()).add(where)
    return fields


def _schema_fields() -> dict[str, set[str]]:
    """Every property name, at every depth, of the published schemas."""
    fields: dict[str, set[str]] = {}
    for path in sorted((_REPO / "docs/reference/schemas").glob("*.json")):
        for name in _schema_properties(json.loads(path.read_text(encoding="utf-8"))):
            fields.setdefault(name, set()).add(str(path.relative_to(_REPO)))
    return fields


def _taskmd_keys() -> dict[str, set[str]]:
    """Every key a task file can set, as the vendored taskmd closes its
    tables: the strings of its module-level KEYS and *_KEYS constants
    (scoring, criteria, rubric, judge, outputs, services and the rest)."""
    path = _REPO / "src/benchflow/taskmd/_vendor/taskmd.py"
    nodes = _source_index()[_source_paths().index(path)][0]
    module = nodes[0]
    keys: dict[str, set[str]] = {}
    for node in module.body if isinstance(module, ast.Module) else []:
        if not isinstance(node, ast.Assign) or not any(
            isinstance(target, ast.Name)
            and (target.id == "KEYS" or target.id.endswith("_KEYS"))
            for target in node.targets
        ):
            continue
        for item in ast.walk(node.value):
            if isinstance(item, ast.Constant) and isinstance(item.value, str):
                keys.setdefault(item.value, set()).add(str(path.relative_to(_REPO)))
    return keys


def _our_credential_fields() -> dict[str, list[str]]:
    """Each field name of ours the classifier calls a credential, with the
    files it comes from."""
    flagged: dict[str, set[str]] = {}
    for fields in (_record_fields(), _schema_fields(), _taskmd_keys()):
        for name, files in fields.items():
            if name_kind(name) == "credential":
                flagged.setdefault(name, set()).update(files)
    return {name: sorted(files) for name, files in flagged.items()}


@pytest.mark.xdist_group("credential-name-corpus")
def test_no_field_of_our_schemas_records_or_task_files_is_a_credential():
    """The lead's guard (review of 2026-11-13): every field name of our
    published schemas, of the records our writers build (result.json,
    summary.json, evaluation.json, the job export and run summary) and of a
    task file's tables classifies as no credential, but a field that holds a
    secret by design (SECRET_BY_DESIGN) and the documented exceptions
    (DOCUMENTED_EXCEPTIONS). A new field of ours cannot become a password
    unnoticed."""
    assert len(_record_fields()) > 500
    assert len(_schema_fields()) > 300
    assert len(_taskmd_keys()) > 100
    flagged = _our_credential_fields()
    unexpected = {
        name: files
        for name, files in flagged.items()
        if name not in SECRET_BY_DESIGN and name not in DOCUMENTED_EXCEPTIONS
    }
    assert not unexpected, (
        f"fields of ours named as credentials: {unexpected}; add a rule exit "
        "for them, or put one that holds a secret into SECRET_BY_DESIGN on purpose"
    )
    assert set(SECRET_BY_DESIGN) <= set(flagged), "a secret-by-design field is gone"


@pytest.mark.parametrize("name", list(DOCUMENTED_EXCEPTIONS))
def test_a_documented_exception_is_still_a_credential(name):
    """The lead's ruling of 2026-11-13: these fields of ours stay
    credentials by name until the EN issue lets the classifier tell a
    record field from an env or override key; this fails, on purpose, when
    one stops being a credential, so the exception is lifted with it."""
    assert name_kind(name) == "credential"


#: Prices and credentials around per, as the classifier and the redactor are
#: to agree on them (the lead's ruling, review of PR #173, 2026-11-13): the
#: price rule yields when a provider's or a credential's word sits between
#: per and the token; user, request and the like are units with a price word.
PER_AGREEMENT = {
    "cost_per_api_access_token": "credential",
    "price_per_refresh_token": "credential",
    "usd_per_access_token": "credential",
    "cost_per_auth_token": "credential",
    "cost_per_user_token": "setting",
    "usd_per_request_token": "setting",
    "cost_per_input_token": "setting",
    "PER_USER_TOKEN": "credential",
}


@pytest.mark.parametrize("name", list(PER_AGREEMENT))
def test_a_credential_word_between_per_and_token_makes_no_price(name):
    """The lead's ruling (review of PR #173, 2026-11-13): a price word with
    per and a unit is a setting, unless a provider's or a credential's word
    sits between per and the token."""
    assert name_kind(name) == PER_AGREEMENT[name]


#: Entries whose kind depends on the value (the lead's option 2, review of
#: PR #173, 2026-11-13): a test's, an outcome's or a count's pass is a
#: setting while it holds a test result, and a credential otherwise; a
#: metric's name and a name by name do not depend on the value.
#: A primary or a secondary key's name holds a schema's key or an access key
#: (the lead's ruling, 2026-11-14): a key of 24 characters or more (an Azure
#: key of 44 or 88) is a credential, a short identifier, a number or a list
#: of them a setting. The key values are made up.
PRIMARY_KEY_ROWS = [
    ("primary_key", "FAKEq7W2e9R4t6Y1u3I5o8P0aSdFgHjKlZxCvBnM1a+=", "credential"),
    (
        "primaryKey",
        "FAKEq7W2e9R4t6Y1u3I5o8P0aSdFgHjKlZxCvBnM1a+FAKEq7W2e9R4t6Y1u3I5o8P0aSdFgHjKlZxCvBnM1a+=",
        "credential",
    ),
    ("secondaryKey", "FAKEq7W2e9R4t6Y1u3I5o8P0aSdFgHjKlZxCvBnM1a+=", "credential"),
    (
        "COSMOS_PRIMARY_KEY",
        "FAKEq7W2e9R4t6Y1u3I5o8P0aSdFgHjKlZxCvBnM1a+FAKEq7W2e9R4t6Y1u3I5o8P0aSdFgHjKlZxCvBnM1a+=",
        "credential",
    ),
    ("secondary_key", "sk-FAKE0aZ9bY7Lm2Qx4Nc8Vd6", "credential"),
    ("primary_key", "customer_account_number_id", "setting"),
    # an identifier of any length is a schema's, a long value with a digit
    # or with + / = a key (the lead's ruling, 2026-11-14)
    ("primary_key", 123456, "setting"),
    ("primary_key", "external_subscription_id", "setting"),
    ("primary_key", "distributionHostedZoneId", "setting"),
    ("primary_key", "FAKEq7W2e9R4t6Y1u3I5o8P0aSdFgHjKlZxCvBnM1aQ2", "credential"),
    ("primary_key", "FAKE0aZ9bY7Lm2Qx4Nc8Vd6W", "credential"),
    ("primary_key", "FAKE0aZ9bY7Lm2Qx4Nc8Vd6", "setting"),
    ("primary_key", ["external_subscription_id", "ts"], "setting"),
    ("secondary_key", "id", "setting"),
    (
        "storage_primary_key",
        "FAKEq7W2e9R4t6Y1u3I5o8P0aSdFgHjKlZxCvBnM1a+FAKEq7W2e9R4t6Y1u3I5o8P0aSdFgHjKlZxCvBnM1a+F=",
        "credential",
    ),
    ("PRIMARY_KEY_2", 12345, "credential"),
    ("PRIMARY_KEY_2", "FAKEq7W2e9R4t6Y1u3I5o8P0aSdFgHjKlZxCvBnM1a+=", "credential"),
    ("primary_key", {"columns": ["id"]}, "credential"),
    (
        "primary_key",
        ["id", "FAKEq7W2e9R4t6Y1u3I5o8P0aSdFgHjKlZxCvBnM1a+="],
        "credential",
    ),
    ("primary_key", "id", "setting"),
    ("primary_key", ["id", "ts"], "setting"),
    ("primary_key", '["id", "ts"]', "setting"),
    ("PRIMARY_KEY", "user_id", "setting"),
    ("primaryKey", "table.id", "setting"),
    ("primary_key", True, "setting"),
    ("primary_key", 12345, "setting"),
    ("secondary_key", "48213907", "setting"),
    ("IsPrimaryKey", "true", "setting"),
    ("DataModelPrimaryKey", "pk", "setting"),
    ("primary_key", None, "setting"),
    # a word after the key's name, or another credential word: by name
    ("PRIMARY_KEY_2", "id", "credential"),
    ("token_primary_key", "id", "credential"),
]
ENTRY_KINDS = [
    ("TESTS_PASS", "hunter2FAKE", "credential"),
    ("TESTS_PASS", "12", "setting"),
    ("GATES_PASS", "3", "setting"),
    ("BLOCKER_PASS", "3", "setting"),
    ("FIRST_PASS", "hunter2FAKE", "credential"),
    ("DB_PASS", "hunter2FAKE", "credential"),
    ("FIRST_PASS", "true", "setting"),
    ("FINAL_PASS", "3", "setting"),
    ("TESTS_PASS", "12/15", "setting"),
    ("ALL_BLOCKERS_PASS", True, "setting"),
    ("tests_pass", 1234, "setting"),
    ("tests_pass", 123456, "credential"),
    ("TESTS_PASS", None, "setting"),
    ("TESTS_PASS", '["tests/test_a.py::test_x", "tests/test_b.py::test_y"]', "setting"),
    ("n_pass", "hunter2FAKE", "credential"),
    ("TOTAL_PASS", "hunter2FAKE", "credential"),
    ("FAIL_TO_PASS", ["tests/test_a.py::test_x"], "setting"),
    ("FAIL_TO_PASS", "hunter2FAKE", "setting"),
    ("PASS_TO_PASS", "hunter2FAKE", "setting"),
    ("pass_rate", "hunter2FAKE", "setting"),
    ("pass@1", "hunter2FAKE", "setting"),
    ("MAX_TOKENS", "hunter2FAKE", "setting"),
    ("DB_PASS", "12", "credential"),
    # a passphrase's name holding anything but a switch word, a number
    # too, is a passphrase
    ("has_passphrase", "hunter2FAKE", "credential"),
    ("has_passphrase", 12345, "credential"),
    ("DB_PASSPHRASE", "hunter2FAKE", "credential"),
    ("passphrase", 12, "credential"),
    ("PASSPHRASE", "null", "credential"),
    # after a pure count word (n, num, total, count) a pass holds a count
    # of any length; after a test's or an outcome's word up to five digits,
    # as six can be a PIN (the lead's ruling on a review low, 2026-11-13)
    ("n_pass", 123456, "setting"),
    ("TOTAL_PASS", "1234567", "setting"),
    ("num_pass", 100000, "setting"),
    ("count_pass", 250000, "setting"),
    ("n_pass", "12/15", "setting"),
    ("TOTAL_PASS", "120000/150000", "credential"),
    ("TESTS_PASS", 123456, "credential"),
    ("FIRST_PASS", "123456", "credential"),
    ("pass_count", "hunter2FAKE", "setting"),
    # a switch is kept only under a passphrase's flag, a flag word just
    # before the passphrase (the lead's ruling on the resume test,
    # 2026-11-13); under a bare passphrase name it is a passphrase
    ("has_passphrase", False, "setting"),
    ("has_passphrase", "False", "setting"),
    ("requires_passphrase", "true", "setting"),
    ("use_passphrase", 1, "setting"),
    ("no_passphrase", "true", "setting"),
    ("has_passphrase", 12, "credential"),
    ("SSH_KEY_PASSPHRASE", "false", "credential"),
    ("ssh_key_passphrase", True, "credential"),
    ("PASSPHRASE", "no", "credential"),
    ("PASSPHRASE", None, "credential"),
    ("DB_PASSPHRASE", 0, "credential"),
    ("pass_phrase", "on", "credential"),
    # enabled after the passphrase is a last-word qualifier: a setting by name
    ("passphrase_enabled", "true", "setting"),
    # a name that ends in a Cookie or an Authorization header's name after a
    # prefix holds a secret by its value (the lead's rulings, 2026-11-14):
    # anything but prose, a count or a switch is a secret, an opaque cookie
    # value included
    ("HTTP_COOKIE", "theme=dark; sid=FAKEsid0123", "credential"),
    ("http_cookie", "sid%3DFAKEsid0123", "credential"),
    ("HTTP_SET_COOKIE", "sid=FAKEsid0123; Path=/", "credential"),
    ("HTTP_COOKIE", "12", "setting"),
    ("httpCookie", "strict", "setting"),
    ("has_cookie", 1, "setting"),
    ("HTTP_COOKIE", None, "setting"),
    ("HTTP_COOKIE", 482139, "credential"),
    ("HTTP_COOKIE", ["sid=FAKEsid0123"], "credential"),
    ("HTTP_COOKIE", {"sid": "FAKEsid0123"}, "credential"),
    ("HTTP_COOKIE", "8f14e45fceea167a5a36dedd4bea2543", "credential"),
    ("SESSION_COOKIE", "theme=dark; sid=FAKEsid0123", "credential"),
    ("AUTH_COOKIE", "FAKEsess0aZ9bY7Lm2Qx", "credential"),
    ("accept_cookie", "true", "setting"),
    ("HTTP_COOKIE", 12345, "setting"),
    ("HTTP_AUTHORIZATION", "Bearer FAKEtoken0123", "credential"),
    ("HTTP_AUTHORIZATION", "hunter2FAKE", "credential"),
    ("HTTP_AUTHORIZATION", "please sign in first", "credential"),
    ("HTTP_AUTHORIZATION", "Splunk FAKE0123abcd", "credential"),
    ("HTTP_AUTHORIZATION", "", "credential"),
    ("HTTP_AUTHORIZATION", 123456, "credential"),
    ("HTTP_AUTHORIZATION", ["required"], "credential"),
    ("HTTP_AUTHORIZATION", {"scheme": "none"}, "credential"),
    (
        "HTTP_AUTHORIZATION",
        "required",
        "credential",
    ),  # every name ending in authorization holds a credential (the lead, 2026-11-16 10:53Z)
    (
        "REDIRECT_HTTP_AUTHORIZATION",
        "none.",
        "credential",
    ),  # every name ending in authorization holds a credential (the lead, 2026-11-16 10:53Z)
    # A code Authorization name holds a credential whatever it holds
    # (the lead's ruling (b), 2026-11-15 01:3xZ).
    ("redirectHttpAuthorization", "true", "credential"),
    (
        "HTTP_PROXY_AUTHORIZATION",
        "12",
        "credential",
    ),  # every name ending in authorization holds a credential (the lead, 2026-11-16 10:53Z)
    (
        "has_authorization",
        True,
        "setting",
    ),  # a flag's switch (the lead, 2026-11-16 10:53Z: every name that ends in `authorization` is a carrier; a has_/requires_ name's switch, number or null is kept)
    (
        "HTTP_AUTHORIZATION",
        12345,
        "credential",
    ),  # every name ending in authorization holds a credential (the lead, 2026-11-16 10:53Z)
    (
        "HTTP_AUTHORIZATION",
        None,
        "credential",
    ),  # every name ending in authorization holds a credential (the lead, 2026-11-16 10:53Z)
    # the header's own name whole, and a name with bearer and a token word
    # in a row, are credentials whatever they hold
    ("Cookie", "12", "credential"),
    ("Set-Cookie", "strict", "credential"),
    ("set_cookie", None, "credential"),
    ("Authorization", "required", "credential"),
    ("Proxy-Authorization", "none", "credential"),
    ("X_BEARER_TOKEN_COOKIE", "12", "credential"),
    ("BEARER_TOKEN_AUTHORIZATION", "required", "credential"),
    *PRIMARY_KEY_ROWS,
    # the lead's ruling (2026-11-14): tests_to_pass, on_pass, fidelity_pass
    # and norm_pass are settings by name; all, only and two are outcome
    # words before a last pass (five digits), max a count's (any count)
    ("tests_to_pass", "hunter2FAKE", "setting"),
    ("TESTS_TO_PASS", '["tests/test_a.py::test_x"]', "setting"),
    ("on_pass", "hunter2FAKE", "setting"),
    ("fidelity_pass", "hunter2FAKE", "setting"),
    ("norm_pass", 123456, "setting"),
    ("all_pass", True, "setting"),
    ("all_pass", "12", "setting"),
    ("all_pass", "hunter2FAKE", "credential"),
    ("ALL_PASS", 123456, "credential"),
    ("only_pass", "3", "setting"),
    ("onlyPass", "hunter2FAKE", "credential"),
    ("two_pass", True, "setting"),
    ("two_pass", "hunter2FAKE", "credential"),
    ("max_pass", 123456, "setting"),
    ("MAX_PASS", "hunter2FAKE", "credential"),
]
#: The switch shape (is_switch): what the lead ruled a passphrase's name may
#: hold as no secret.
SWITCHES = [
    (None, True), (False, True), (True, True), (0, True), (1, True),
    ("false", True), ("No", True), ("ON", True), ("1", True), ("none", True),
    (2, False), (12345, False), ("12", False), ("null", False), ("hunter2", False),
    ("true.", False), (0.0, False),
]  # fmt: skip


@pytest.mark.parametrize(("value", "switch"), SWITCHES)
def test_a_switch_is_a_switch_word_none_a_bool_or_0_or_1(value, switch):
    """The lead's ruling of 2026-11-13: the switch shape the redactor builds
    from: no other count, no ratio, no null word, no trailing punctuation."""
    assert credential_names.is_switch(value) is switch


@pytest.mark.parametrize(("name", "value", "kind"), ENTRY_KINDS)
def test_an_entrys_kind_depends_on_its_value_only_for_a_pass_outcome(name, value, kind):
    """The lead's option 2 (review of PR #173, 2026-11-13): entry_kind is
    name_kind but for a test's, an outcome's or a count's pass name, which
    holds a test result as a setting and anything else as a password; a
    metric's name (pass_rate) and a name by name (FAIL_TO_PASS) are by name."""
    assert entry_kind(name, value) == kind


#: The agreement rows through the override record's env mapping and the
#: CLI note (the lead's option 2, review of PR #173, 2026-11-13).
OVERRIDE_ENV_ENTRIES = [
    ("TESTS_PASS", "hunter2FAKE", False),
    ("TESTS_PASS", "12", True),
    ("GATES_PASS", "3", True),
    ("BLOCKER_PASS", "3", True),
    ("FIRST_PASS", "hunter2FAKE", False),
    ("DB_PASS", "hunter2FAKE", False),
    ("SSH_PASSPHRASE", "hunter2FAKE", False),
    ("HAS_PASSPHRASE", "false", True),
    ("SSH_KEY_PASSPHRASE", "false", False),
]


@pytest.mark.parametrize(("name", "value", "stored"), OVERRIDE_ENV_ENTRIES)
def test_an_override_env_entry_is_kept_by_its_name_and_value(
    name, value, stored, monkeypatch
):
    """The lead's option 2 (review of PR #173, 2026-11-13): in an
    override's env mapping, a test's or an outcome's pass entry holding a
    test result is stored and compared, and one holding anything else is
    withheld, never stored, and a rotated value resumes, as for any
    credential."""
    from benchflow._utils.config_override import (
        CONFIG_OVERRIDE_ENV,
        ConfigOverrideNeededError,
        recorded_overlay,
    )

    monkeypatch.delenv(CONFIG_OVERRIDE_ENV, raising=False)
    overlay_record = _overlay_record()
    overlay = {"sandbox": {"env": {name: value}}}
    record = overlay_record(overlay)
    if stored:
        assert (record["patch"], record["compare"], record["withheld"]) == (
            overlay,
            overlay,
            [],
        )
        with pytest.raises(ConfigOverrideNeededError):
            recorded_overlay(record, {"sandbox": {"env": {name: "13"}}}, what="regrade")
        return
    assert record["withheld"] == [f"sandbox.env.{name}"]
    assert value not in json.dumps(record)
    rotated = {"sandbox": {"env": {name: "hunter3FAKE"}}}
    assert recorded_overlay(record, rotated, what="regrade") == rotated


@pytest.mark.parametrize(("name", "value", "shown"), OVERRIDE_ENV_ENTRIES)
def test_the_command_note_shows_a_pass_outcome_only_with_a_test_result(
    name, value, shown
):
    """The lead's option 2 (review of PR #173, 2026-11-13): the note of
    ``bench eval run --agent-env NAME=VALUE`` shows the entry as typed when
    the record keeps it, and a reference to the variable otherwise."""
    from benchflow.cli import verbs

    words = ["eval", "run", "--tasks-dir", "t", "--agent", "oracle"]
    note = verbs.note_for([*words, "--agent-env", f"{name}={value}"]) or ""
    assert (value in note) is shown


#: The header value rule through the override record's env mapping, the CLI
#: note and config.json (the lead's follow-up PR, 2026-11-14): kept while the
#: value holds no cookie pairs or is prose, withheld otherwise; a rotated
#: credential resumes. (name, value, kept, the value it rotates to)
HEADER_ENV_ENTRIES = [
    (
        "HTTP_COOKIE",
        "theme=dark; sid=FAKEsid0123",
        False,
        "theme=dark; sid=FAKEsid4567",
    ),
    ("HTTP_COOKIE", "12", True, "13"),
    (
        "HTTP_COOKIE",
        "8f14e45fceea167a5a36dedd4bea2543",
        False,
        "3452aeb4dded63a5a761aeecf54e41f8",
    ),
    (
        "SESSION_COOKIE",
        "8f14e45fceea167a5a36dedd4bea2543",
        False,
        "3452aeb4dded63a5a761aeecf54e41f8",
    ),
    ("has_cookie", "1", True, "0"),
    (
        "HTTP_AUTHORIZATION",
        "required",
        False,
        "none",
    ),  # withheld now (the lead, 2026-11-16 10:53Z)
    ("HTTP_AUTHORIZATION", "Bearer FAKEtoken0123", False, "Bearer FAKEtoken4567"),
    ("REDIRECT_HTTP_AUTHORIZATION", "please sign in first", False, "please sign in"),
    ("Authorization", "required", False, "none"),
    ("Cookie", "strict", False, "lax"),
]


@pytest.mark.parametrize(("name", "value", "kept", "rotated"), HEADER_ENV_ENTRIES)
def test_a_header_env_entry_is_kept_by_its_value(
    name, value, kept, rotated, monkeypatch
):
    """The lead's follow-up PR (2026-11-14): the header value rule is
    entry_kind's, so the override record stores and compares
    HTTP_AUTHORIZATION=required and withholds HTTP_COOKIE holding cookie
    pairs, never storing them, and a rotated cookie resumes. The header's own
    name (Authorization, Cookie) is withheld whatever it holds."""
    from benchflow._utils.config_override import (
        CONFIG_OVERRIDE_ENV,
        ConfigOverrideNeededError,
        recorded_overlay,
    )

    monkeypatch.delenv(CONFIG_OVERRIDE_ENV, raising=False)
    overlay = {"sandbox": {"env": {name: value}}}
    record = _overlay_record()(overlay)
    changed = {"sandbox": {"env": {name: rotated}}}
    if kept:
        assert (record["patch"], record["compare"], record["withheld"]) == (
            overlay,
            overlay,
            [],
        )
        with pytest.raises(ConfigOverrideNeededError):
            recorded_overlay(record, changed, what="regrade")
        return
    assert record["withheld"] == [f"sandbox.env.{name}"]
    assert value not in json.dumps(record)
    assert recorded_overlay(record, changed, what="regrade") == changed


@pytest.mark.parametrize(("name", "value", "kept", "rotated"), HEADER_ENV_ENTRIES)
def test_the_command_note_shows_a_header_entry_by_its_value(name, value, kept, rotated):
    """The lead's follow-up PR (2026-11-14): ``bench eval run --agent-env
    HTTP_COOKIE=<pairs>`` shows a reference to the variable, and
    ``--agent-env HTTP_AUTHORIZATION=required`` the entry as typed."""
    from benchflow.cli import verbs

    words = ["eval", "run", "--tasks-dir", "t", "--agent", "oracle"]
    note = verbs.note_for([*words, "--agent-env", f"{name}={value}"]) or ""
    assert (value in note) is kept
    if not kept:
        assert f'{name}="${name}"' in note


@pytest.mark.parametrize(("name", "value", "kind"), PRIMARY_KEY_ROWS)
def test_a_primary_key_entry_is_kept_by_its_value(name, value, kind, monkeypatch):
    """The lead's ruling (2026-11-14): config.json records a primary or a
    secondary key's entry that holds a schema's key and withholds one that
    holds an access key; the override record stores the one and withholds
    the other, never storing it."""
    from benchflow._utils.config_override import CONFIG_OVERRIDE_ENV
    from benchflow._utils.config_redaction import _should_record_env_entry

    kept = kind == "setting"
    text = value if isinstance(value, str) else json.dumps(value)
    assert _should_record_env_entry(name, text) is kept
    monkeypatch.delenv(CONFIG_OVERRIDE_ENV, raising=False)
    record = _overlay_record()({"sandbox": {"env": {name: text}}})
    if kept:
        assert record["withheld"] == []
        return
    assert record["withheld"] == [f"sandbox.env.{name}"]
    assert text not in json.dumps(record)


#: The redactor's number rule under a key's name: primary_key's exemption
#: (types.py) holds, and the others follow entry_kind.
KEY_NUMBERS = [
    ("primary_key", 12345, False),
    ("secondary_key", 12345, False),
    ("COSMOS_PRIMARY_KEY", 48213907, False),
    ("PRIMARY_KEY_2", 482139, True),
]


@pytest.mark.parametrize(("name", "value", "masked"), KEY_NUMBERS)
def test_the_redactors_number_rule_under_a_keys_name(name, value, masked):
    """The lead's ruling (2026-11-14): a number under a primary or a
    secondary key's name is a schema's; PRIMARY_KEY_2 stays a credential."""
    assert is_secret_number(name, value) is masked


#: The lead's narrowed ruling (2026-11-14): url, name and policy are
#: qualifiers right after a header's word only, and session just before
#: cookie is part of the cookie's name.
HEADER_QUALIFIED_SETTINGS = [
    "authorization_url", "cookie_policy", "session_cookie_name",
    "session_cookie_policy", "cookie_name", "CookieName", "set_cookie_policy",
    "proxy_authorization_url", "SESSION_COOKIE_FILE",
]  # fmt: skip
HEADER_UNQUALIFIED_CREDENTIALS = [
    "TOKEN_URL", "TOKEN_NAME", "SESSION_TOKEN_FILE", "secret_url",
    "_password_url", "HLSStreamingSessionURL", "token_url", "secret_name",
    "key_name", "password_policy", "session_cookie", "cookie_jar_url",
    "session_url",
]  # fmt: skip


@pytest.mark.parametrize("name", HEADER_QUALIFIED_SETTINGS)
def test_a_headers_url_name_or_policy_names_a_setting(name):
    """The lead's narrowed ruling (2026-11-14): a Cookie or an
    Authorization header's endpoint, name or rules is a setting whatever
    it holds."""
    assert name_kind(name) == "setting"
    assert entry_kind(name, "hunter2FAKE0aZ9") == "setting"


@pytest.mark.parametrize("name", HEADER_UNQUALIFIED_CREDENTIALS)
def test_url_name_or_policy_after_another_word_names_no_setting(name):
    """The lead's narrowed ruling (2026-11-14): after any other credential
    word url, name and policy are no qualifiers (TOKEN_URL, secret_url),
    and session before anything but cookie is still a session's word
    (SESSION_TOKEN_FILE)."""
    assert name_kind(name) == "credential"


#: session_cookie itself follows the cookie value rule: prose is a setting's,
#: an opaque value a credential's.
SESSION_COOKIE_ROWS = [
    ("session_cookie", "strict", "setting"),
    ("session_cookie", "12", "setting"),
    ("session_cookie", "8f14e45fceea167a5a36dedd4bea2543", "credential"),
    ("SESSION_COOKIE", "theme=dark; sid=FAKEsid0123", "credential"),
]


@pytest.mark.parametrize(("name", "value", "kind"), SESSION_COOKIE_ROWS)
def test_session_cookie_follows_the_cookie_value_rule(name, value, kind):
    """The lead's narrowed ruling (2026-11-14)."""
    assert entry_kind(name, value) == kind


#: URLs under authorization_url: a setting's name, and the value's own
#: credentials are withheld by its shape (made up).
URL_VALUES = [
    ("https://h.example/oauth/authorize", True),
    (("ht" + "tps://us" + "er:FAKEp" + "ass0aZ9@" + "h.exampl" + "e/oauth"), False),
    (
        (
            "po"
            + "stgresql"
            + "://app:F"
            + "AKEpass0"
            + "aZ9@db.e"
            + "xample:5"
            + "432/app"
        ),
        False,
    ),
    ("https://h.example/x?sig=CiAzFAKEq7W2e9R4t6Y1u3I5o8P0&se=2026-11-15", False),
    (
        "https://b.s3.amazonaws.com/k?X-Amz-Credential=AKIAFAKEFAKEFAKE"
        "&X-Amz-Signature=" + "ab12" * 16,
        False,
    ),
]


@pytest.mark.parametrize(("value", "kept"), URL_VALUES)
def test_a_url_under_authorization_url_is_kept_only_without_credentials(
    value, kept, monkeypatch
):
    """The lead's ruling (2026-11-14): authorization_url is a setting's
    name, and a URL's credentials (a login, a signed query) are still
    withheld by the value's shape, from config.json and the override
    record, under a key and in an env mapping."""
    from benchflow._utils.config_override import CONFIG_OVERRIDE_ENV
    from benchflow._utils.config_redaction import _should_record_env_entry

    assert entry_kind("authorization_url", value) == "setting"
    assert _should_record_env_entry("authorization_url", value) is kept
    monkeypatch.delenv(CONFIG_OVERRIDE_ENV, raising=False)
    for overlay in (
        {"agent": {"authorization_url": value}},
        {"sandbox": {"env": {"authorization_url": value}}},
    ):
        assert (value in json.dumps(_overlay_record()(overlay))) is kept


#: A usage or count companion beside plural tokens wins over a number after
#: it, whatever the number; bare tokens just before a number and a time unit
#: is a count by time window (the lead's rulings, 2026-11-14).
NUMBERED_TOKEN_COUNTS = [
    "input_tokens_2", "output_tokens_v2", "prompt_tokens_12345",
    "completion_tokens_1", "total_tokens_2", "cache_read_tokens_1h",
    "cache_write_tokens_5m", "cache_creation_input_tokens_1h",
    "ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens",
    "cache_creation_tokens_5m", "tokens_1h", "tokens_5m", "TOKENS_1H",
    "total_access_tokens_2",
]  # fmt: skip
#: And what a number does not make a count: a number alone, other letters,
#: a provider's word just before tokens, a singular token, a key.
NUMBERED_TOKEN_CREDENTIALS = [
    "TOKENS_2", "tokens", "tokens_v2", "tokens_2fa", "KEY_2",
    "ACCESS_TOKEN_2", "TOKEN_V2", "access_tokens_2", "access_tokens_1h",
    "token_1h", "tokens_1h_secret",
]  # fmt: skip


@pytest.mark.parametrize("name", NUMBERED_TOKEN_COUNTS)
def test_a_numbered_token_count_names_a_setting(name):
    """The lead's rulings (2026-11-14): input_tokens_2 and
    ephemeral_5m_input_tokens are counts by their usage word, tokens_1h and
    tokens_5m by their time window."""
    assert name_kind(name) == "setting"
    assert entry_kind(name, "1234") == "setting"


@pytest.mark.parametrize("name", NUMBERED_TOKEN_CREDENTIALS)
def test_a_number_alone_makes_no_token_count(name):
    """The lead's rulings (2026-11-14): TOKENS_2 can list secrets as TOKENS
    can, and the provider-word rule holds with a number after."""
    assert name_kind(name) == "credential"


#: GOOGLE_APPLICATION_CREDENTIALS holds a path or a service account's JSON
#: (the lead's ruling, 2026-11-14); values made up.
CREDENTIALS_FILE_ROWS = [
    ("GOOGLE_APPLICATION_CREDENTIALS", "/run/x.json", "setting"),
    (
        "GOOGLE_APPLICATION_CREDENTIALS",
        "~/.config/gcloud/application_default_credentials.json",
        "setting",
    ),
    ("GOOGLE_APPLICATION_CREDENTIALS", None, "setting"),
    ("GOOGLE_APPLICATION_CREDENTIALS", "", "setting"),
    (
        "GOOGLE_APPLICATION_CREDENTIALS",
        '{"type": "service_account", "project_id": "fake-project"}',
        "credential",
    ),
    (
        "GOOGLE_APPLICATION_CREDENTIALS",
        ' \n {"type": "service_account", "project_id": "fake-project"}',
        "credential",
    ),
    (
        "googleApplicationCredentials",
        '{"type": "service_account", "project_id": "fake-project"}',
        "credential",
    ),
    ("GOOGLE_APPLICATION_CREDENTIALS", 'x "private_key": "FAKEkey"', "credential"),
    (
        "GOOGLE_APPLICATION_CREDENTIALS",
        '"{\\"private_key\\": \\"FAKE\\"}"',
        "credential",
    ),
    ("GOOGLE_APPLICATION_CREDENTIALS", {"type": "service_account"}, "credential"),
    ("GOOGLE_APPLICATION_CREDENTIALS", ["/run/x.json"], "credential"),
    ("GOOGLE_APPLICATION_CREDENTIALS_JSON", "/run/x.json", "credential"),
]

_GAC = "GOOGLE_APPLICATION_CREDENTIALS"
_KEY44 = "FAKE" * 10 + "abc="
_OPAQUE = "FAKEtok" + "0123456789abcdef"
#: The review of PR #192, L1: only a path-shaped value is kept under
#: GOOGLE_APPLICATION_CREDENTIALS; values made up.
PATH_VALUES = [
    "/run/x.json",
    "/var/run/secrets/google/key",
    "/Users/me/My Keys/service account.json",
    "/",
    "~/.config/gcloud/application_default_credentials.json",
    "~me/adc.json",
    "./sa.json",
    "../keys/sa.json",
    "C:\\Users\\me\\My Keys\\sa.json",
    "C:/keys/sa.json",
    "\\\\server\\share\\sa.json",
    "$HOME/.config/gcloud/adc.json",
    "${HOME}/adc.json",
    "%USERPROFILE%\\adc.json",
    "sa.json",
    "service-account.json",
    "creds/sa.json",
    "creds\\sa.json",
    ".config/gcloud/adc.json",
    ".adc.json",
    "key.p12",
    "/" + "a" * 4095,
]
NOT_PATH_VALUES = [
    _OPAQUE,
    _KEY44,
    "/" + _KEY44[1:],  # a key that starts with ``/`` and is padded
    "FAKE/abcd+FAKE/abcdFAKE/abcdFAKE/abcdFAKEabc",  # base64 with slashes
    "Bearer " + _OPAQUE,
    "Bearer FAKE/abcd.FAKE",
    "Basic " + _KEY44,
    "sid=" + _OPAQUE + "; theme=dark",
    "sid=" + _OPAQUE + "; path=/",
    "/run/x.json?token=" + _OPAQUE,
    "secrets/sa",  # a relative path with no extension: masked, fail closed
    "sa",
    "my key.json",
    "/run/x.json\n",
    " /run/x.json",
    "/run/x.json\n" + _OPAQUE,
    "https://example.com/sa.json",
    "gs://bucket/sa.json",
    "10.0.0.1",
    "sa.verylongextension",
    "$" + _OPAQUE,
    "~" + _OPAQUE,
    "." + _OPAQUE,
    "true",
    "none",
    "1",
    0,
    1,
    482139,
    1.5,
    True,
    False,
    b"/run/x.json",
    "/" + "a" * 4096,
    "a" * 5000 + ".json",
    "{home}/.config/gcloud/application_default_credentials.json",
]


@pytest.mark.parametrize("value", PATH_VALUES)
@pytest.mark.parametrize("name", [_GAC, "googleApplicationCredentials"])
def test_review_192_l1_a_path_under_the_credentials_file_name_is_kept(name, value):
    """The review of PR #192 (L1): a path, anchored or a file's name with
    an extension, is a setting's value."""
    assert credential_names._path_value(value)
    assert entry_kind(name, value) == "setting"


@pytest.mark.parametrize("value", NOT_PATH_VALUES)
@pytest.mark.parametrize(
    "name", [_GAC, "googleApplicationCredentials", "google_application_credentials"]
)
def test_review_192_l1_any_other_value_is_a_credentials(name, value):
    """The review of PR #192 (L1): an opaque run, ``Bearer <value>``, a
    cookie pair and a 44-character key under GOOGLE_APPLICATION_CREDENTIALS
    were a setting's, kept by the redactors and recorded in config.json.
    Only a path is kept now: every other value is a credential's, a key
    that starts with ``/`` and ends in ``=`` included."""
    assert entry_kind(name, value) == "credential"


def test_review_192_l1_a_slash_led_key_with_no_padding_reads_as_a_path():
    """The residual of L1, pinned: a key-shaped run that starts with ``/``
    and has no ``=`` cannot be told from /var/run/secrets/google/key. The
    name is defined to hold a path or a document, never a bare key."""
    assert entry_kind(_GAC, "/" + "FAKE" * 10 + "abc") == "setting"


def test_review_192_l1_the_path_shape_is_linear():
    """A long value that is no path is answered at once: the length cap
    comes before the shape."""
    import time

    for value in ("a." * 2_000_000, "a/" * 2_000_000, "/" + "a" * 4_000_000):
        start = time.process_time()
        assert entry_kind(_GAC, value) == "credential"
        assert time.process_time() - start < 1.0


@pytest.mark.parametrize(("name", "value", "kind"), CREDENTIALS_FILE_ROWS)
def test_a_credentials_file_name_holds_a_path_or_its_content(name, value, kind):
    """The lead's ruling (2026-11-14): a path under
    GOOGLE_APPLICATION_CREDENTIALS is a setting, the file's JSON inline a
    credential; GOOGLE_APPLICATION_CREDENTIALS_JSON is one by name."""
    assert entry_kind(name, value) == kind


@pytest.mark.parametrize(
    ("value", "kept"),
    [
        ("/run/x.json", True),
        ('{"type": "service_account", "project_id": "fake-project"}', False),
    ],
)
def test_a_credentials_file_entry_is_kept_only_as_a_path(value, kept, monkeypatch):
    """The lead's ruling (2026-11-14): the override record's env mapping
    stores the path and withholds the inline JSON, never storing it."""
    from benchflow._utils.config_override import CONFIG_OVERRIDE_ENV

    monkeypatch.delenv(CONFIG_OVERRIDE_ENV, raising=False)
    name = "GOOGLE_APPLICATION_CREDENTIALS"
    overlay = {"sandbox": {"env": {name: value}}}
    record = _overlay_record()(overlay)
    if kept:
        assert (record["patch"], record["withheld"]) == (overlay, [])
        return
    assert record["withheld"] == [f"sandbox.env.{name}"]
    assert "fake-project" not in json.dumps(record)


#: The redactor's number rule on pass names (the lead's ruling on a review
#: low, 2026-11-13): a count of any length after a pure count word is kept;
#: six digits under a test's or an outcome's pass name, or under a
#: password's, are masked.
PASS_NUMBERS = [
    ("n_pass", 123456, False),
    ("TOTAL_PASS", 1000000, False),
    ("TESTS_PASS", 123456, True),
    ("DB_PASS", 123456, True),
]


@pytest.mark.parametrize(("name", "value", "masked"), PASS_NUMBERS)
def test_the_redactors_number_rule_keeps_a_count_after_a_count_word(
    name, value, masked
):
    """The lead's ruling on a review low (2026-11-13): an integer of 100000
    or more was masked under n_pass and TOTAL_PASS. TESTS_PASS=123456 is
    masked by entry_kind, which the redactor's number rule asks once its
    wiring (the redactor's side) is in; until then it asks name_kind."""
    from benchflow.trajectories import types

    if name == "TESTS_PASS" and not hasattr(types, "entry_kind"):
        pytest.skip("the redactor asks entry_kind on the combined branch")
    assert is_secret_number(name, value) is masked


def _table_names():
    """Every name in this file's tables: the rows of names, and the first
    item of the rows that pair a name with a value or a kind."""
    names = set()
    for table in (SETTINGS, CREDENTIALS, NEITHER, ENTRY_KINDS, CREDENTIALS_FILE_ROWS):
        for row in table:
            item = row
            if hasattr(item, "values"):  # a pytest.param
                item = item.values
            while isinstance(item, tuple | list) and item:
                item = item[0]
            if isinstance(item, str):
                names.add(item)
    return sorted(names)


def test_review_192_l3_the_kept_answer_is_name_kinds():
    """The review of PR #192 (L3): entry_kind keeps name_kind's answers for
    short names. For every name in the tables the kept answer, asked cold
    and then from the cache, is name_kind's own, which is never kept."""
    names = _table_names()
    assert len(names) > 400
    credential_names._NAME_KINDS.clear()
    for name in names:
        expected = name_kind(name)
        assert credential_names._kept_name_kind(name) == expected, name
        assert name in credential_names._NAME_KINDS
        assert credential_names._kept_name_kind(name) == expected, name
    assert len(credential_names._NAME_KINDS) == len(names)
    credential_names._NAME_KINDS.clear()


def test_review_192_l3_entry_kind_is_unchanged_by_the_cache():
    """With the cache cold, warm, and bypassed, every row of the value
    tables gets the same answer from entry_kind."""
    rows = [row for row in CREDENTIALS_FILE_ROWS if len(row) == 3]
    rows += [(_GAC, value, "setting") for value in PATH_VALUES]
    rows += [(_GAC, value, "credential") for value in NOT_PATH_VALUES]
    credential_names._NAME_KINDS.clear()
    cold = [entry_kind(name, value) for name, value, _ in rows]
    warm = [entry_kind(name, value) for name, value, _ in rows]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(credential_names, "_kept_name_kind", name_kind)
        bypassed = [entry_kind(name, value) for name, value, _ in rows]
    assert cold == warm == bypassed == [kind for _, _, kind in rows]
    credential_names._NAME_KINDS.clear()


def test_review_192_l3_the_cache_is_bounded_and_holds_no_long_name():
    """At most _NAME_CACHE_SIZE names are kept, none longer than
    _NAME_CACHE_CHARS characters: a long name is answered and not kept, so
    the cache cannot grow with the text it is asked about."""
    cache = credential_names._NAME_KINDS
    size, chars = credential_names._NAME_CACHE_SIZE, credential_names._NAME_CACHE_CHARS
    assert (size, chars) == (4096, 256)
    cache.clear()
    long_name = "a_key_2_b_token_v2_" * 20_000
    assert entry_kind(long_name, "x") == name_kind(long_name)
    edge = "API_KEY_" + "x" * (chars - 8)
    assert entry_kind(edge, "x") == entry_kind(edge + "x", "x") == "credential"
    assert list(cache) == [edge]
    for index in range(size + 500):
        assert entry_kind(f"NAME_{index}_TOKEN", "x") == "credential"
        assert len(cache) <= size
    assert 0 < len(cache) <= size
    assert all(len(name) <= chars for name in cache)
    cache.clear()


def test_review_192_l3_the_cache_keeps_names_not_values():
    """Only the name is a key: asking about one name with many values keeps
    one entry, and no value is held."""
    cache = credential_names._NAME_KINDS
    cache.clear()
    for index in range(1000):
        entry_kind("TESTS_PASS", f"FAKEvalue{index}")
        entry_kind("GOOGLE_APPLICATION_CREDENTIALS", f"/run/{index}.json")
    assert sorted(cache) == ["GOOGLE_APPLICATION_CREDENTIALS", "TESTS_PASS"]
    assert set(cache.values()) <= {"credential", "setting", None}
    cache.clear()


def test_review_192_l3_name_kind_itself_keeps_nothing():
    """name_kind is asked by tests that swap a rule list (_changed), so it
    answers from the rules every time and leaves the cache alone."""
    credential_names._NAME_KINDS.clear()
    assert name_kind("API_KEY") == "credential"
    assert credential_names._NAME_KINDS == {}


#: The lead's ruling (2026-11-14, after the delta review of PR #192): under
#: an Authorization name that is a name in code, only a code-shaped value
#: is a setting's. Values made up.
_OPAQUE44 = "FAKE" * 11
CODE_NAMES = [
    "has_authorization",
    "requires_authorization",
    "source_authorization",
    "self.source_authorization",
    "sourceAuthorization",
    "hasAuthorization",
    "copy_source_authorization",
    "Has_Authorization",
]
#: Authorization names that are no names in code (an env var in upper
#: case, a header with dashes), and cookie names, which the ruling does not
#: cover. These keep the prose rule.
NOT_CODE_NAMES = [
    "HTTP_AUTHORIZATION",
    "REDIRECT_HTTP_AUTHORIZATION",
    "x-proxy-authorization",
    "X-Forwarded-Authorization",
]
COOKIE_NAMES = ["http_cookie", "HTTP_COOKIE", "sessionCookie"]
CODE_VALUES = [
    "self.flag",
    "config.auth.value",
    "a.b.c.d.e.f",
    "f(x)",
    "get_auth()",
    "x[0]",
    "Optional[str]",
    "Optional[str] = None",
    "dict[str, str]",
    "os.environ['X']",
    'os.getenv("HTTP_AUTHORIZATION")',
    'kwargs.pop("authorization", None)',
    "kwargs.pop(...)",
    "True",
    "False",
    "None",
    "true",
    "false",
    "null",
    "0",
    "42",
    "-1",
    "99999",
    "3.14",
    "bool",
    "str",
    "int",
    "float",
    "dict",
    "list",
    "bool = False",
    "str = None",
    None,
    True,
    False,
    0,
    42,
    1.5,
]
#: No code, and prose: one word of up to 15 letters. Kept under these
#: names by the prose rule, as under every other prefixed header name (the
#: lead's revised ruling, 2026-11-14: prose or code).
PROSE_WORDS = [
    "flag",
    "required",
    "Required",
    "token",
    "abc",
    "x",
    "strict",
    "TRUE",
    "none",
    "yes",
    "Bearer",
    "booleans",
    "boolean",
    "abcdefghijklmno",
]
NOT_CODE_VALUES = [
    "a1b2c3d4",
    "abcdefghijklmnop",
    "target_authorization",
    "authorization_source",
    "abc123",
    _OPAQUE,
    _OPAQUE44,
    _KEY44,
    "Bearer " + _OPAQUE,
    "please sign in first",
    '"quoted words"',
    '"' + _OPAQUE + '"',
    "'" + _OPAQUE + "'",
    "b'" + _OPAQUE + "'",
    'f"{x}"',
    "lambda: x",
    "not x",
    "a or b",
    "[1, 2]",
    "{}",
    "()",
    "482139",
    "123456",
    "1.2.3",
    "",
    " self.flag",
    "self.flag ",
    "self.flag\n",
    "self." + "a" * 25,
    "a.b.c.d.e.f.g",
    'f("' + _OPAQUE44 + '")',
    "f('" + _OPAQUE44 + "')",
    "f(" + _OPAQUE44 + ")",
    "x[" + _OPAQUE44 + "]",
    'f("has space")',
    "f(x",
    "f(x))  # " + _OPAQUE,
    "Optional[str] = " + _OPAQUE,
    "strict_mode",
    "self.flag = " + _OPAQUE,
    "f(" + "x, " * 80 + "x)",
    482139,
    ["self.flag"],
    {"a": 1},
    b"self.flag",
]
#: The two shapes that read as code though they could hold a value: pinned.
CODE_RESIDUALS = ["abc.def123", 'f("abc123")']


# The lead, 2026-11-16 10:53Z (the rc twin of rel's F4): every name that ends
# in `authorization` is a carrier. Under a has_ or requires_ Authorization
# name only an object's boolean or null is no credential; every value as text
# is masked, as under every code Authorization name (F2b, 2026-11-16 21:47Z).
def _flag_entry(name, value) -> bool:
    words = name_words(name)
    return (
        len(words) >= 2
        and words[0] in ("has", "requires")
        and words[-1] == "authorization"
        and (value is None or isinstance(value, bool))
    )


def _ends_in_authorization(name: str) -> bool:
    return name.lower().replace("-", "_").endswith("authorization")


@pytest.mark.parametrize("value", CODE_VALUES)
@pytest.mark.parametrize("name", CODE_NAMES)
def test_delta_192_code_under_a_code_authorization_name_is_a_credential(name, value):
    """The lead's ruling (2026-11-14): a dotted name, a call, a subscript,
    a generic, a literal and a builtin type's name are code. Under
    has_authorization and its like they were a setting's; under the lead's ruling (b), 2026-11-15 01:3xZ,
    every value there is a credential's, code included (finer display of
    code returns in a later release)."""
    assert credential_names.is_code_value(value)
    assert entry_kind(name, value) == (
        "setting" if _flag_entry(name, value) else "credential"
    )


@pytest.mark.parametrize("value", NOT_CODE_VALUES)
@pytest.mark.parametrize("name", CODE_NAMES)
def test_delta_192_anything_else_under_a_code_authorization_name_is_a_credentials(
    name, value
):
    """The lead's ruling (2026-11-14): the delta review of PR #192 found an
    identifier-shaped opaque value kept as code under these names. Now
    everything that is neither code nor prose is a credential's: an
    identifier with a digit or an underscore (a1b2c3d4), 16 letters, an
    opaque run, a quoted string, a bytes or f-string, six digits. A call
    around a 44-character quoted value is no code."""
    assert not credential_names.is_code_value(value)
    assert entry_kind(name, value) == (
        "setting" if _flag_entry(name, value) else "credential"
    )


@pytest.mark.parametrize("value", PROSE_WORDS)
@pytest.mark.parametrize("name", CODE_NAMES)
def test_delta_192_a_word_under_a_code_authorization_name_is_a_credential(name, value):
    """A word of up to 15 letters is no code. The prose rule kept it, and
    ``my_authorization: required`` with it (the review of PR #148); under
    the lead's ruling (b), 2026-11-15 01:3xZ, every value under a code Authorization name is a
    credential's, prose included."""
    assert not credential_names.is_code_value(value)
    assert entry_kind(name, value) == (
        "setting" if _flag_entry(name, value) else "credential"
    )


def test_delta_192_the_prose_edge_is_pinned_in_both_directions():
    """The lead's two rows: an identifier with digits is masked, a
    capitalised word was kept; under the lead's ruling (b), 2026-11-15 01:3xZ both are a credential's."""
    assert entry_kind("has_authorization", "a1b2c3d4") == "credential"
    assert entry_kind("my_authorization", "Required") == "credential"
    assert entry_kind("my_authorization", "required") == "credential"


@pytest.mark.parametrize("value", CODE_RESIDUALS)
def test_delta_192_the_two_code_residuals_are_credentials(value):
    """Accepted by the lead (2026-11-14) as residuals: a dotted value of
    short parts and a call around a short quoted name read as code. Under
    the lead's ruling (b), 2026-11-15 01:3xZ they are a credential's, as every value there is."""
    assert credential_names.is_code_value(value)
    assert entry_kind("has_authorization", value) == "credential"


@pytest.mark.parametrize("name", [*NOT_CODE_NAMES, *COOKIE_NAMES])
def test_delta_192_other_prefixed_header_names_keep_the_prose_rule(name):
    """The lead's ruling (2026-11-14): the prose rule for other prefixed
    header names is unchanged. A word is kept, code is not prose."""
    if name in NOT_CODE_NAMES:
        assert not credential_names._code_name(name)
    # Every name that ends in authorization holds a credential, prose
    # included (the lead, 2026-11-16 10:53Z); a cookie name keeps the rule.
    prose = (
        "credential"
        if name.lower().replace("-", "_").endswith("authorization")
        else "setting"
    )
    assert entry_kind(name, "required") == prose
    assert entry_kind(name, "true") == prose
    assert entry_kind(name, "self.flag") == "credential"
    assert entry_kind(name, "f(x)") == "credential"
    assert entry_kind(name, _OPAQUE) == "credential"


def test_delta_192_the_code_shape_is_built_from_its_own_lists():
    """One list of builtin type names, in the shape and nowhere else."""
    assert credential_names._CODE_TYPES == (
        "bool",
        "str",
        "int",
        "float",
        "dict",
        "list",
    )
    for name in credential_names._CODE_TYPES:
        assert credential_names.is_code_value(name)
    assert not credential_names.is_code_value("bytes")


def test_delta_192_the_code_shape_does_not_blow_up():
    """Long values that are no code are answered at once (the length cap),
    and the shape itself stays under the shared ceiling on values at the
    cap."""
    from tests._timing import within_ceiling

    cases = [
        "a." * 2_000_000,
        "f(" + "x," * 2_000_000,
        "f(" + "(" * 127 + ")" * 126,
        "f(" + "x" * 250 + ")",
        "a" * 24 + "." + "(" * 200,
        "f(" + '"a' * 120,
    ]
    for value in cases:
        kind = within_ceiling(entry_kind, "has_authorization", value)
        assert kind in ("credential", "setting")


#: The lead's addition (2026-11-14): under a code Authorization name, a bare
#: identifier made of the name's own words, all of them or a trailing run,
#: is code. (name, value)
OWN_WORD_VALUES = [
    ("copy_source_authorization", "copy_source_authorization"),
    ("copy_source_authorization", "source_authorization"),
    ("copy_source_authorization", "sourceAuthorization"),
    ("copy_source_authorization", "SOURCE_AUTHORIZATION"),
    ("copy_source_authorization", "_source_authorization"),
    ("copySourceAuthorization", "sourceAuthorization"),
    ("copySourceAuthorization", "copy_source_authorization"),
    ("self.source_authorization", "source_authorization"),
    ("has_authorization", "has_authorization"),
    ("source_authorization", "source_authorization"),
]
NOT_OWN_WORD_VALUES = [
    ("copy_source_authorization", "authorization_source"),  # another order
    ("copy_source_authorization", "copy_authorization"),  # a gap
    ("copy_source_authorization", "copy_source"),  # a leading run
    ("copy_source_authorization", "my_copy_source_authorization"),  # longer
    ("source_authorization", "copy_source_authorization"),  # longer
    ("copy_source_authorization", "source_authorization2"),
    ("copy_source_authorization", "source_authorization_" + _OPAQUE),
    ("copy_source_authorization", "source authorization"),
    ("copy_source_authorization", "source-authorization"),
    ("copy_source_authorization", "source_authorization "),
    ("copy_source_authorization", '"source_authorization"'),
    ("has_authorization", "source_authorization"),
    ("has_authorization", _OPAQUE),
]


@pytest.mark.parametrize(("name", "value"), OWN_WORD_VALUES)
def test_delta_192_an_identifier_of_the_names_own_words_is_code(name, value):
    """The lead's addition (2026-11-14): `copy_source_authorization =
    copy_source_authorization` and `'copy_source_authorization':
    source_authorization` are code, an assignment or a mapping of a
    variable of the same name. Only with the name: alone, the value is a
    bare identifier and no code."""
    assert credential_names.is_code_value(value, name)
    assert not credential_names.is_code_value(value)
    # Code all the same, and a credential's under the lead's ruling (b), 2026-11-15 01:3xZ.
    assert entry_kind(name, value) == "credential"


@pytest.mark.parametrize(("name", "value"), NOT_OWN_WORD_VALUES)
def test_delta_192_any_other_identifier_is_a_credentials(name, value):
    """The lead's addition (2026-11-14): the words in another order
    (authorization_source), with a gap, a leading run, more words than the
    name has, a digit or anything after them are not the name's own words:
    a credential's."""
    assert not credential_names.is_code_value(value, name)
    assert entry_kind(name, value) == "credential"


def test_delta_192_a_word_is_prose_not_the_names_own_words():
    """`copy_source_authorization = flag`: flag is none of the name's words
    and no code, and is kept all the same, as prose (the lead's revised
    ruling, prose or code). A word that is the name's last word is both."""
    assert not credential_names.is_code_value("flag", "copy_source_authorization")
    assert credential_names.is_code_value("authorization", "has_authorization")
    # Both are a credential's under the lead's ruling (b), 2026-11-15 01:3xZ.
    assert entry_kind("copy_source_authorization", "flag") == "credential"
    assert entry_kind("has_authorization", "authorization") == "credential"


@pytest.mark.parametrize("name", NOT_CODE_NAMES)
def test_delta_192_the_own_words_rule_is_for_code_names_only(name):
    """An env var's or a header's own name as its value is no prose and is
    a credential's, as before."""
    assert entry_kind(name, name) == "credential"
