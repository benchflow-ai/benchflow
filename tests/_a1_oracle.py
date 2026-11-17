"""A.1 as the lead ruled it, restated clause by clause from the rulings' text in F2b-STATE.md (2026-11-14,
14:1xZ-15:0xZ and 20:05Z), not from the redactor or the classifier. Under a code Authorization name
(has_authorization, copy_source_authorization, ...) a value may stay visible only if one clause below allows it;
anything else must be masked. allowed(value, name) returns the clause's name, or None (must be masked). The value is
the raw text after the name and its separator, before any JSON or HTML escaping the layer adds.

The ruling text, verbatim (the ledger lines this file restates):
- 14:1xZ: "A.1 residual 3 narrows to CODE-SHAPED values only (a classifier value rule: dotted, call/subscript/generic,
  literal, builtin type) -> everything else masked incl. a bare identifier (accepted over-mask)."
- 14:2xZ: "the code-shape rule FINAL: quoted/b'/f'/lambda/[]/{} not code; numbers <=5 digits or decimals code; for the
  A.1 Authorization names ONLY code shape only (bare words masked; lower true/false/null literals); the prose rule
  unchanged elsewhere; residuals abc.def123 + f("abc123") accepted; + a test: f("<44-char key>") masked."
- 14:3xZ (REVISED item 3): "A.1 names = code-shaped OR prose (the one-word <=15-letter rule); else masked; the short
  all-letters-word residual named."
- 14:5xZ: "APPROVED the narrow addition (a bare identifier equal to the name or a trailing run of its words =
  code-shaped; inside _CODE_VALUE; rows)"
- 15:0xZ: "keep 14:3xZ (= flag kept as prose; drop the masked row; add = flag2x9 masked)."
- 20:05Z (settings 1-4 below confirmed; decimals shown; this clause added): "Inside a code shape (call arguments,
  subscripts, dotted parts, defaults), every token of 24 or more characters, quoted or not, is masked unless it is
  name-like. A token is name-like when it splits on `_`, `.`, `-` and lower-to-upper camel boundaries into at most 8
  parts, each part letters only, 2-15 long, optionally ending in at most 2 digits, and the token is not all hex and
  not base64-shaped (mixed case plus digits with no separators). Digit runs inside brackets follow the at-most-5-digit
  number rule."
  The lead's test rows: shown scaled_dot_product_attention, torch.nn.functional.softmax, getUserAccountSettingsV2;
  masked a Stripe-shaped live key (sk_live_ and 24 characters), a 32-hex token, a 40-character base64 token, AbCdEfGhIjKlMnOpQrStUvWx,
  x[123456789012], abc.<40-char token>, f(<44-char unquoted token>).
- 20:2xZ (setting 6 kept; setting 1 wins over "quoted or not", that over-mask accepted; this clause added to the
  name-like test): "Each part must also be all lowercase, Titlecase, or an all-caps acronym of at most 5 letters, and
  every part of 4 or more letters must contain a vowel (a, e, i, o, u, y)."
- 20:34Z (final; the 7-per-4M random survivors accepted as a documented residual, no entropy test): "Apply A, not B",
  where A is "a 6-15-capital part counts when every part is all caps (a constant)"; and "The vowel test gets an
  exact-word list of consonant-only words common in code. Each part is matched case-insensitively, and a listed word
  passes the vowel test as if it had a vowel." The list starts as: "http, https, httpx, grpc, mtls, html, xhtml, smtp,
  sftp, ftps, rtmp, rtsp, dhcp, ntlm, pkcs, lstm, tmpfs, pgsql, nntp" (R528 may extend it with exact words from the
  corpus's most frequent consonant-only parts; every added word is checked, and patterns or shapes are refused).
- 20:4xZ, R528's extension (via F2b; 60 exact words from consonant_parts.tsv sha 04232a96..., in its rank order;
  R528's rule: "a recognised code word or acronym, AND it either un-masks something in the file or ranks in the file's
  top 75"), verbatim: "llms, tqdm, tmpl, stmt, kwds, sqrt, spdx, ctrl, hdrs, vllm, gdch, crlf, jwks, zstd, rpcs, fcntl, nbsp, rcvd, cdll, cmds, xmlns, llvm, sdks, msvcrt, dcgm, xmlrpc, strs, fqdn, msgs, jwts, stmts, lstsq, pkgs, ssrf, mpfr, ttft, rlhf, tcsh, httpd, wsdl, wgsl, mbcs, hkdf, ttls, dtls, snmp, dsls, hdfs, tsql, pbkdf, msft, ntdll, jdbc, gbps, srtp, vpcs, dtmf, mssql, plpgsql, dbrx"

Settings (1-4 confirmed by the lead at 20:05Z; 6 kept at 20:2xZ):
1. QUOTED_IN_CODE_MAX = 23: the longest quoted string inside a call/subscript/generic (f("abc123") accepted;
   f("<44-char key>") masked; 23 from the lead's 17:5xZ rule for #227, "key-shaped (>= 24 chars or opaque)").
2. BUILTIN_TYPES: Python's builtin type names ("builtin type" names no list).
3. LITERAL_DEFAULT = True: a code shape (builtin, dotted, call, own words) may carry ` = <literal or number>`
   (Optional[str] = None); prose may not (flag = None is masked).
4. NOT_CODE_IN_ARGS = False: 14:2xZ's not-code list is read as value shapes, so f(b"x") and f(lambda: 1) are calls.
5. TOKEN_CAP = True, TOKEN_MIN = 24: the 20:05Z clause, one switch (F2b to R528, 20:05Z: "default ON").
6. BASE64_PART_END_DIGITS_OK = True: read literally, "base64-shaped (mixed case plus digits with no separators)" masks
   the lead's own shown row getUserAccountSettingsV2 (mixed case, the digit 2, no separators). So a part's allowed
   trailing digits (at most 2) don't count as "digits" here. With lower-to-upper boundaries only, a separator-free
   token can hold digits only at its end, so under this reading the base64 test masks nothing the part rule doesn't.
   Every masked test row is masked by the part rule alone. Kept by the lead at 20:2xZ.
7. PART_CASE_VOWEL = True: the 20:2xZ clause, its own switch (TOKEN_CAP must be on for it to apply).
8. CAPS_CONSTANT = True: 20:34Z A, its own switch (PART_CASE_VOWEL must be on for it to apply).
9. VOWEL_FREE_WORDS: 20:34Z's exact-word list and R528's 60 (20:4xZ), each quoted word for word.
10. TRAILING_CLOSERS = True: the 21:21Z clause, its own switch.

Readings that are not settings:
- Prose is "the one-word <=15-letter rule", the rule 14:3xZ cites. Its text in the rc (types.py, _PROSE_VALUE) is "a
  plain word (letters only, fifteen or fewer ...) ..., perhaps closing a sentence or a bracket". So a word may be
  followed by up to three of ) ] . ;
- A number may carry one leading sign. .5, 5., 1e5, 0x1F and 1_000 are not "numbers <=5 digits or decimals". A
  decimal at the top level shows at any length (20:05Z "decimals shown").
- 14:5xZ's words: a name splits at _ . - and at lower-to-upper and digit-to-upper case changes, compared without case.
- A quote closes only after the same run of backslashes it opened with, so f(\"abc123\") pairs as written. Where
  the quotes can't be paired, or the brackets ( [ { don't balance, nothing is positively code-shaped: masked.
- 20:05Z tokens:
  - A token is a run of characters other than whitespace, ( ) [ ] { } , : = * < > | & ^ ! ; and quotes.
  - The backslash run that opens or closes a quote belongs to the quote.
  - A dotted chain is one token, because the clause splits a token on `.`. So a 24+ chain with a one-letter part
    (self.x.attention_dropout_rate) is masked.
  - The clause covers every token of a dotted value or call/subscript/generic: the callee and attribute names (a
    key-shaped callee would otherwise show), the arguments and subscripts, quoted contents, and a default. It
    doesn't cover own words, since every one of their words comes from the name.
  - With setting 1, every quoted string over 23 characters inside a code shape is masked whether name-like or not.
  - Empty parts (__init__, _private_name) don't count. A part's 2-15 includes its trailing digits, as V2 needs.
  - "Inside brackets" means at bracket depth 1 or more, quoted or not. A run of 6+ digits there is masked unless it
    lies in a decimal (digits.digits), the number rule with decimals shown. A decimal of 24+ characters inside a
    code shape is a 24+ token that is not name-like, so it is masked.
- 20:2xZ part case and vowels:
  - A part is what the 20:05Z split leaves: tokens split at _ . - and at lower-to-upper boundaries only. So an acronym
    run stays joined to the word after it (HTTPResponse, XMLHttp, JSONWeb), and that part is none of the three cases,
    so it fails. An all-caps part comes only from separators (HTTP_SERVER, parse_JSON_web) or at the end of a token
    (getV2, parseJSON).
  - The case and the vowel test read the part's letters, without its 0-2 trailing digits (V2 is the acronym V).
    "4 or more letters" counts letters only (utf8 has 3).
  - All lowercase is [a-z]+, Titlecase is [A-Z][a-z]+, and an all-caps acronym is [A-Z]{1,5}.
  - The vowel test covers every part, acronyms included (HTTP and HTML fail, JSON and UUID pass). It ignores case,
    and y counts wherever it is written (rhythm passes).
  - A digit-only part has no letters, so it already fails the 20:05Z part rule (layer_1_output).
  - 20:34Z A:
    - "Every part" means every part of the token, its letters without trailing digits. A dotted chain is one token, so
      a constant read through a lowercase name (obj.CLIENT_LOGGING_SUPPORTED, google.rpc.Code.UNIMPLEMENTED) is not
      all caps, and its 6+-capital part fails.
    - A doesn't touch the vowel test: CLIENT_CERTIFICATE passes, while BCDFGH_SETTINGS fails on its vowel-less part.
  - 20:34Z list:
    - The listed word is compared with the part's letters, lowercased and without the part's 0-2 trailing digits, so
      http2, HTTP, Https and pkcs12 match.
    - A listed word still has to pass the case test (hTTp fails) and the 20:05Z part rule.
    - Exact words only: httpss and xhttp are not listed.
  - The clause applies only where the 20:05Z clause does: to tokens of 24 or more characters inside a code shape.
    XMLHttpRequest (14), getHTTPResponseCode (19), HTTPServerV2 (12) and parseJSONWebTokenClaims (23) are shorter.
- 21:21Z (the lead, via F2b): "An unmatched trailing `)`, `]` or `}` closes the surrounding code, so the value is what
  precedes it." (So self.x] shows as a code name, and a key followed by ) stays masked by the name shapes.) And:
  test-result counts like 12/15 under a credential name are masked, no exemption: no clause here allows them.
- 21:21Z trailing closers (setting 10):
  - "Unmatched" is read quote-aware, as the call scanner reads quotes. A closer inside a quoted string is no closer,
    and a closer that meets no opener, or the wrong one, is unmatched.
  - Only the trailing run of unmatched closers is stripped: f(x)) is f(x), and Gb9z()}) is Gb9z().
  - A matched closer ends the run: A[] stays A[], and f({a: 1}}) stays itself, unbalanced and masked.
  - A value that is nothing but closers is empty, and an empty value shows nothing (clause "empty").
"""
import re

QUOTED_IN_CODE_MAX = 23
BUILTIN_TYPES = ("bool", "str", "int", "float", "complex", "dict", "list", "tuple", "set", "frozenset", "bytes",
                 "bytearray", "object", "type", "range", "memoryview")
LITERAL_DEFAULT = True
NOT_CODE_IN_ARGS = False
TOKEN_CAP = True
TOKEN_MIN = 24
BASE64_PART_END_DIGITS_OK = True
PART_CASE_VOWEL = True
CAPS_CONSTANT = True
TRAILING_CLOSERS = True
VOWEL_FREE_WORDS = ("http", "https", "httpx", "grpc", "mtls", "html", "xhtml", "smtp", "sftp", "ftps", "rtmp", "rtsp",
                    "dhcp", "ntlm", "pkcs", "lstm", "tmpfs", "pgsql", "nntp",
                    # R528's 60 (20:4xZ)
                    "llms", "tqdm", "tmpl", "stmt", "kwds", "sqrt", "spdx", "ctrl", "hdrs", "vllm", "gdch", "crlf",
                    "jwks", "zstd", "rpcs", "fcntl", "nbsp", "rcvd", "cdll", "cmds", "xmlns", "llvm", "sdks",
                    "msvcrt", "dcgm", "xmlrpc", "strs", "fqdn", "msgs", "jwts", "stmts", "lstsq", "pkgs", "ssrf",
                    "mpfr", "ttft", "rlhf", "tcsh", "httpd", "wsdl", "wgsl", "mbcs", "hkdf", "ttls", "dtls", "snmp",
                    "dsls", "hdfs", "tsql", "pbkdf", "msft", "ntdll", "jdbc", "gbps", "srtp", "vpcs", "dtmf",
                    "mssql", "plpgsql", "dbrx")

IDENT = r"[A-Za-z_][A-Za-z0-9_]*"

# 14:3xZ "prose (the one-word <=15-letter rule)" (15:0xZ: "= flag kept as prose"; "= flag2x9 masked").
PROSE = re.compile(r"[A-Za-z]{1,15}[)\].;]{0,3}")
# 14:2xZ "lower true/false/null literals" (and the literals True/False/None, 14:1xZ "literal").
LITERAL = re.compile(r"True|False|None|true|false|null")
# 14:2xZ "numbers <=5 digits or decimals code" (six digits or more are no code).
NUMBER = re.compile(r"[-+]?(?:[0-9]{1,5}|[0-9]+\.[0-9]+)")
# 14:1xZ "dotted" (14:2xZ residual "abc.def123 accepted").
DOTTED = re.compile(rf"{IDENT}(?:\.{IDENT})+")
# 14:1xZ "builtin type".
BUILTIN = re.compile("|".join(BUILTIN_TYPES))
# 14:2xZ "quoted/b'/f'/lambda/[]/{} not code" and 14:3xZ "code-shaped OR prose": a quoted string stays only when it
# holds prose ("quoted strings masked unless prose"), at any escape depth, the same run of backslashes on both quotes.
QUOTED_PROSE = re.compile(r"(\\*)([\"'])[A-Za-z]{1,15}\1\2[)\].;]{0,3}")
NOT_CODE_START = re.compile(r"(?:[rRbBuUfF]{1,3}[\"']|lambda\b|\[|\{)")
DEFAULT = re.compile(r"(.+?)[ \t]*=[ \t]*(True|False|None|true|false|null|[-+]?(?:[0-9]{1,5}|[0-9]+\.[0-9]+))")

# 20:05Z "every token of 24 or more characters, quoted or not, is masked unless it is name-like".
TOKEN_SPLIT = re.compile(r"[\s()\[\]{},:=*<>|&^!;\"']+")
# 20:05Z "each part letters only, 2-15 long, optionally ending in at most 2 digits".
PART = re.compile(r"(?=.{2,15}\Z)[A-Za-z]+[0-9]{0,2}")
# 20:05Z "not all hex".
ALL_HEX = re.compile(r"[0-9a-fA-F]+")
# 20:2xZ "all lowercase, Titlecase, or an all-caps acronym of at most 5 letters" and "a vowel (a, e, i, o, u, y)".
PART_CASE = re.compile(r"[a-z]+|[A-Z][a-z]+|[A-Z]{1,5}")
VOWEL = re.compile(r"[aeiouyAEIOUY]")
# 20:05Z "Digit runs inside brackets follow the at-most-5-digit number rule" (decimals shown, 20:05Z (c)).
DIGIT_RUN = re.compile(r"[0-9]{6,}")
DECIMAL = re.compile(r"(?<![A-Za-z0-9_.])[0-9]+\.[0-9]+(?![A-Za-z0-9_.])")


def _base64_shaped(tok: str) -> bool:
    """20:05Z "base64-shaped (mixed case plus digits with no separators)" (setting 6 on a part's trailing digits)."""
    if re.search(r"[_.\-]", tok) or not (re.search(r"[a-z]", tok) and re.search(r"[A-Z]", tok)):
        return False
    body = re.sub(r"[0-9]{1,2}\Z", "", tok) if BASE64_PART_END_DIGITS_OK else tok
    return re.search(r"[0-9]", body) is not None


def name_like(tok: str) -> bool:
    """20:05Z "splits on `_`, `.`, `-` and lower-to-upper camel boundaries into at most 8 parts, each part letters
    only, 2-15 long, optionally ending in at most 2 digits, and the token is not all hex and not base64-shaped"."""
    if ALL_HEX.fullmatch(tok) or _base64_shaped(tok):
        return False
    parts = [p for p in re.split(r"[_.\-]+", re.sub(r"(?<=[a-z])(?=[A-Z])", "_", tok)) if p]
    if not (0 < len(parts) <= 8 and all(PART.fullmatch(p) for p in parts)):
        return False
    # 20:34Z A "a 6-15-capital part counts when every part is all caps (a constant)".
    constant = CAPS_CONSTANT and all(re.fullmatch(r"[A-Z]+", p.rstrip("0123456789")) for p in parts)
    return all(_part_case_vowel_ok(p, constant) for p in parts)


def _part_case_vowel_ok(part: str, constant: bool = False) -> bool:
    """20:2xZ "Each part must also be all lowercase, Titlecase, or an all-caps acronym of at most 5 letters, and every
    part of 4 or more letters must contain a vowel (a, e, i, o, u, y)"; 20:34Z A and the exact-word list."""
    if not PART_CASE_VOWEL:
        return True
    letters = part.rstrip("0123456789")
    case_ok = PART_CASE.fullmatch(letters) or (constant and re.fullmatch(r"[A-Z]{6,15}", letters))
    vowel_ok = len(letters) < 4 or VOWEL.search(letters) is not None or letters.lower() in VOWEL_FREE_WORDS
    return bool(case_ok) and vowel_ok


def _digit_runs_ok(text: str) -> bool:
    spans = [m.span() for m in DECIMAL.finditer(text)]
    return all(any(a <= d.start() and d.end() <= b for a, b in spans) for d in DIGIT_RUN.finditer(text))


def _tokens_ok(texts, in_brackets: bool) -> bool:
    if not TOKEN_CAP:
        return True
    for s in texts:
        if in_brackets and not _digit_runs_ok(s):
            return False
        for t in TOKEN_SPLIT.split(s):
            if len(t) >= TOKEN_MIN and not name_like(t):
                return False
    return True


def _split_default(v: str):
    m = DEFAULT.fullmatch(v) if LITERAL_DEFAULT else None
    return (m.group(1), m.group(2)) if m else (v, None)


def _scan_call(v: str):
    """14:1xZ "call/subscript/generic": a callee (a name or a dotted name), then one or more bracket groups to the
    value's end (attribute access between them: f(x)[0], a.b(c).d(e)). The brackets must balance, and every quoted
    string inside must be short (14:2xZ: f("abc123") accepted, f("<44-char key>") masked). Returns None when v is not
    call-shaped, else (outside, inside, quoted): the callee and attribute names, the unquoted text inside brackets,
    and each quoted string's content (without its escape runs)."""
    m = re.match(rf"{IDENT}(?:\.{IDENT})*", v)
    if not m or m.end() == len(v) or v[m.end()] not in "([":
        return None
    pair = {"(": ")", "[": "]", "{": "}"}
    outside, inside, quoted = [v[:m.end()]], [], []
    i, groups = m.end(), 0
    while i < len(v):
        c = v[i]
        if c in "([":
            stack, q, qd, qs, bs, j, seg = [], None, 0, 0, 0, i, i
            while j < len(v):
                ch = v[j]
                if q:
                    if ch == q and bs == qd:   # a quote closes after the backslash run it opened with
                        body = v[qs + 1 : j - qd]
                        if len(body) > QUOTED_IN_CODE_MAX:
                            return None
                        quoted.append(body)
                        seg, q = j + 1, None
                elif ch in "\"'":
                    if NOT_CODE_IN_ARGS and re.search(r"[rRbBuUfF]{1,3}$", v[:j - bs]):
                        return None
                    inside.append(v[seg : j - bs])
                    q, qd, qs = ch, bs, j
                elif ch in "([{":
                    stack.append(ch)
                elif ch in ")]}":
                    if not stack or pair[stack.pop()] != ch:
                        return None
                    if not stack:
                        break
                bs = bs + 1 if ch == "\\" else 0
                j += 1
            if j >= len(v) or q:
                return None
            inside.append(v[seg : j + 1])
            if NOT_CODE_IN_ARGS and re.search(r"\blambda\b|[(,]\s*[\[{]", v[i:j + 1]):
                return None
            groups += 1
            i = j + 1
        elif c == "." and (a := re.match(rf"(?:\.{IDENT})+", v[i:])):
            outside.append(a.group())
            i += a.end()
        else:
            return None
    return (outside, inside, quoted) if groups else None


def strip_unmatched_closers(v: str) -> str:
    """21:21Z "An unmatched trailing `)`, `]` or `}` closes the surrounding code, so the value is what precedes it"."""
    pair = {")": "(", "]": "[", "}": "{"}
    stack, q, qd, bs, unmatched = [], None, 0, 0, []
    for j, ch in enumerate(v):
        if q:
            if ch == q and bs == qd:   # a quote closes after the backslash run it opened with
                q = None
        elif ch in "\"'":
            q, qd = ch, bs
        elif ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if stack and stack[-1] == pair[ch]:
                stack.pop()
            else:
                unmatched.append(j)
        bs = bs + 1 if ch == "\\" else 0
    end = len(v)
    while end and end - 1 in unmatched:
        end -= 1
    return v[:end]


def words(s: str) -> list[str]:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return [w.lower() for w in re.split(r"[_.\-]+", s) if w]


def allowed(value: str, name: str):
    v = strip_unmatched_closers(value) if TRAILING_CLOSERS else value
    if not v and value:
        return "empty"
    if LITERAL.fullmatch(v):
        return "literal"
    if NUMBER.fullmatch(v):
        return "number"
    core, default = _split_default(v)
    dflt = [default] if default else []
    if BUILTIN.fullmatch(core):
        return "builtin" if _tokens_ok(dflt, False) else None
    if PROSE.fullmatch(v):
        return "prose"
    if QUOTED_PROSE.fullmatch(v):
        return "quoted-prose"
    if NOT_CODE_START.match(v):
        return None
    if DOTTED.fullmatch(core):
        return "dotted" if _tokens_ok([core] + dflt, False) else None
    sc = _scan_call(core)
    if sc:
        outside, inside, quoted = sc
        ok = _tokens_ok(outside + dflt, False) and _tokens_ok(inside + quoted, True)
        return "call" if ok else None
    # 14:5xZ "a bare identifier equal to the name or a trailing run of its words = code-shaped".
    if re.fullmatch(IDENT, core):
        w, own = words(core), words(name)
        if w and len(w) <= len(own) and own[-len(w):] == w and _tokens_ok(dflt, False):
            return "own-words"
    # 14:1xZ "everything else masked incl. a bare identifier (accepted over-mask)".
    return None
