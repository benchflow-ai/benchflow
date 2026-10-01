#!/usr/bin/env python3
# ruff: noqa: UP017 - this tool runs on stock Python 3.10 (the VM's python3), which has no datetime.UTC
"""Rent Prime Intellect pods, and a watchdog that stops the ones we created.

Standard library only, so it runs on any machine with Python 3.10+. One ledger
(JSON lines) records every pod the cooperating agents create: its owner, hourly
price, and maximum lifetime. The watchdog pass (``watch``), run every 5 minutes by
``run_watchdog.sh``, acts only on *our* pods: those in the ledger under one of the
policy's ``owners``, or named with one of its ``name_prefixes``. It never touches a
pod someone else created on the same account; it only logs it. For our pods:

- it terminates any pod older than its maximum lifetime (the ledger's, else the
  policy's; never more than 8 hours);
- it terminates an owner's pods once that owner's estimated spend reaches its cap
  (``owner_caps`` in ``policy.json``);
- it terminates every one of our pods once our estimated total reaches the spend cap.

The estimate is price x hours for every ledger pod, from creation to termination
(or now), plus any running pod of ours the ledger lacks (named with our prefix).
It is an estimate, not the bill: ``history`` shows what Prime billed, and
``wallet`` the account's balance.

The API key comes from ``PRIME_API_KEY`` or ``~/.config/benchflow/primeintellect.env``
(``PRIME_ENV_FILE``) and is never printed. ``PRIME_TEAM_ID``, from the same places,
bills new pods to that team's wallet and makes ``wallet`` and the watchdog read it;
without it, Prime bills the key owner's personal wallet. State lives in ``--dir`` (default
``$PRIME_POD_DIR`` or ``~/prime-pods``): ``ledger.jsonl``, ``policy.json``,
``watchdog.log``, ``watchdog_state.json``, ``ssh_key_id``. ``policy.json`` is read
on every pass, so a new cap takes effect without a restart. Touch ``STOP_ALL``
there to make the next pass terminate every pod.

Commands::

    prime_pods.py offers [--gpu-type H100_80GB] [--gpu-count 2]
    prime_pods.py ssh-key-register --name NAME --pub ~/.ssh/prime_bf.pub
    prime_pods.py create --owner ME --gpu-type H100_80GB --gpu-count 2 \
        --image ubuntu_22_cuda_12 --name NAME --max-hours 6 [--provider P] [--cloud-id C]
    prime_pods.py register POD_ID --owner ME --max-hours 6   # a pod made elsewhere
    prime_pods.py wait POD_ID          # until ACTIVE with SSH; prints user@host:port
    prime_pods.py status POD_ID
    prime_pods.py list | spend | history
    prime_pods.py delete POD_ID
    prime_pods.py watch                # one watchdog pass
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

API = os.environ.get("PRIME_API_URL", "https://api.primeintellect.ai/api/v1")
HARD_MAX_HOURS = 8.0
DEFAULT_SPEND_CAP = 1400.0
# Our running pods that no ledger entry names (matched by name prefix).
UNATTRIBUTED = "unattributed"
DEFAULT_OWNERS = ("rl-prime", "miles")
DEFAULT_NAME_PREFIXES = ("rl-prime-", "rl-miles-", "miles-")
# Ledger records written before the owner column existed were all rl-prime's.
LEGACY_OWNER = "rl-prime"
# A ledger pod missing from /pods/ counts as ended only after the watchdog saw it
# running, or once it is this old: a pod can take a moment to appear after create.
UNSEEN_GRACE_SECONDS = 900


class ApiError(RuntimeError):
    def __init__(self, status: int | None, detail: str) -> None:
        super().__init__(f"HTTP {status}: {detail}" if status else detail)
        self.status = status


# --- time and money ----------------------------------------------------------


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_time(value: str | None) -> datetime | None:
    """Parse the API's ISO timestamps; a naive timestamp is UTC."""
    if not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hours_between(start: datetime, end: datetime) -> float:
    return max(0.0, (end - start).total_seconds() / 3600.0)


# --- the API -------------------------------------------------------------------


def env_file() -> Path:
    return Path(
        os.environ.get("PRIME_ENV_FILE", "~/.config/benchflow/primeintellect.env")
    ).expanduser()


def env_file_value(name: str, lines: list[str]) -> str | None:
    for line in lines:
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if line.startswith(name + "="):
            value = line.split("=", 1)[1].strip().strip("'\"")
            if value:
                return value
    return None


def load_key() -> str:
    key = os.environ.get("PRIME_API_KEY", "").strip()
    if key:
        return key
    path = env_file()
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise SystemExit(
            f"PRIME_API_KEY is not set and {path} is unreadable: {exc}"
        ) from None
    value = env_file_value("PRIME_API_KEY", lines)
    if value:
        return value
    raise SystemExit(f"PRIME_API_KEY is not set and not found in {path}")


def load_team_id() -> str | None:
    """The team whose wallet pays, or None for the key owner's personal wallet."""
    team = os.environ.get("PRIME_TEAM_ID", "").strip()
    if team:
        return team
    try:
        lines = env_file().read_text().splitlines()
    except OSError:
        return None
    return env_file_value("PRIME_TEAM_ID", lines)


class Prime:
    def __init__(
        self,
        key: str,
        *,
        team_id: str | None = None,
        base: str = API,
        timeout: float = 60.0,
    ) -> None:
        self._key = key
        self.team_id = team_id
        self.base = base.rstrip("/")
        self.timeout = timeout

    def request(
        self,
        method: str,
        path: str,
        *,
        body: Any = None,
        params: dict[str, Any] | None = None,
        retries: int = 3,
    ) -> Any:
        url = self.base + path
        if params:
            url += "?" + urllib.parse.urlencode(params, doseq=True)
        data = json.dumps(body).encode() if body is not None else None
        last: Exception | None = None
        for attempt in range(retries):
            req = urllib.request.Request(
                url,
                data=data,
                method=method,
                headers={
                    "Authorization": f"Bearer {self._key}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read()
                return json.loads(raw) if raw.strip() else None
            except urllib.error.HTTPError as exc:
                detail = exc.read()[:1500].decode(errors="replace")
                last = ApiError(exc.code, detail)
                if exc.code < 500 and exc.code != 429:
                    raise last from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last = ApiError(None, f"{type(exc).__name__}: {exc}")
            time.sleep(min(30.0, 2.0 * 2**attempt))
        assert last is not None
        raise last

    def pods(self) -> list[dict]:
        pods: list[dict] = []
        offset = 0
        while True:
            page = self.request(
                "GET", "/pods/", params={"offset": offset, "limit": 100}
            )
            data = (page or {}).get("data") or []
            pods.extend(data)
            total = int((page or {}).get("total_count") or 0)
            offset += len(data)
            if not data or offset >= total:
                return pods

    def pod(self, pod_id: str) -> dict:
        return self.request("GET", f"/pods/{pod_id}") or {}

    def history(self, limit: int = 100) -> list[dict]:
        page = self.request(
            "GET", "/pods/history", params={"offset": 0, "limit": limit}
        )
        return (page or {}).get("data") or []

    def status(self, pod_id: str) -> dict:
        page = self.request("GET", "/pods/status", params={"pod_ids": [pod_id]})
        rows = (page or {}).get("data") or []
        return rows[0] if rows else {}

    def delete(self, pod_id: str) -> Any:
        return self.request("DELETE", f"/pods/{pod_id}")

    def wallet(self) -> dict:
        params = {"teamId": self.team_id} if self.team_id else None
        return self.request("GET", "/billing/wallet", params=params) or {}

    def offers(self, gpu_type: str | None = None) -> list[dict]:
        params = {"gpu_type": gpu_type} if gpu_type else None
        page = self.request("GET", "/availability/", params=params) or {}
        return [offer for group in page.values() for offer in (group or [])]


# --- the ledger and the policy -----------------------------------------------------


@dataclass
class LedgerPod:
    id: str
    name: str | None
    created_at: datetime
    price_hr: float
    max_hours: float
    owner: str = LEGACY_OWNER
    gpu_type: str | None = None
    gpu_count: int | None = None
    ended_at: datetime | None = None
    billed: float | None = None


def ledger_append(directory: Path, record: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True) + "\n"
    # One small O_APPEND write per record, so concurrent writers never interleave.
    fd = os.open(
        directory / "ledger.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
    )
    try:
        os.write(fd, line.encode())
    finally:
        os.close(fd)


def ledger_read(directory: Path) -> dict[str, LedgerPod]:
    path = directory / "ledger.jsonl"
    pods: dict[str, LedgerPod] = {}
    if not path.exists():
        return pods
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            print(f"ledger line {number} is not JSON; skipped", file=sys.stderr)
            continue
        pod_id = str(record.get("id") or "")
        if not pod_id:
            continue
        event = record.get("event")
        if event == "create":
            pods[pod_id] = LedgerPod(
                id=pod_id,
                name=record.get("name"),
                created_at=parse_time(record["created_at"]) or utcnow(),
                price_hr=float(record.get("price_hr") or 0.0),
                max_hours=min(
                    HARD_MAX_HOURS, float(record.get("max_hours") or HARD_MAX_HOURS)
                ),
                owner=str(record.get("owner") or LEGACY_OWNER),
                gpu_type=record.get("gpu_type"),
                gpu_count=record.get("gpu_count"),
            )
        elif event == "ended" and pod_id in pods:
            ended = parse_time(record.get("ended_at"))
            pod = pods[pod_id]
            if ended is not None and (pod.ended_at is None or ended < pod.ended_at):
                pod.ended_at = ended
            if record.get("billed") is not None:
                pod.billed = float(record["billed"])
    return pods


@dataclass
class Policy:
    max_hours: float = HARD_MAX_HOURS
    spend_cap: float = DEFAULT_SPEND_CAP
    owner_caps: dict[str, float] = field(default_factory=dict)
    owners: tuple[str, ...] = DEFAULT_OWNERS
    name_prefixes: tuple[str, ...] = DEFAULT_NAME_PREFIXES


def load_policy(directory: Path, *, max_hours: float, spend_cap: float) -> Policy:
    """The command line's limits, tightened by ``policy.json`` if it is there.

    A policy file can lower the global limits and set per-owner caps; it can never
    raise a limit above the command line's, nor the lifetime above 8 hours."""
    policy = Policy(min(max_hours, HARD_MAX_HOURS), spend_cap)
    path = directory / "policy.json"
    if not path.exists():
        return policy
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"policy.json unreadable, using the command line's limits: {exc}",
            file=sys.stderr,
        )
        return policy
    if data.get("max_hours") is not None:
        policy.max_hours = min(policy.max_hours, float(data["max_hours"]))
    if data.get("spend_cap") is not None:
        policy.spend_cap = min(policy.spend_cap, float(data["spend_cap"]))
    policy.owner_caps = {
        str(k): float(v) for k, v in (data.get("owner_caps") or {}).items()
    }
    if data.get("owners"):
        policy.owners = tuple(str(owner) for owner in data["owners"])
    if data.get("name_prefixes"):
        policy.name_prefixes = tuple(str(prefix) for prefix in data["name_prefixes"])
    return policy


def is_ours(
    pod: dict,
    ledger: dict[str, LedgerPod],
    owners: tuple[str, ...],
    prefixes: tuple[str, ...],
) -> bool:
    """A pod we created: in the ledger under one of our owners, or named with our prefix."""
    entry = ledger.get(str(pod.get("id")))
    if entry is not None and entry.owner in owners:
        return True
    return str(pod.get("name") or "").startswith(tuple(prefixes)) if prefixes else False


# --- the watchdog's decisions (pure, so they are testable offline) ---------------


@dataclass
class Spend:
    total: float
    ledger: float
    foreign: float
    lines: list[str]
    by_owner: dict[str, float] = field(default_factory=dict)


def estimate_spend(
    ledger: dict[str, LedgerPod],
    active: list[dict],
    now: datetime,
) -> Spend:
    """Price x hours for every ledger pod, plus every running pod the ledger lacks.

    ``active`` should hold only our pods (see ``is_ours``). A ledger pod that is not
    running counts until its recorded end; with no end recorded it counts until now
    (the watchdog records the end when it notices).
    """
    active_by_id = {str(pod.get("id")): pod for pod in active}
    ledger_total = 0.0
    foreign_total = 0.0
    by_owner: dict[str, float] = {}
    lines: list[str] = []
    for pod in ledger.values():
        running = active_by_id.get(pod.id)
        start = pod.created_at
        price = pod.price_hr
        if running is not None:
            api_start = parse_time(running.get("createdAt"))
            if api_start is not None and api_start < start:
                start = api_start
            price = max(price, float(running.get("priceHr") or 0.0))
            end = now
        else:
            end = pod.ended_at or now
        cost = price * hours_between(start, end)
        ledger_total += cost
        by_owner[pod.owner] = by_owner.get(pod.owner, 0.0) + cost
        lines.append(
            f"{pod.id} {pod.name or '-'} [{pod.owner}] ${price:.3f}/h x "
            f"{hours_between(start, end):.2f}h = ${cost:.2f}"
        )
    for pod_id, pod in active_by_id.items():
        if pod_id in ledger:
            continue
        start = parse_time(pod.get("createdAt")) or now
        price = float(pod.get("priceHr") or 0.0)
        cost = price * hours_between(start, now)
        foreign_total += cost
        by_owner[UNATTRIBUTED] = by_owner.get(UNATTRIBUTED, 0.0) + cost
        lines.append(
            f"{pod_id} {pod.get('name') or '-'} [{UNATTRIBUTED}] ${price:.3f}/h = ${cost:.2f}"
        )
    return Spend(
        ledger_total + foreign_total, ledger_total, foreign_total, lines, by_owner
    )


def pods_to_terminate(
    ledger: dict[str, LedgerPod],
    active: list[dict],
    now: datetime,
    *,
    max_hours: float,
    spend_cap: float,
    owner_caps: dict[str, float] | None = None,
    stop_all: bool = False,
    owners: tuple[str, ...] = DEFAULT_OWNERS,
    name_prefixes: tuple[str, ...] = DEFAULT_NAME_PREFIXES,
) -> tuple[list[tuple[str, str]], Spend]:
    """Which of our running pods to terminate, each with its reason. Pods we did not
    create are neither counted nor terminated."""
    max_hours = min(max_hours, HARD_MAX_HOURS)
    owner_caps = owner_caps or {}
    ours = [pod for pod in active if is_ours(pod, ledger, owners, name_prefixes)]
    spend = estimate_spend(ledger, ours, now)
    doomed: list[tuple[str, str]] = []
    for pod in ours:
        pod_id = str(pod.get("id"))
        if str(pod.get("status", "")).upper() in {"TERMINATED", "TERMINATING"}:
            continue
        if stop_all:
            doomed.append((pod_id, "STOP_ALL file present"))
            continue
        if spend.total >= spend_cap:
            doomed.append(
                (pod_id, f"estimated spend ${spend.total:.2f} >= cap ${spend_cap:.2f}")
            )
            continue
        entry = ledger.get(pod_id)
        owner = entry.owner if entry is not None else UNATTRIBUTED
        cap = owner_caps.get(owner)
        if cap is not None and spend.by_owner.get(owner, 0.0) >= cap:
            doomed.append(
                (
                    pod_id,
                    f"owner {owner} spend ${spend.by_owner.get(owner, 0.0):.2f} >= its cap ${cap:.2f}",
                )
            )
            continue
        limit = max_hours
        if entry is not None:
            limit = min(limit, entry.max_hours)
        created = parse_time(pod.get("createdAt"))
        if entry is not None and (created is None or entry.created_at < created):
            created = entry.created_at
        if created is None:
            continue
        age = hours_between(created, now)
        if age >= limit:
            doomed.append((pod_id, f"age {age:.2f}h >= max {limit:.2f}h"))
    return doomed, spend


# --- commands -----------------------------------------------------------------------


def log(directory: Path, message: str) -> None:
    line = f"{iso(utcnow())} {message}"
    print(line, flush=True)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "watchdog.log", "a") as handle:
        handle.write(line + "\n")


def cmd_watch(prime: Prime, args: argparse.Namespace) -> int:
    directory: Path = args.dir
    now = utcnow()
    policy = load_policy(directory, max_hours=args.max_hours, spend_cap=args.spend_cap)
    ledger = ledger_read(directory)
    try:
        active = prime.pods()
    except ApiError as exc:
        log(directory, f"pass FAILED listing pods: {exc}")
        return 1
    state_path = directory / "watchdog_state.json"
    try:
        state = json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError):
        state = {}
    seen: dict[str, str] = state.get("last_seen", {})
    active_ids = {str(pod.get("id")) for pod in active}
    for pod_id in active_ids:
        seen[pod_id] = iso(now)

    # Ledger pods that stopped running: record their end (from history when it has it).
    gone = [
        pod
        for pod in ledger.values()
        if pod.ended_at is None
        and pod.id not in active_ids
        and (
            pod.id in seen
            or hours_between(pod.created_at, now) * 3600 > UNSEEN_GRACE_SECONDS
        )
    ]
    if gone:
        try:
            history = {str(item.get("id")): item for item in prime.history()}
        except ApiError:
            history = {}
        for pod in gone:
            item = history.get(pod.id, {})
            ended = (
                parse_time(item.get("terminatedAt"))
                or parse_time(seen.get(pod.id))
                or now
            )
            ledger_append(
                directory,
                {
                    "event": "ended",
                    "id": pod.id,
                    "ended_at": iso(ended),
                    "billed": item.get("totalBilledPrice"),
                    "source": "watchdog",
                },
            )
            log(
                directory,
                f"ended: {pod.id} {pod.name or '-'} [{pod.owner}] at {iso(ended)}"
                f" billed={item.get('totalBilledPrice')}",
            )
        ledger = ledger_read(directory)

    doomed, spend = pods_to_terminate(
        ledger,
        active,
        now,
        max_hours=policy.max_hours,
        spend_cap=policy.spend_cap,
        owner_caps=policy.owner_caps,
        stop_all=(directory / "STOP_ALL").exists(),
        owners=policy.owners,
        name_prefixes=policy.name_prefixes,
    )
    ours = [
        pod
        for pod in active
        if is_ours(pod, ledger, policy.owners, policy.name_prefixes)
    ]
    others = [pod for pod in active if pod not in ours]

    def label(pod: dict) -> str:
        entry = ledger.get(str(pod.get("id")))
        owner = entry.owner if entry is not None else UNATTRIBUTED
        age = hours_between(parse_time(pod.get("createdAt")) or now, now)
        return f"{pod.get('id')}:{pod.get('name') or '-'}:{owner}:{pod.get('status')}:{age:.2f}h"

    owners = ", ".join(
        f"{owner} ${value:.2f}"
        + (f"/${policy.owner_caps[owner]:.0f}" if owner in policy.owner_caps else "")
        for owner, value in sorted(spend.by_owner.items())
    )
    not_ours = ", ".join(
        f"{pod.get('id')}:{pod.get('name') or '-'}:${float(pod.get('priceHr') or 0):.2f}/h"
        for pod in others
    )
    # The account's balance is shared with whoever else uses it: warn, never act on it
    # (terminating would lose unexported work; the owner stops at a checkpoint instead).
    wallet_note = ""
    try:
        balance = float(prime.wallet().get("balance_usd") or 0.0)
        burn = sum(float(pod.get("priceHr") or 0.0) for pod in active)
        runway = balance / burn if burn > 0 else float("inf")
        wallet_note = (
            f" wallet=${balance:.2f} runway={runway:.1f}h"
            if burn > 0
            else f" wallet=${balance:.2f}"
        )
        if runway < 1.0:
            wallet_note += " LOW-WALLET: stop at a checkpoint and export now"
    except ApiError:
        wallet_note = " wallet=?"
    log(
        directory,
        f"pass: ours={len(ours)} [{', '.join(label(pod) for pod in ours)}]"
        f" spend_est=${spend.total:.2f} ({owners or 'none'}) cap=${policy.spend_cap:.0f}"
        f" max={policy.max_hours:g}h terminate={len(doomed)}{wallet_note}"
        + (f" | not ours, never touched: [{not_ours}]" if others else ""),
    )
    failures = 0
    for pod_id, reason in doomed:
        log(directory, f"TERMINATE {pod_id}: {reason}")
        if args.dry_run:
            continue
        try:
            prime.delete(pod_id)
        except ApiError as exc:
            failures += 1
            log(directory, f"terminate {pod_id} FAILED: {exc}")
    state["last_seen"] = seen
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, sort_keys=True))
    tmp.replace(state_path)
    return 1 if failures else 0


def pick_offer(offers: list[dict], args: argparse.Namespace) -> dict:
    matches = [
        offer
        for offer in offers
        if offer.get("gpuType") == args.gpu_type
        and int(offer.get("gpuCount") or 0) == args.gpu_count
        and (args.provider is None or offer.get("provider") == args.provider)
        and (args.cloud_id is None or offer.get("cloudId") == args.cloud_id)
        and (args.data_center is None or offer.get("dataCenter") == args.data_center)
        and args.image in (offer.get("images") or [])
        and str(offer.get("stockStatus", "")).lower()
        not in {"unavailable", "out of stock"}
        and not (offer.get("isSpot") and not args.allow_spot)
    ]
    if not matches:
        raise SystemExit(
            f"no available offer for {args.gpu_count}x {args.gpu_type} with image "
            f"{args.image!r}; see `prime_pods.py offers --gpu-type {args.gpu_type}`"
        )
    return min(matches, key=lambda offer: offer_price(offer) or float("inf"))


def offer_price(offer: dict) -> float:
    prices = offer.get("prices") or {}
    value = prices.get("onDemand") or prices.get("communityPrice")
    return float(value or 0.0)


def wallet_label(prime: Prime) -> str:
    return f"team {prime.team_id}" if prime.team_id else "the personal wallet"


def require_owner(args: argparse.Namespace) -> str:
    owner = (args.owner or os.environ.get("PRIME_POD_OWNER", "")).strip()
    if not owner:
        raise SystemExit("name the pod's owner: --owner NAME (or $PRIME_POD_OWNER)")
    return owner


def cmd_create(prime: Prime, args: argparse.Namespace) -> int:
    if args.max_hours > HARD_MAX_HOURS:
        raise SystemExit(f"--max-hours is at most {HARD_MAX_HOURS:g}")
    directory: Path = args.dir
    pid_file = directory / "watchdog.pid"
    if not args.no_watchdog_check and not _watchdog_running(pid_file):
        raise SystemExit(
            f"the watchdog is not running ({pid_file}); start run_watchdog.sh first "
            "(or pass --no-watchdog-check)"
        )
    owner = require_owner(args)
    offer = pick_offer(prime.offers(args.gpu_type), args)
    ssh_key_id = args.ssh_key_id
    key_file = directory / "ssh_key_id"
    if ssh_key_id is None and key_file.exists():
        ssh_key_id = key_file.read_text().strip() or None
    pod: dict[str, Any] = {
        "name": args.name,
        "cloudId": offer["cloudId"],
        "gpuType": offer["gpuType"],
        "socket": offer["socket"],
        "gpuCount": offer["gpuCount"],
        "diskSize": args.disk or (offer.get("disk") or {}).get("defaultCount"),
        "vcpus": (offer.get("vcpu") or {}).get("defaultCount"),
        "memory": (offer.get("memory") or {}).get("defaultCount"),
        "image": args.image,
        "dataCenterId": offer.get("dataCenter"),
        "autoRestart": False,
        "sshKeyId": ssh_key_id,
    }
    body: dict[str, Any] = {
        "pod": {key: value for key, value in pod.items() if value is not None},
        "provider": {"type": offer["provider"]} if offer.get("provider") else {},
    }
    # Without a team, Prime bills the key owner's personal wallet.
    if prime.team_id:
        body["team"] = {"teamId": prime.team_id}
    price = offer_price(offer)
    print(
        f"creating {offer['gpuCount']}x {offer['gpuType']} ({offer['cloudId']}, "
        f"{offer.get('provider')}/{offer.get('dataCenter')}) at ${price:.3f}/h, "
        f"max {args.max_hours:g}h, image {args.image}, owner {owner}, "
        f"billed to {wallet_label(prime)}",
        flush=True,
    )
    requested = utcnow()
    created = prime.request("POST", "/pods/", body=body, retries=1)
    pod_id = str((created or {}).get("id") or "")
    if not pod_id:
        raise SystemExit(f"create returned no pod id: {str(created)[:500]}")
    ledger_append(
        directory,
        {
            "event": "create",
            "id": pod_id,
            "name": args.name,
            "owner": owner,
            # The request time, not the API's: never later than billing starts.
            "created_at": iso(
                min(requested, parse_time(created.get("createdAt")) or requested)
            ),
            "price_hr": max(price, float(created.get("priceHr") or 0.0)),
            "max_hours": args.max_hours,
            "gpu_type": offer["gpuType"],
            "gpu_count": offer["gpuCount"],
            "cloud_id": offer["cloudId"],
            "provider": offer.get("provider"),
            "team_id": prime.team_id,
        },
    )
    log(
        directory,
        f"created {pod_id} {args.name} [{owner}] {offer['gpuCount']}x{offer['gpuType']}"
        f" ${price:.3f}/h max {args.max_hours:g}h, billed to {wallet_label(prime)}",
    )
    print(pod_id)
    return 0


def cmd_register(prime: Prime, args: argparse.Namespace) -> int:
    """Put a pod created some other way (Prime's CLI or web app) in the ledger."""
    if args.max_hours > HARD_MAX_HOURS:
        raise SystemExit(f"--max-hours is at most {HARD_MAX_HOURS:g}")
    owner = require_owner(args)
    if args.pod_id in ledger_read(args.dir):
        raise SystemExit(f"{args.pod_id} is already in the ledger")
    pod = prime.pod(args.pod_id)
    created = parse_time(pod.get("createdAt")) or utcnow()
    price = max(float(pod.get("priceHr") or 0.0), float(args.price_hr or 0.0))
    ledger_append(
        args.dir,
        {
            "event": "create",
            "id": args.pod_id,
            "name": pod.get("name"),
            "owner": owner,
            "created_at": iso(created),
            "price_hr": price,
            "max_hours": args.max_hours,
            "gpu_type": pod.get("gpuName"),
            "gpu_count": pod.get("gpuCount"),
            "source": "register",
        },
    )
    log(
        args.dir,
        f"registered {args.pod_id} {pod.get('name')} [{owner}] ${price:.3f}/h max {args.max_hours:g}h",
    )
    return 0


def ssh_target(status: dict) -> str | None:
    """`user@host:port` from a status row's sshConnection, e.g. 'root@1.2.3.4 -p 22'."""
    raw = status.get("sshConnection")
    if isinstance(raw, list):
        raw = next((item for item in raw if item), None)
    if not raw or not isinstance(raw, str):
        return None
    parts = raw.replace("ssh ", "").split()
    target = next((part for part in parts if "@" in part), None)
    if target is None:
        return None
    port = "22"
    if "-p" in parts:
        index = parts.index("-p")
        if index + 1 < len(parts):
            port = parts[index + 1]
    return f"{target}:{port}"


def cmd_wait(prime: Prime, args: argparse.Namespace) -> int:
    deadline = time.monotonic() + args.timeout
    last = None
    while time.monotonic() < deadline:
        status = prime.status(args.pod_id)
        summary = (
            status.get("status"),
            status.get("installationProgress"),
            status.get("installationFailure"),
        )
        if summary != last:
            print(
                f"{iso(utcnow())} {args.pod_id}: status={summary[0]} progress={summary[1]} failure={summary[2]}",
                flush=True,
            )
            last = summary
        if status.get("installationFailure"):
            return 2
        target = ssh_target(status)
        if str(status.get("status", "")).upper() == "ACTIVE" and target:
            print(target)
            return 0
        time.sleep(15)
    print(f"timed out after {args.timeout}s", file=sys.stderr)
    return 1


def cmd_status(prime: Prime, args: argparse.Namespace) -> int:
    status = prime.status(args.pod_id)
    keep = (
        "podId",
        "status",
        "installationProgress",
        "installationFailure",
        "priceHr",
        "sshConnection",
        "ip",
    )
    print(json.dumps({key: status.get(key) for key in keep}, indent=1))
    return 0


def cmd_list(prime: Prime, args: argparse.Namespace) -> int:
    now = utcnow()
    ledger = ledger_read(args.dir)
    for pod in prime.pods():
        created = parse_time(pod.get("createdAt")) or now
        entry = ledger.get(str(pod.get("id")))
        print(
            f"{pod.get('id')} {pod.get('name')} [{entry.owner if entry else UNATTRIBUTED}]"
            f" {pod.get('status')} {pod.get('gpuCount')}x{pod.get('gpuName')}"
            f" ${float(pod.get('priceHr') or 0):.3f}/h age {hours_between(created, now):.2f}h"
        )
    return 0


def cmd_spend(prime: Prime, args: argparse.Namespace) -> int:
    ledger = ledger_read(args.dir)
    policy = load_policy(
        args.dir, max_hours=HARD_MAX_HOURS, spend_cap=DEFAULT_SPEND_CAP
    )
    ours = [
        pod
        for pod in prime.pods()
        if is_ours(pod, ledger, policy.owners, policy.name_prefixes)
    ]
    spend = estimate_spend(ledger, ours, utcnow())
    for line in spend.lines:
        print(line)
    for owner, value in sorted(spend.by_owner.items()):
        print(f"  {owner}: ${value:.2f}")
    print(
        f"estimated spend ${spend.total:.2f} (ledger ${spend.ledger:.2f}, not in ledger ${spend.foreign:.2f})"
    )
    return 0


def cmd_history(prime: Prime, args: argparse.Namespace) -> int:
    ledger = ledger_read(args.dir)
    total = 0.0
    for item in prime.history():
        entry = ledger.get(str(item.get("id")))
        if args.all or entry is not None:
            billed = float(item.get("totalBilledPrice") or 0.0)
            total += billed
            print(
                f"{item.get('id')} {item.get('name')} [{entry.owner if entry else UNATTRIBUTED}]"
                f" {item.get('gpuCount')}x{item.get('gpuName')} ${float(item.get('priceHr') or 0):.3f}/h"
                f" {item.get('createdAt')} -> {item.get('terminatedAt')} billed ${billed:.2f}"
            )
    print(f"billed total ${total:.2f}")
    return 0


def cmd_wallet(prime: Prime, args: argparse.Namespace) -> int:
    """The account's balance, and how long our running pods can run on it."""
    wallet = prime.wallet()
    balance = float(wallet.get("balance_usd") or 0.0)
    ledger = ledger_read(args.dir)
    policy = load_policy(
        args.dir, max_hours=HARD_MAX_HOURS, spend_cap=DEFAULT_SPEND_CAP
    )
    pods = prime.pods()
    burn = sum(float(pod.get("priceHr") or 0.0) for pod in pods)
    ours = sum(
        float(pod.get("priceHr") or 0.0)
        for pod in pods
        if is_ours(pod, ledger, policy.owners, policy.name_prefixes)
    )
    runway = f"{balance / burn:.1f}h" if burn > 0 else "no pod running"
    print(
        f"{wallet_label(prime)}: balance ${balance:.2f}; running pods ${burn:.2f}/h"
        f" (ours ${ours:.2f}/h); runway {runway}"
    )
    return 0


def cmd_delete(prime: Prime, args: argparse.Namespace) -> int:
    prime.delete(args.pod_id)
    ledger_append(
        args.dir,
        {
            "event": "ended",
            "id": args.pod_id,
            "ended_at": iso(utcnow()),
            "source": "delete",
        },
    )
    log(args.dir, f"deleted {args.pod_id} (requested)")
    for _ in range(40):
        if all(str(pod.get("id")) != args.pod_id for pod in prime.pods()):
            print(f"{args.pod_id} is gone from /pods/")
            return 0
        time.sleep(15)
    print(f"{args.pod_id} still listed after 10 minutes", file=sys.stderr)
    return 1


def cmd_ssh_key_register(prime: Prime, args: argparse.Namespace) -> int:
    public_key = Path(args.pub).expanduser().read_text().strip()
    if "PRIVATE KEY" in public_key:
        raise SystemExit("that is a private key; pass the .pub file")
    created = prime.request(
        "POST",
        "/ssh_keys/",
        body={"name": args.name, "publicKey": public_key},
        retries=1,
    )
    key_id = str((created or {}).get("id") or "")
    if not key_id:
        raise SystemExit(f"no key id in response: {str(created)[:300]}")
    (args.dir / "ssh_key_id").write_text(key_id + "\n")
    print(key_id)
    return 0


def cmd_offers(prime: Prime, args: argparse.Namespace) -> int:
    rows = []
    for offer in prime.offers(args.gpu_type):
        if args.gpu_count and int(offer.get("gpuCount") or 0) != args.gpu_count:
            continue
        rows.append(
            (
                offer_price(offer),
                f"{offer.get('gpuCount')}x {offer.get('gpuType')} {offer.get('socket')} "
                f"{offer.get('provider')}/{offer.get('dataCenter')} {offer.get('cloudId')} "
                f"${offer_price(offer):.3f}/h spot={bool(offer.get('isSpot'))} "
                f"stock={offer.get('stockStatus')} images={','.join(offer.get('images') or [])}",
            )
        )
    for _, line in sorted(rows):
        print(line)
    return 0


def _watchdog_running(pid_file: Path) -> bool:
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dir",
        type=lambda value: Path(value).expanduser(),
        default=Path(os.environ.get("PRIME_POD_DIR", "~/prime-pods")).expanduser(),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    watch = sub.add_parser("watch", help="one watchdog pass")
    watch.add_argument("--max-hours", type=float, default=HARD_MAX_HOURS)
    watch.add_argument("--spend-cap", type=float, default=DEFAULT_SPEND_CAP)
    watch.add_argument("--dry-run", action="store_true")

    create = sub.add_parser("create")
    create.add_argument(
        "--owner", help="who the pod belongs to (default $PRIME_POD_OWNER)"
    )
    create.add_argument("--gpu-type", required=True)
    create.add_argument("--gpu-count", type=int, default=1)
    create.add_argument("--image", default="ubuntu_22_cuda_12")
    create.add_argument("--name", required=True)
    create.add_argument("--max-hours", type=float, required=True)
    create.add_argument("--provider")
    create.add_argument("--cloud-id")
    create.add_argument("--data-center")
    create.add_argument("--disk", type=int)
    create.add_argument("--ssh-key-id")
    create.add_argument("--allow-spot", action="store_true")
    create.add_argument("--no-watchdog-check", action="store_true")

    register = sub.add_parser(
        "register", help="add a pod created elsewhere to the ledger"
    )
    register.add_argument("pod_id")
    register.add_argument(
        "--owner", help="who the pod belongs to (default $PRIME_POD_OWNER)"
    )
    register.add_argument("--max-hours", type=float, required=True)
    register.add_argument("--price-hr", type=float, help="if above the API's price")

    for name in ("wait", "status", "delete"):
        command = sub.add_parser(name)
        command.add_argument("pod_id")
        if name == "wait":
            command.add_argument("--timeout", type=int, default=1800)

    sub.add_parser("list")
    sub.add_parser("spend")
    sub.add_parser("wallet")
    history = sub.add_parser("history")
    history.add_argument(
        "--all",
        action="store_true",
        help="every pod on the account, not only the ledger's",
    )
    offers = sub.add_parser("offers")
    offers.add_argument("--gpu-type")
    offers.add_argument("--gpu-count", type=int)
    key = sub.add_parser("ssh-key-register")
    key.add_argument("--name", required=True)
    key.add_argument("--pub", required=True)

    args = parser.parse_args(argv)
    args.dir.mkdir(parents=True, exist_ok=True)
    prime = Prime(load_key(), team_id=load_team_id())
    handler = {
        "watch": cmd_watch,
        "create": cmd_create,
        "register": cmd_register,
        "wait": cmd_wait,
        "status": cmd_status,
        "delete": cmd_delete,
        "list": cmd_list,
        "spend": cmd_spend,
        "wallet": cmd_wallet,
        "history": cmd_history,
        "offers": cmd_offers,
        "ssh-key-register": cmd_ssh_key_register,
    }[args.command]
    return handler(prime, args)


if __name__ == "__main__":
    sys.exit(main())
