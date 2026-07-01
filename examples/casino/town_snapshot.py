"""Live Stanford-Town-style Casino Floor viewer + per-agent run drawer.

The MAIN BOARD is casinobench's own town viewer (``viewer_html.render_html`` — the
top-down canvas floor where agents walk to game stations, with Standings /
Now-playing / Floor-log panels), set to LIVE mode so it polls ``/_admin/viewer``
and animates in real time. We add a click-to-expand **agent dossier**: clicking a
player in Standings opens a side drawer with that agent's raw run — every thought
and ``casino`` move.

Because the multiplayer World does not serve ``/_admin/viewer``, this sidecar
builds it from ``/_admin/events`` via ``to_viewer_data`` and writes it (plus
``traj/<seat>.json`` + ``state.json``) into the served directory, so the browser
polls everything same-origin through the tunnel.

Run inside casinobench's env (for the casinobench imports):
    cd "$CASINOBENCH_DIR" && uv run python path/to/town_snapshot.py \
        <world_url> <run_dir> <serve_dir>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx
from casinobench.catalog import default_registry
from casinobench.event_log import EventLog
from casinobench.viewer_data import to_viewer_data
from casinobench.viewer_html import render_html

REG = default_registry()

# Injected before </body>. Two jobs:
#  (1) Fix the three side panels — share the aside height, each scrolls only when
#      its own content overflows (Standings/Now cap to content, Floor-log fills
#      the rest). The town's #ticker was overflow:hidden (clipped) — make it auto.
#  (2) Replace the old fixed bottom bar (it covered the Play/Step controls) with a
#      click-to-open AGENT DOSSIER: each Standings row is the handle; clicking it
#      slides in that player's run, styled in the town's Courier/gold felt.
PANEL = r"""
<style>
 /* the sidebar never overflows; its three panels share the height and each
    scrolls on its own when its content is too long (!important beats the town's
    inline flex:1 on the Floor-log panel). */
 aside{overflow:hidden}
 aside>.panel{min-height:0!important;overflow:hidden!important;display:flex!important;flex-direction:column!important}
 aside>.panel:nth-child(1){flex:1 1 0!important}      /* Standings */
 aside>.panel:nth-child(2){flex:1.5 1 0!important}    /* Now playing — usually the longest */
 aside>.panel:nth-child(3){flex:1.2 1 0!important}    /* Floor log */
 aside>.panel>h2{flex:0 0 auto!important}
 aside>.panel>table{display:block!important;overflow:auto!important;min-height:0!important;flex:1 1 auto!important}
 #now,#ticker{overflow:auto!important;min-height:0!important;flex:1 1 auto!important}
 #board tr{cursor:pointer}
 #board tr:hover{background:rgba(231,198,107,.12)}
 #board tr td:first-child::after{content:" \25B8";color:var(--gold-dim);opacity:.55}
 #runDrawer{position:fixed;top:0;right:0;width:min(440px,94vw);height:100%;
   background:linear-gradient(180deg,#1a0e18,#100810);border-left:1px solid var(--gold-dim);
   transform:translateX(100%);transition:transform .22s ease;z-index:60;display:flex;
   flex-direction:column;box-shadow:-18px 0 44px rgba(0,0,0,.55);
   font-family:"Courier New",ui-monospace,monospace}
 #runDrawer.open{transform:none}
 #runDrawer .hd{display:flex;align-items:baseline;gap:.5rem;padding:.7rem .9rem;border-bottom:1px solid var(--line)}
 #runDrawer .hd .t{color:var(--gold);font-weight:700;letter-spacing:.03em}
 #runDrawer .hd .s{color:var(--gold-dim);font-size:.62rem;letter-spacing:.18em;text-transform:uppercase}
 #runDrawer .hd .x{margin-left:auto;cursor:pointer;color:#9a8aa0;border:1px solid var(--line);
   border-radius:6px;padding:.12rem .55rem;font-size:.74rem}
 #runDrawer .hd .x:hover{color:var(--ink);border-color:var(--gold-dim)}
 #runBody{overflow:auto;padding:.1rem .9rem 1.2rem;flex:1 1 auto}
 #runBody .turn{border-top:1px solid #2a1726;padding:.5rem 0}
 #runBody .r{font-size:.6rem;letter-spacing:.16em;text-transform:uppercase;color:var(--gold-dim)}
 #runBody .think{white-space:pre-wrap;margin-top:.18rem;color:var(--ink);font-size:.82rem;line-height:1.5}
 #runBody .tool{font-size:.78rem;background:#0c0610;border:1px solid var(--line);
   border-left:2px solid var(--gold-dim);border-radius:5px;padding:.32rem .5rem;margin-top:.3rem;
   white-space:pre-wrap;color:#ffe9a6}
 #runBody .empty{color:#9a8aa0;padding:1rem 0;font-size:.84rem;line-height:1.5}
</style>
<div id="runDrawer">
  <div class="hd"><span class="s">dossier</span><span class="t" id="rdWho"></span>
    <span class="s" id="rdSub"></span><span class="x" onclick="closeRun()">close &#10005;</span></div>
  <div id="runBody"><div class="empty">Pick a player in Standings to read their run &mdash; every thought and casino move.</div></div>
</div>
<script>
let RD_SEAT=null, SEATS=[];
function esc(s){return (s??'').toString().replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function toolText(x){
  if(typeof x.command==='string') return x.command;
  const c=x.content;
  if(typeof c==='string') return c;
  if(Array.isArray(c)) return c.map(p=>(p&&p.content&&p.content.text)||p.text||(typeof p==='string'?p:'')).filter(Boolean).join(' ');
  return x.text||JSON.stringify(x);
}
function closeRun(){document.getElementById('runDrawer').classList.remove('open');RD_SEAT=null}
async function openRun(seat){
  RD_SEAT=seat;
  document.getElementById('runDrawer').classList.add('open');
  document.getElementById('rdWho').textContent=seat;
  const body=document.getElementById('runBody');
  try{
    const t=await fetch('traj/'+seat+'.json?ts='+Date.now()).then(r=>r.json());
    document.getElementById('rdSub').textContent=t.length?('· '+t.length+' steps'):'';
    if(!t.length){body.innerHTML='<div class="empty">No moves yet &mdash; this player just sat down.</div>';return}
    const follow = body.scrollTop+body.clientHeight >= body.scrollHeight-48;
    body.innerHTML=t.map(x=>{
      if(x.type==='tool_call')
        return '<div class="turn"><div class="r">casino &middot; '+esc(x.title||x.kind||'play')+'</div><div class="tool">'+esc(toolText(x))+'</div></div>';
      const role=x.type==='user_message'?'table rules':x.type==='agent_message'?'says':x.type==='agent_thought'?'thinks':esc(x.type);
      return '<div class="turn"><div class="r">'+role+'</div><div class="think">'+esc(x.text||x.content||'')+'</div></div>';
    }).join('');
    if(follow) body.scrollTop=body.scrollHeight;   // keep following the live tail
  }catch(e){body.innerHTML='<div class="empty">Run not available.</div>'}
}
document.addEventListener('click',function(e){
  const tr=e.target.closest&&e.target.closest('#board tr'); if(!tr) return;
  const txt=tr.textContent||''; const seat=SEATS.find(s=>txt.includes(s)); if(seat) openRun(seat);
});
let BOARD_ROSTER=null;   // the roster the page was rendered with
async function refreshSeats(){
  try{
    const st=await fetch('state.json?ts='+Date.now()).then(r=>r.json());
    SEATS=(st.agents||[]).map(a=>a.seat);
    const live=SEATS.slice().sort().join(',');
    if(BOARD_ROSTER===null)
      BOARD_ROSTER=(typeof RUN!=='undefined'&&RUN.players)?RUN.players.slice().sort().join(','):live;
    if(BOARD_ROSTER && live && live!==BOARD_ROSTER){   // a genuinely new run -> reseat
      // reload at most ONCE per distinct roster so a stale/mismatched state.json
      // can never spin an infinite reload loop (belt-and-suspenders).
      if(sessionStorage.getItem('bfRoster')!==live){ sessionStorage.setItem('bfRoster',live); location.reload(); }
      return;
    }
    if(RD_SEAT) openRun(RD_SEAT);
  }catch(e){}
}
refreshSeats(); setInterval(refreshSeats,3000);
</script>
"""


def _payload(world: str, run_dir: Path) -> tuple[dict, bool]:
    running = True
    try:
        jsonl = httpx.get(f"{world}/_admin/events", timeout=5).json()["jsonl"]
        standings = httpx.get(f"{world}/_admin/standings", timeout=5).json()
    except Exception:  # noqa: BLE001 — World down → persisted snapshot
        running = False
        ev = run_dir / "events.jsonl"
        jsonl = ev.read_text() if ev.exists() else ""
        fj = run_dir / "floor.json"
        standings = json.loads(fj.read_text()).get("standings", {}) if fj.exists() else {}
    events = list(EventLog.from_jsonl(jsonl).events) if jsonl.strip() else []
    players = sorted(standings) or sorted({e.actor for e in events if getattr(e, "actor", "")})
    data = to_viewer_data(events, players, 1000, REG, {"stake": 50})
    data["done"] = not running
    return data, running


def _roster(run_dir: Path) -> list[dict]:
    f = run_dir / "roster.json"
    return json.loads(f.read_text()) if f.exists() else []


def main() -> None:
    world, run_dir, serve = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    (serve / "traj").mkdir(parents=True, exist_ok=True)
    prev_players: list = []
    while True:
        try:
            data, running = _payload(world, run_dir)
            (serve / "_admin").mkdir(exist_ok=True)
            (serve / "_admin" / "viewer").write_text(json.dumps(data))
            # casinobench's live poll streams events but never re-seats the board,
            # so rebuild index.html whenever the ROSTER changes (a new run) — the
            # board, sprites and Standings then reflect the current players.
            players = data.get("players", [])
            if players and players != prev_players:
                html = render_html(data, live={"url": "", "interval": 1.0})
                html = html.replace("</body>", PANEL + "</body>", 1)
                (serve / "index.html").write_text(html)
                prev_players = list(players)
            roster = _roster(run_dir)
            try:
                standings = httpx.get(f"{world}/_admin/standings", timeout=4).json()
            except Exception:  # noqa: BLE001
                fj = run_dir / "floor.json"
                standings = json.loads(fj.read_text()).get("standings", {}) if fj.exists() else {}
            agents = []
            for r in roster:
                seat = r["seat"]
                p = run_dir / seat / "trajectory" / "acp_trajectory.jsonl"
                rows = []
                if p.exists():
                    for line in p.read_text().splitlines():
                        if line.strip():
                            try:
                                rows.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
                (serve / "traj" / f"{seat}.json").write_text(json.dumps(rows))
                agents.append({"seat": seat, "agent": r.get("agent"), "model": r.get("model"),
                               "chips": standings.get(seat),
                               "tool_calls": sum(1 for x in rows if x.get("type") == "tool_call")})
            (serve / "state.json").write_text(json.dumps(
                {"running": running, "agents": agents, "standings": standings}))
        except Exception as exc:  # noqa: BLE001
            print("town_snapshot:", type(exc).__name__, str(exc)[:140], flush=True)
        time.sleep(1.0)


if __name__ == "__main__":
    main()
