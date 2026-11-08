#!/usr/bin/env python3
"""
RFScout web dashboard. Python standard library only, no internet needed.

The board runs the SDR, the AI and this small web server. Open the page from a
laptop on the same cable/network (or from the board itself if it has a browser).

  python3 rfscout_web.py            # live SDR, serves on port 8000
  python3 rfscout_web.py --mock     # simulated SDR data
  python3 rfscout_web.py --port 9000
"""
import argparse, json, os, shutil, socket, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import rfscout as core
except ImportError:
    sys.exit("Put rfscout_web.py in the same folder as rfscout.py and run it from there.")

STATE = {"mock": False, "load_s": None, "checks": [0, 0], "last": None, "queries": 0}
AI_LOCK = threading.Lock()


def scan_json(res, max_pts=900):
    pts = res["points"]
    step = max(1, len(pts) // max_pts)
    floor = res["floor"]
    slim = []
    for i in range(0, len(pts), step):
        seg = pts[i:i + step]
        f = seg[len(seg) // 2][0]
        slim.append([round(f, 4), round(max(p for _, p in seg) - floor, 2)])
    return {"label": res["label"], "lo": res["lo"], "hi": res["hi"], "floor": round(floor, 1),
            "points": slim,
            "peaks": [{"freq": round(k["freq"], 3), "db": round(k["db"], 1), "band": k["band"], "desc": k["desc"]}
                      for k in res["peaks"]]}


def stats():
    return {"model": os.path.basename(core.MODEL), "load_s": STATE["load_s"], "mem_mb": core.peak_mem_mb(),
            "last": STATE["last"], "checks": STATE["checks"], "queries": STATE["queries"],
            "mock": STATE["mock"]}


def ask(q):
    """Handle one question. Returns a JSON-able dict with messages, scans and an AI reply."""
    msgs, scans, reply = [], [], None
    r = core.route(q)
    k = r[0]
    if k == "quit":
        msgs.append(["info", "The web dashboard keeps running. Stop it with Ctrl-C on the board."])
    elif k == "help":
        msgs.append(["info", core.HELP.replace("\n  quit", "")])
    elif k == "bands":
        msgs.append(["info", "\n".join(f"{lo:8.3f}-{hi:8.3f} MHz  {n}" for lo, hi, n, _ in core.BANDS)])
    elif k == "stats":
        s = stats()
        msgs.append(["info", f"Model {s['model']} | load {s['load_s'] or 0:.1f}s | "
                             f"peak memory {s['mem_mb'] or 0:.0f} MB | fact check {s['checks'][0]}/{s['checks'][1]}"])
    elif k == "badfreq":
        msgs.append(["error", f"{r[1]:g} MHz is outside the RTL-SDR range (24-1766 MHz)."])
    else:
        STATE["queries"] += 1
        job = core.plan(r)
        if job["note"]:
            msgs.append(["alert" if k == "disaster" else "info", job["note"]])
        facts, total = [], 0
        for label, lo, hi, binw in job["scans"]:
            res = core.analyze(label, lo, hi, binw, STATE["mock"])
            if not res["ok"]:
                msgs.append(["error", f"No data from the SDR for {label}. Is it plugged in?"])
                continue
            scans.append(scan_json(res))
            n = len(res["peaks"])
            msgs.append(["info", f"{label}: {n} signal{'s' if n != 1 else ''} above the noise floor"])
            facts.append(res["facts"]); total += n
        if facts and not total:
            reply = {"text": job["empty"], "checked": True, "latency": None, "tps": None, "model": False}
        elif facts:
            with AI_LOCK:
                out = core.explain_full(job["prefix"] + "\n".join(facts))
            STATE["checks"][1] += 1
            if out["checked"]:
                STATE["checks"][0] += 1
            STATE["last"] = {"latency": round(out["latency"], 2), "tps": out["tps"] and round(out["tps"], 1)}
            try:
                core.log_metric(q, out["latency"], out["tps"])
            except Exception:
                pass
            reply = {"text": out["text"], "checked": out["checked"], "latency": out["latency"],
                     "tps": out["tps"], "model": True}
    return {"messages": msgs, "scans": scans, "reply": reply, "stats": stats()}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/stats":
            self.send(200, stats())
        else:
            self.send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/ask":
            return self.send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            q = json.loads(self.rfile.read(n) or b"{}").get("q", "")[:300]
            self.send(200, ask(q))
        except Exception as e:
            self.send(500, {"error": f"{type(e).__name__}: {e}"})


def local_ips():
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=2).stdout.split()
        return [ip for ip in out if ":" not in ip]
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser(description="RFScout web dashboard")
    ap.add_argument("--mock", action="store_true", help="simulated SDR data")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    if not a.mock and not shutil.which("rtl_power"):
        sys.exit("rtl_power not found. Install rtl-sdr, or run with --mock for simulated data.")
    STATE["mock"] = a.mock
    print("RFScout web dashboard" + ("  (SIMULATED SDR)" if a.mock else ""))
    print("  loading local AI model...")
    STATE["load_s"] = core.ensure_server()
    print(f"  model ready ({STATE['load_s']:.1f}s)")
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    print(f"  open  http://localhost:{a.port}")
    for ip in local_ips():
        print(f"  or    http://{ip}:{a.port}   (from a laptop on the same network)")
    print("  Ctrl-C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        if core.server_proc and core.server_proc.poll() is None:
            core.server_proc.terminate()


PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RFScout</title>
<style>
:root{
  --bg:#0a0e12; --panel:#10161c; --line:#1f2a33; --text:#d7e2ea; --dim:#6f8291;
  --accent:#3ee6a8; --accent2:#22b8cf; --warn:#ffb347; --bad:#ff5f6d; --grid:#17212a;
}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--text);
  font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace}
header{display:flex;align-items:center;gap:14px;padding:12px 18px;border-bottom:1px solid var(--line);flex-wrap:wrap}
.logo{font-weight:700;font-size:18px;letter-spacing:.5px}
.logo b{color:var(--accent)}
.sub{color:var(--dim)}
.tags{margin-left:auto;display:flex;gap:8px;flex-wrap:wrap}
.tag{border:1px solid var(--line);padding:3px 9px;border-radius:999px;font-size:12px;color:var(--dim)}
.tag.live{color:var(--accent);border-color:#1f5c48}
.tag.sim{color:#111;background:var(--warn);border-color:var(--warn);font-weight:700}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--accent);margin-right:6px;
  box-shadow:0 0 8px var(--accent);animation:pulse 2s infinite}
@keyframes pulse{50%{opacity:.35}}
main{display:grid;grid-template-columns:1fr 320px;grid-template-rows:minmax(280px,46vh) minmax(0,1fr);grid-template-areas:'spec side' 'chat side';gap:14px;padding:14px 18px;max-width:1500px;margin:0 auto;height:calc(100vh - 56px)}
main>section.panel:first-child{grid-area:spec}
.side{grid-area:side;overflow:auto}
.chatp{grid-area:chat}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px}
.ph{display:flex;align-items:center;gap:10px;padding:10px 14px;border-bottom:1px solid var(--line);
  color:var(--dim);font-size:12px;text-transform:uppercase;letter-spacing:1px}
.ph .t{color:var(--text)}
.tabs{margin-left:auto;display:flex;gap:6px}
.tabs button{background:none;border:1px solid var(--line);color:var(--dim);border-radius:6px;padding:2px 8px;
  font:inherit;font-size:11px;cursor:pointer}
.tabs button.on{color:var(--accent);border-color:var(--accent)}
main>section.panel:first-child{display:flex;flex-direction:column}
#specwrap{position:relative;flex:1;min-height:200px}
canvas{display:block;width:100%;height:100%}
#empty{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:var(--dim)}
.side{display:flex;flex-direction:column;gap:14px}
.stats{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line);border-radius:0 0 10px 10px;overflow:hidden}
.stat{background:var(--panel);padding:10px 14px}
.stat .k{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:1px}
.stat .v{font-size:20px;font-weight:700;margin-top:2px;font-variant-numeric:tabular-nums}
.stat .v small{font-size:12px;color:var(--dim);font-weight:400}
#sigs{list-style:none;margin:0;padding:6px 0;max-height:300px;overflow:auto}
#sigs li{display:grid;grid-template-columns:22px 1fr auto;gap:2px 8px;padding:7px 14px;cursor:default}
#sigs li:hover,#sigs li.hl{background:#152029}
#sigs .n{color:var(--dim)}
#sigs .f{font-weight:700}
#sigs .d{font-weight:700;text-align:right}
#sigs .b{grid-column:2/4;color:var(--dim);font-size:12px}
.chatp{display:flex;flex-direction:column;min-height:0}
#log{flex:1;overflow:auto;padding:12px 16px;min-height:80px}
.m{margin:5px 0;white-space:pre-wrap;word-wrap:break-word}
.m.you{color:var(--accent)} .m.you::before{content:"> "}
.m.info{color:var(--dim)}
.m.error{color:var(--bad)} .m.alert{color:var(--bad);font-weight:700}
.m.warn{color:var(--warn)}
.m.ai{background:#0d1d1a;border:1px solid #1c3b33;border-left:3px solid var(--accent);border-radius:8px;padding:9px 12px}
.m.ai .who{color:var(--accent);font-weight:700;margin-right:6px}
.m.ai .meta{display:block;color:var(--dim);font-size:11px;margin-top:4px}
.badge{display:inline-block;font-size:10px;padding:1px 6px;border-radius:4px;margin-left:6px;vertical-align:1px}
.badge.ok{background:#173b2f;color:var(--accent)} .badge.fb{background:#3d2b12;color:var(--warn)}
form{display:flex;gap:8px;padding:12px 14px;border-top:1px solid var(--line)}
input{flex:1;background:#0b1015;border:1px solid var(--line);color:var(--text);border-radius:8px;padding:10px 12px;font:inherit}
input:focus{outline:none;border-color:var(--accent2)}
button.go{background:var(--accent);color:#04140e;border:0;border-radius:8px;padding:0 18px;font:inherit;font-weight:700;cursor:pointer}
button.go:disabled{opacity:.4;cursor:wait}
.chips{display:flex;gap:6px;flex-wrap:wrap;padding:0 14px 12px}
.chip{background:none;border:1px solid var(--line);color:var(--text);border-radius:999px;padding:4px 11px;font:inherit;font-size:12px;cursor:pointer}
.chip:hover{border-color:var(--accent2);color:var(--accent2)}
.chip.red{border-color:#5a2028;color:var(--bad)}
.busy{display:none;align-items:center;gap:8px;color:var(--warn);padding:0 16px 10px}
.busy.on{display:flex}
.spin{width:12px;height:12px;border:2px solid var(--warn);border-top-color:transparent;border-radius:50%;animation:r 0.8s linear infinite}
@keyframes r{to{transform:rotate(360deg)}}
@media (max-width:900px){main{grid-template-columns:1fr;grid-template-rows:auto;grid-template-areas:'spec' 'chat' 'side';height:auto;padding:12px 16px} #log{max-height:50vh} #specwrap{min-height:260px} .sub{display:none}}
</style></head>
<body>
<header>
  <div class="logo"><b>RF</b>Scout</div>
  <div class="sub">Offline RF spectrum assistant · TI SK-AM62B</div>
  <div class="tags">
    <span id="sdrtag" class="tag live"><span class="dot"></span>LIVE SDR</span>
    <span class="tag">LOCAL AI · NO CLOUD</span>
    <span class="tag" id="modeltag">model</span>
  </div>
</header>
<main>
  <section class="panel">
    <div class="ph"><span>Spectrum</span><span class="t" id="spectitle">no scan yet</span><div class="tabs" id="tabs"></div></div>
    <div id="specwrap"><canvas id="c"></canvas><div id="empty">Run a scan to see the spectrum</div></div>
  </section>
  <aside class="side">
    <section class="panel">
      <div class="ph"><span>Signals detected</span></div>
      <ul id="sigs"><li><span></span><span class="b">None yet</span></li></ul>
    </section>
    <section class="panel">
      <div class="ph"><span>On-device performance</span></div>
      <div class="stats">
        <div class="stat"><div class="k">Model load</div><div class="v" id="s_load">–</div></div>
        <div class="stat"><div class="k">Peak memory</div><div class="v" id="s_mem">–</div></div>
        <div class="stat"><div class="k">Reply time</div><div class="v" id="s_lat">–</div></div>
        <div class="stat"><div class="k">Speed</div><div class="v" id="s_tps">–</div></div>
        <div class="stat" style="grid-column:1/-1"><div class="k">AI fact check passed</div><div class="v" id="s_chk">–</div></div>
      </div>
    </section>
  </aside>
  <section class="panel chatp">
    <div class="ph"><span>Ask RFScout</span></div>
    <div id="log"></div>
    <div class="busy" id="busy"><div class="spin"></div><span id="busytxt">Sweeping the spectrum…</span></div>
    <div class="chips">
      <button class="chip" data-q="scan fm">FM radio</button>
      <button class="chip" data-q="scan noaa">NOAA weather</button>
      <button class="chip" data-q="scan air">Aircraft voice</button>
      <button class="chip" data-q="scan ism433">433 MHz sensors</button>
      <button class="chip" data-q="what's at 162.55">What's at 162.55?</button>
      <button class="chip red" data-q="disaster">Disaster mode</button>
      <button class="chip" data-q="bands">Bands</button>
    </div>
    <form id="f"><input id="q" autocomplete="off" placeholder="Ask: what's active between 88 and 108 MHz?"><button class="go" id="go">Scan</button></form>
  </section>
</main>
<script>
const $=id=>document.getElementById(id);
let scans=[], cur=0, hl=-1;

function msg(role, text, extra){
  const d=document.createElement("div"); d.className="m "+role;
  if(role==="ai"){
    d.innerHTML='<span class="who">RFScout</span>';
    d.appendChild(document.createTextNode(text));
    if(extra) d.insertAdjacentHTML("beforeend", extra);
  } else d.textContent=text;
  $("log").appendChild(d); $("log").scrollTop=1e9;
}

function setStats(s){
  if(!s) return;
  $("modeltag").textContent=s.model;
  if(s.mock){const t=$("sdrtag"); t.className="tag sim"; t.textContent="SIMULATED SDR";}
  $("s_load").innerHTML=s.load_s!=null? s.load_s.toFixed(1)+"<small> s</small>":"–";
  $("s_mem").innerHTML=s.mem_mb? Math.round(s.mem_mb)+"<small> MB</small>":"–";
  $("s_lat").innerHTML=s.last? s.last.latency.toFixed(1)+"<small> s</small>":"–";
  $("s_tps").innerHTML=s.last&&s.last.tps? s.last.tps+"<small> tok/s</small>":"–";
  $("s_chk").innerHTML=s.checks[1]? s.checks[0]+"<small> / "+s.checks[1]+"</small>":"–";
}

function color(db){return db<15?"#3ee6a8":db<25?"#ffb347":"#ff5f6d";}

function draw(){
  const c=$("c"), ctx=c.getContext("2d"), dpr=window.devicePixelRatio||1;
  const W=c.clientWidth, H=c.clientHeight; c.width=W*dpr; c.height=H*dpr; ctx.scale(dpr,dpr);
  ctx.clearRect(0,0,W,H);
  const s=scans[cur]; $("empty").style.display=s?"none":"flex"; if(!s) return;
  const L=46,R=14,T=26,B=30, pw=W-L-R, ph=H-T-B, top=Math.max(30, ...s.peaks.map(p=>p.db+5));
  const x=f=>L+(f-s.lo)/(s.hi-s.lo)*pw, y=d=>T+ph-Math.max(0,Math.min(top,d))/top*ph;
  ctx.font="11px ui-monospace,monospace"; ctx.textBaseline="middle";
  // grid
  ctx.strokeStyle="#17212a"; ctx.fillStyle="#6f8291"; ctx.lineWidth=1;
  for(let d=0; d<=top; d+=10){ctx.beginPath();ctx.moveTo(L,y(d));ctx.lineTo(W-R,y(d));ctx.stroke(); ctx.textAlign="right"; ctx.fillText(d+" dB",L-6,y(d));}
  ctx.textAlign="center";
  for(let i=0;i<=5;i++){const f=s.lo+(s.hi-s.lo)*i/5; ctx.beginPath();ctx.moveTo(x(f),T);ctx.lineTo(x(f),T+ph);ctx.stroke();
    ctx.fillText(+f.toFixed(3)+(i===5?" MHz":""), Math.min(W-R-20,Math.max(L+12,x(f))), H-B+16);}
  // 10 dB detection threshold
  ctx.setLineDash([4,4]); ctx.strokeStyle="#3a4a56"; ctx.beginPath(); ctx.moveTo(L,y(10)); ctx.lineTo(W-R,y(10)); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle="#3a4a56"; ctx.textAlign="left"; ctx.fillText("detection threshold", L+6, y(10)-8);
  // trace
  const g=ctx.createLinearGradient(0,T,0,T+ph); g.addColorStop(0,"rgba(255,95,109,.55)"); g.addColorStop(.45,"rgba(255,179,71,.35)"); g.addColorStop(1,"rgba(62,230,168,.08)");
  ctx.beginPath(); ctx.moveTo(x(s.points[0][0]),y(0));
  for(const [f,d] of s.points) ctx.lineTo(x(f),y(d));
  ctx.lineTo(x(s.points[s.points.length-1][0]),y(0)); ctx.closePath(); ctx.fillStyle=g; ctx.fill();
  ctx.beginPath(); s.points.forEach(([f,d],i)=>i?ctx.lineTo(x(f),y(d)):ctx.moveTo(x(f),y(d)));
  ctx.strokeStyle="#3ee6a8"; ctx.lineWidth=1.4; ctx.stroke();
  // peaks
  s.peaks.forEach((p,i)=>{
    const px=x(p.freq), py=y(p.db), on=i===hl;
    ctx.fillStyle=color(p.db); ctx.beginPath(); ctx.arc(px,py,on?6:4,0,7); ctx.fill();
    ctx.fillStyle="#d7e2ea"; ctx.textAlign="center"; ctx.font=(on?"bold ":"")+"11px ui-monospace,monospace";
    ctx.fillText((i+1)+"  "+p.freq.toFixed(3), Math.min(W-R-40,Math.max(L+40,px)), Math.max(T-12,py-14));
  });
}

function showScan(i){
  cur=i; hl=-1; const s=scans[i];
  $("spectitle").textContent=s? `${s.label} · ${+s.lo.toFixed(3)}–${+s.hi.toFixed(3)} MHz`:"no scan yet";
  [...$("tabs").children].forEach((b,j)=>b.classList.toggle("on",j===i));
  const ul=$("sigs"); ul.innerHTML="";
  if(!s||!s.peaks.length){ul.innerHTML='<li><span></span><span class="b">Nothing above the noise floor</span></li>';}
  else s.peaks.forEach((p,j)=>{
    const li=document.createElement("li");
    li.innerHTML=`<span class="n">${j+1}</span><span class="f">${p.freq.toFixed(3)} MHz</span><span class="d" style="color:${color(p.db)}">+${Math.round(p.db)} dB</span><span class="b"></span>`;
    li.querySelector(".b").textContent=p.band+" — "+p.desc;
    li.onmouseenter=()=>{hl=j;draw()}; li.onmouseleave=()=>{hl=-1;draw()};
    ul.appendChild(li);
  });
  draw();
}

async function ask(q){
  q=q.trim(); if(!q) return;
  msg("you",q); $("q").value=""; $("go").disabled=true; $("busy").classList.add("on");
  const scanLike=!/^(help|\?|band|stat|quit|exit)/i.test(q);
  $("busytxt").textContent=scanLike?"Sweeping the spectrum, then the local AI writes a summary…":"Working…";
  try{
    const r=await fetch("/api/ask",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({q})});
    const j=await r.json();
    if(j.error){msg("error",j.error);return;}
    j.messages.forEach(([role,t])=>msg(role,t));
    if(j.scans.length){
      scans=j.scans; $("tabs").innerHTML="";
      if(scans.length>1) scans.forEach((s,i)=>{const b=document.createElement("button"); b.textContent=s.label; b.onclick=()=>showScan(i); $("tabs").appendChild(b);});
      showScan(scans.length-1);
    }
    if(j.reply){
      const rp=j.reply; let meta="";
      if(rp.model){
        meta=`<span class="meta">${rp.latency.toFixed(1)} s${rp.tps?" · "+rp.tps.toFixed(1)+" tok/s":""} · on-device`+
             (rp.checked?'<span class="badge ok">fact-checked</span>':'<span class="badge fb">AI drifted, verified summary shown</span>')+`</span>`;
      } else meta='<span class="meta">no signals, so no AI call needed</span>';
      msg("ai",rp.text,meta);
    }
    setStats(j.stats);
  }catch(e){msg("error","Lost connection to the board: "+e);}
  finally{$("go").disabled=false; $("busy").classList.remove("on"); $("q").focus();}
}

$("f").onsubmit=e=>{e.preventDefault(); ask($("q").value);};
document.querySelectorAll(".chip").forEach(b=>b.onclick=()=>ask(b.dataset.q));
window.onresize=draw;
fetch("/api/stats").then(r=>r.json()).then(s=>{setStats(s);
  msg("info","Connected to RFScout. The radio, the AI and this page are all served from the board. No internet.");
  if(s.mock) msg("warn","SIMULATED MODE: spectrum data is fake. The AI is still real and local.");
});
</script>
</body></html>
"""

if __name__ == "__main__":
    main()
