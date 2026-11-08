#!/usr/bin/env python3
"""
RFScout terminal dashboard (curses, Python standard library only).

Runs on the board's HDMI console. Uses the same scan + AI code as rfscout.py.

  python3 rfscout_tui.py           # live SDR
  python3 rfscout_tui.py --mock    # simulated SDR data (labeled SIMULATED)
  python3 rfscout_tui.py --ascii   # if the console font can't draw block characters

Views: spectrum (scans), emergency status card, waterfall (watch), signal meter (hunt),
aircraft radar (ADS-B).  Keys: Enter send | Esc stop | Up/Down history | PgUp/PgDn scroll | Ctrl-C quit
"""
import argparse, curses, locale, math, os, queue, re, shutil, sys, textwrap, threading, time
from collections import deque

try:
    import rfscout as core
except ImportError:
    sys.exit("Put rfscout_tui.py in the same folder as rfscout.py and run it from there.")

locale.setlocale(locale.LC_ALL, "")
UTF8 = "utf" in (locale.getpreferredencoding() or "").lower()

DEMO_HINT = ("Try: emergency check | scan fm | watch fm | hunt 162.55 | aircraft | "
             "what frequency do police use? | report | help")
WF_COLS = 240           # resolution kept for each waterfall row
CONTINUOUS = ("watch", "hunt", "adsb")


class App:
    def __init__(self, mock, ascii_only):
        self.mock = mock
        u = UTF8 and not ascii_only
        self.u = u
        self.FULL, self.HALF = ("█", "▄") if u else ("#", ".")
        self.SHADE = (" ", "░", "▒", "▓", "█") if u else (" ", ".", ":", "*", "#")
        self.HL, self.VL, self.DOT = ("─", "│", "·") if u else ("-", "|", ".")
        self.ev = queue.Queue()
        self.chat = deque(maxlen=400)
        self.hist, self.hidx, self.entry = [], None, ""
        self.scroll = 0
        self.view = "none"
        self.scan = None                     # last scan result (spectrum view)
        self.comms = None                    # last comms report
        self.wf = deque(maxlen=200)          # waterfall rows
        self.wf_res = None; self.wf_n = 0
        self.hunt = {"f": None, "hist": deque(maxlen=240), "peak": 0.0}
        self.planes = []; self.adsb_frames = 0
        self.ready = self.busy = False
        self.mode = None                     # continuous mode running
        self.stop_evt = threading.Event()
        self.running = True
        self.status = "Loading local AI model..."
        self.spin = 0
        self.load_s = None
        self.last = {}
        self.checks = [0, 0]
        self.queries = 0
        self.cfg = core.load_cfg()
        self.say("info", "RFScout: offline RF spectrum assistant. Radio, AI and dashboard all run on this board.")
        if mock:
            self.say("warn", "SIMULATED MODE: radio data is fake. The AI is still real and local.")

    # ---------- state ----------
    def say(self, role, text):
        self.chat.append((role, str(text)))
        self.scroll = 0

    def post(self, kind, v=None):
        self.ev.put((kind, v))

    def load_model(self):
        try:
            self.post("ready", core.ensure_server())
        except SystemExit as e:
            self.post("fatal", re.sub(r"\x1b\[[0-9;]*m", "", str(e)) or "model failed to start")
        except Exception as e:
            self.post("fatal", f"model failed to start: {e}")

    def ai(self, q, facts, system=core.SYSTEM, fallback=None):
        self.post("status", "Local AI is writing a summary")
        out = core.explain_full(facts, system, fallback)
        try:
            core.log_metric(q, out["latency"], out["tps"])
        except Exception:
            pass
        self.post("reply", out)

    # ---------- workers (run off the UI thread) ----------
    def run_worker(self, fn, *args):
        def wrap():
            try:
                fn(*args)
            except Exception as e:
                self.post("say", ("error", f"{type(e).__name__}: {e}"))
            finally:
                self.post("done")
        threading.Thread(target=wrap, daemon=True).start()

    def w_scan(self, q, r):
        job = core.plan(r)
        if job["note"]:
            self.post("say", ("info", job["note"]))
        facts, total = [], 0
        for label, lo, hi, binw in job["scans"]:
            self.post("status", f"Sweeping {label} ({lo:g}-{hi:g} MHz)")
            res = core.analyze(label, lo, hi, binw, self.mock)
            if not res["ok"]:
                self.post("say", ("error", f"No data from the SDR for {label}. Is it plugged in?")); continue
            self.post("scan", res)
            facts.append(res["facts"]); total += len(res["peaks"])
        if not facts:
            return
        if not total:
            self.post("say", ("ai", job["empty"])); return
        self.ai(q, job["prefix"] + "\n".join(facts))

    def w_comms(self, q):
        self.post("say", ("alert", "EMERGENCY CHECK: scanning emergency communication channels"))
        self.post("comms", {"items": [], "summary": "", "pending": [c[1] for c in core.COMMS]})
        rep = core.comms_report(self.mock, progress=lambda s: self.post("status", s))
        self.post("comms", rep)
        self.ai(q, rep["facts"], core.SYSTEM_COMMS, rep["summary"])

    def w_ask(self, q):
        self.post("status", "Looking it up in the band table")
        out = core.answer_question(q)
        if out.get("model"):
            try:
                core.log_metric(q, out["latency"], out["tps"])
            except Exception:
                pass
            self.post("reply", out)
        else:
            self.post("say", ("ai", out["text"]))

    def w_watch(self, target):
        label, lo, hi, binw = target
        w = core.Watcher(core.watch_tol(lo, hi))
        self.post("say", ("info", f"Watching {label} ({lo:g}-{hi:g} MHz). New and lost signals are announced. "
                                  "Type 'stop' or press Esc to end."))
        n = 0
        while not self.stop_evt.is_set():
            self.post("status", f"Watching {label}  ·  sweep {n + 1}" if self.u else f"Watching {label} - sweep {n + 1}")
            res = core.analyze(label, lo, hi, binw, self.mock, log=False)
            if not res["ok"]:
                self.post("say", ("error", "No data from the SDR. Watch stopped.")); break
            n += 1
            self.post("wf", (res, core.waterfall_row(res, WF_COLS), n))
            for ev, f, db, band in w.update(res):
                t = time.strftime("%H:%M:%S")
                if ev == "new":
                    self.post("say", ("good", f"{t}  NEW signal  {f:.3f} MHz  +{db:.0f} dB  ({band})"))
                else:
                    self.post("say", ("error", f"{t}  LOST signal {f:.3f} MHz  ({band}) went off the air"))
            if self.mock:
                self.stop_evt.wait(1.0)
        self.post("say", ("info", f"Watch ended after {n} sweeps."))

    def w_hunt(self, mhz):
        name, _ = core.lookup(mhz)
        self.post("say", ("info", f"Hunting {mhz:g} MHz ({name}). Walk around: the level rises as you get closer. "
                                  "Turn so your body blocks the antenna; the direction where the level drops most "
                                  "is AWAY from the transmitter. 'stop' or Esc ends."))
        while not self.stop_evt.is_set():
            v = core.hunt_level(mhz, self.mock)
            if v is None:
                self.post("say", ("error", "No data from the SDR. Hunt stopped.")); break
            self.post("hunt", v)
            if self.mock:
                self.stop_evt.wait(0.4)

    def w_adsb(self):
        if not self.mock and not shutil.which("rtl_adsb"):
            self.post("say", ("error", "rtl_adsb not found (it comes with rtl-sdr).")); return
        tr = core.AdsbTracker(self.cfg)
        self.post("say", ("info", f"Listening for aircraft on 1090 MHz around {self.cfg['place']}. Planes broadcast "
                                  "their own GPS position, so these locations are real. 'stop' or Esc ends."))
        self.post("adsb", ([], 0))
        while not self.stop_evt.is_set():
            self.post("status", f"Listening on 1090 MHz  ·  {len(tr.planes)} aircraft" if self.u else
                      f"Listening on 1090 MHz - {len(tr.planes)} aircraft")
            if not core.adsb_capture(tr, 5, self.mock, self.stop_evt):
                break
            self.post("adsb", (tr.current(), tr.frames))
        self.post("say", ("info", f"Aircraft listening ended. {tr.frames} frames decoded."))

    # ---------- commands ----------
    def submit(self, q):
        q = q.strip()
        if not q:
            return
        self.hist = (self.hist + [q])[-100:]; self.hidx = None
        self.say("you", q)
        r = core.route(q)
        k = r[0]
        if k == "quit":
            self.running = False
        elif k == "stop":
            if self.mode:
                self.stop_evt.set(); self.status = "Stopping..."
            else:
                self.say("info", "Nothing is running.")
        elif k == "clear":
            self.chat.clear()
        elif k == "help":
            self.say("info", core.HELP)
        elif k == "bands":
            self.say("info", "US band table (local, offline):\n" +
                     "\n".join(f"{lo:8.3f}-{hi:8.3f} MHz  {n}" for lo, hi, n, _ in core.BANDS))
        elif k == "stats":
            self.say("info", self.stats_text())
        elif k == "badfreq":
            self.say("error", f"{r[1]:g} MHz is outside the RTL-SDR range (24-1766 MHz).")
        elif k == "hunthelp":
            self.say("ai", core.HUNT_HELP)
        elif k == "setloc":
            self.cfg.update(lat=r[1], lon=r[2], place=f"{r[1]:.3f}, {r[2]:.3f}"); core.save_cfg(self.cfg)
            self.say("info", f"Receiver location set to {r[1]:.4f}, {r[2]:.4f} (used for the aircraft map).")
        elif k == "report":
            text, path = core.make_report()
            self.say("info", text)
            self.say("good", f"Report saved to {path}")
        elif not self.ready:
            self.say("error", "The AI model is still loading.")
        elif self.busy:
            self.say("error", "Still busy. Type 'stop' (or press Esc) to end the current mode first."
                     if self.mode else "Still working on the last request.")
        else:
            self.busy = True; self.queries += 1
            if k in CONTINUOUS:
                self.mode = k; self.stop_evt.clear()
            if k == "disaster":
                self.view = "comms"; self.run_worker(self.w_comms, q)
            elif k == "ask":
                self.run_worker(self.w_ask, q)
            elif k == "watch":
                self.view = "watch"; self.wf.clear(); self.wf_res = None
                self.run_worker(self.w_watch, core.scan_target(r[1]))
            elif k == "hunt":
                self.view = "hunt"; self.hunt = {"f": r[1], "hist": deque(maxlen=240), "peak": 0.0}
                self.run_worker(self.w_hunt, r[1])
            elif k == "adsb":
                self.view = "adsb"; self.planes = []
                self.run_worker(self.w_adsb)
            else:
                self.view = "scan"; self.run_worker(self.w_scan, q, r)

    def stats_text(self):
        mem = core.peak_mem_mb()
        p, t = self.checks
        parts = [f"Model {os.path.basename(core.MODEL)}", f"load {self.load_s or 0:.1f}s"]
        if mem: parts.append(f"peak model memory {mem:.0f} MB")
        if self.last:
            parts.append(f"last reply {self.last['latency']:.1f}s" +
                         (f", {self.last['tps']:.1f} tok/s" if self.last.get("tps") else ""))
        parts.append(f"fact check passed {p}/{t}")
        parts.append(f"queries {self.queries}")
        return " | ".join(parts)

    def drain(self):
        while True:
            try:
                kind, v = self.ev.get_nowait()
            except queue.Empty:
                return
            if kind == "ready":
                self.ready = True; self.load_s = v; self.status = "Ready"
                self.say("info", f"Local AI ready ({v:.1f}s)." if v else "Local AI ready (reused running server).")
                self.say("info", DEMO_HINT)
            elif kind == "fatal":
                self.status = "AI failed to start"; self.say("error", v)
            elif kind == "status":
                self.status = v
            elif kind == "say":
                self.say(*v)
            elif kind == "scan":
                self.scan = v
                n = len(v["peaks"])
                self.say("info", f"{v['label']}: {n} signal{'s' if n != 1 else ''} above the noise floor")
            elif kind == "comms":
                self.comms = v
            elif kind == "wf":
                self.wf_res, row, self.wf_n = v
                self.wf.appendleft(row)
            elif kind == "hunt":
                self.hunt["hist"].append(v)
                self.hunt["peak"] = max(self.hunt["peak"], v)
            elif kind == "adsb":
                self.planes, self.adsb_frames = v
            elif kind == "reply":
                if v.get("latency") is not None:
                    self.last = {"latency": v["latency"], "tps": v["tps"]}
                self.checks[1] += 1
                if v["checked"]:
                    self.checks[0] += 1
                else:
                    self.say("warn", "AI reply failed the fact check. Showing the verified summary instead.")
                self.say("ai", v["text"])
            elif kind == "done":
                self.busy = False; self.mode = None; self.status = "Ready"

    # ---------- drawing helpers ----------
    def put(self, y, x, s, a=0):
        h, w = self.scr.getmaxyx()
        if 0 <= y < h and 0 <= x < w:
            try:
                self.scr.addnstr(y, x, s, max(0, w - x - (1 if y == h - 1 else 0)), a)
            except curses.error:
                pass

    def colors(self):
        self.A = {}
        if curses.has_colors():
            curses.start_color()
            try:
                curses.use_default_colors(); bg = -1
            except curses.error:
                bg = curses.COLOR_BLACK
            for i, fg in enumerate([curses.COLOR_CYAN, curses.COLOR_GREEN, curses.COLOR_YELLOW,
                                    curses.COLOR_RED, curses.COLOR_WHITE, curses.COLOR_MAGENTA,
                                    curses.COLOR_BLUE], 1):
                curses.init_pair(i, fg, bg)
            curses.init_pair(8, curses.COLOR_BLACK, curses.COLOR_CYAN)
            curses.init_pair(9, curses.COLOR_BLACK, curses.COLOR_YELLOW)
            curses.init_pair(10, curses.COLOR_WHITE, curses.COLOR_RED)
            P = curses.color_pair
            self.A = {"title": P(8) | curses.A_BOLD, "sim": P(9) | curses.A_BOLD, "redbg": P(10) | curses.A_BOLD,
                      "cyan": P(1), "green": P(2), "yellow": P(3), "red": P(4), "white": P(5),
                      "mag": P(6), "blue": P(7), "dim": curses.A_DIM, "bold": curses.A_BOLD}

    def a(self, k):
        return self.A.get(k, 0)

    def lvl_color(self, v):
        return (self.a("green") | curses.A_DIM) if v < 6 else self.a("green") if v < 15 else \
            self.a("yellow") if v < 25 else self.a("red")

    def title(self, top, text, right=""):
        self.put(top, 1, text, self.a("cyan") | curses.A_BOLD)
        if right:
            w = self.scr.getmaxyx()[1]
            self.put(top, max(len(text) + 3, self.pw - len(right) - 1), right, self.a("dim"))

    def freq_axis(self, y, x0, pw, lo, hi):
        self.put(y, x0, self.HL * (pw + 1), self.a("dim"))
        span = hi - lo
        for frac in (0, 0.25, 0.5, 0.75, 1.0):
            f = lo + span * frac
            t = f"{f:.3f}".rstrip("0").rstrip(".")
            x = x0 + 1 + int(frac * (pw - 1)) - (len(t) if frac == 1.0 else len(t) // 2 if frac else 0)
            self.put(y + 1, max(x0, x), t, self.a("dim"))

    # ---------- views ----------
    def draw_header(self, w):
        self.put(0, 0, " " * w, self.a("title"))
        self.put(0, 1, "RFScout  ·  Offline RF Spectrum Assistant  ·  TI SK-AM62B" if self.u else
                 "RFScout | Offline RF Spectrum Assistant | TI SK-AM62B", self.a("title"))
        tag = " SIMULATED SDR " if self.mock else " LIVE SDR "
        tag2 = " LOCAL AI · NO CLOUD " if self.u else " LOCAL AI - NO CLOUD "
        self.put(0, w - len(tag) - len(tag2) - 2, tag, self.a("sim") if self.mock else self.a("title"))
        self.put(0, w - len(tag2) - 1, tag2, self.a("title"))
        mem = core.peak_mem_mb()
        p, t = self.checks
        bits = [f"model {os.path.basename(core.MODEL)}",
                f"load {self.load_s:.1f}s" if self.load_s else "load -",
                f"mem {mem:.0f} MB" if mem else "mem -",
                f"reply {self.last['latency']:.1f}s" if self.last else "reply -",
                f"{self.last['tps']:.1f} tok/s" if self.last.get("tps") else "tok/s -",
                f"fact check {p}/{t}"]
        self.put(1, 1, "   ".join(bits), self.a("dim"))
        if self.mode:
            m = {"watch": " WATCHING ", "hunt": " HUNTING ", "adsb": " AIRCRAFT "}[self.mode]
            self.put(1, w - len(m) - 12, m, self.a("redbg"))
            self.put(1, w - 11, "Esc = stop", self.a("dim"))

    def draw_idle(self, top, h, w):
        self.title(top, " RFScout ")
        lines = ["Ask a question at the prompt below. Examples:", "",
                 "emergency check        which emergency channels are on the air here",
                 "scan fm                FM radio stations",
                 "watch fm               keep scanning, announce new / lost signals",
                 "hunt 162.55            signal meter to walk toward a transmitter",
                 "aircraft               live map of planes overhead (ADS-B)",
                 "what frequency do police use?",
                 "report                 save an incident report"]
        for i, l in enumerate(lines):
            self.put(top + 2 + i, 4, l, self.a("dim") if i < 2 else 0)

    def draw_spectrum(self, top, h, w, s=None, title=None):
        s = s or self.scan
        self.title(top, title or (f" SPECTRUM  {s['label']}  {s['lo']:g}-{s['hi']:g} MHz " if s else " SPECTRUM "))
        if not s:
            return
        ph, y0, lab = h - 3, top + 1, 6
        pw = w - lab - 2
        if ph < 3 or pw < 10:
            return
        top_db = 30.0
        for db in (0, 10, 20, 30):
            self.put(y0 + ph - 1 - int(db / top_db * (ph - 1)), 0, f"{db:>3}dB", self.a("dim"))
        cols = core.waterfall_row(s, pw)
        for c, v in enumerate(cols):
            v = max(0.0, min(top_db, v))
            halves = int(round(v / top_db * ph * 2))
            colr = self.lvl_color(v)
            for j in range(halves // 2):
                self.put(y0 + ph - 1 - j, lab + 1 + c, self.FULL, colr)
            if halves % 2:
                self.put(y0 + ph - 1 - halves // 2, lab + 1 + c, self.HALF, colr)
        span = s["hi"] - s["lo"]
        for i, k in enumerate(s["peaks"], 1):
            c = max(0, min(pw - 1, int((k["freq"] - s["lo"]) / span * pw) if span else 0))
            v = max(0.0, min(top_db, k["db"]))
            yy = max(y0, y0 + ph - 1 - int(round(v / top_db * ph)) - 1)
            self.put(yy, lab + 1 + c, str(i), self.a("white") | curses.A_BOLD)
        self.freq_axis(y0 + ph, lab, pw, s["lo"], s["hi"])

    def draw_waterfall(self, top, h, w):
        s = self.wf_res
        self.title(top, f" WATERFALL  {s['label']}  {s['lo']:g}-{s['hi']:g} MHz " if s else " WATERFALL ",
                   f"sweep {self.wf_n} · newest on top" if self.u else f"sweep {self.wf_n} - newest on top")
        if not s:
            self.put(top + 2, 4, "First sweep in progress...", self.a("dim")); return
        lab = 6
        pw = w - lab - 2
        rows = h - 3
        for i, row in enumerate(list(self.wf)[:rows]):
            y = top + 1 + i
            self.put(y, 0, " now" if i == 0 else (f"{-i:>4}" if i % 5 == 0 else ""), self.a("dim"))
            n = len(row)
            for c in range(pw):
                a, b = c * n // pw, max(c * n // pw + 1, (c + 1) * n // pw)
                v = max(row[a:b])
                idx = 0 if v < 6 else 1 if v < 10 else 2 if v < 14 else 3 if v < 20 else 4
                if idx:
                    self.put(y, lab + 1 + c, self.SHADE[idx], self.lvl_color(v))
        self.freq_axis(top + h - 2, lab, pw, s["lo"], s["hi"])

    def draw_comms(self, top, h, w):
        c = self.comms
        self.title(top, " EMERGENCY COMMUNICATIONS CHECK ", time.strftime("%H:%M:%S"))
        if not c:
            return
        row = top + 2
        if c.get("pending") and not c.get("items"):
            for name in c["pending"]:
                self.put(row, 4, ("○ " if self.u else "o ") + name, self.a("dim")); row += 1
            self.put(row + 1, 4, "Scanning...", self.a("yellow"))
            return
        for it in c["items"]:
            if row >= top + h - 2:
                break
            if it["present"]:
                mark, ma, txt = ("●  ON AIR " if self.u else "*  ON AIR "), self.a("green") | curses.A_BOLD, 0
            elif it["present"] is False:
                mark, ma, txt = ("○  none   " if self.u else "o  none   "), self.a("dim"), self.a("dim")
            else:
                mark, ma, txt = "?  error  ", self.a("yellow"), self.a("yellow")
            self.put(row, 3, mark, ma)
            self.put(row, 14, f"{it['label']:<32}", curses.A_BOLD | txt)
            self.put(row, 47, it["detail"][:max(0, w - 49)], txt)
            row += 1
        if c.get("summary"):
            for i, l in enumerate(textwrap.wrap(c["summary"], w - 6)[:max(0, top + h - row - 2)]):
                self.put(row + 1 + i, 3, l, self.a("cyan"))
        self.put(top + h - 1, 3, "Signal on the air = transmitter detected, not proof the service works.", self.a("dim"))

    def draw_hunt(self, top, h, w):
        hh = self.hunt
        f = hh["f"] or 0
        self.title(top, f" HUNT  {f:g} MHz  ({core.lookup(f)[0]}) ")
        hist = list(hh["hist"])
        if not hist:
            self.put(top + 2, 4, "Measuring...", self.a("dim")); return
        v = hist[-1]
        prev = sum(hist[-4:-1]) / max(1, len(hist[-4:-1])) if len(hist) > 1 else v
        trend = ("▲ STRONGER" if self.u else "^ STRONGER") if v > prev + 1.5 else \
                ("▼ weaker" if self.u else "v weaker") if v < prev - 1.5 else ("● steady" if self.u else "= steady")
        tcol = self.a("green") if "STRONG" in trend else self.a("red") if "weak" in trend else self.a("yellow")
        top_db = 40.0
        # big meter
        mw = w - 16
        fill = int(max(0, min(top_db, v)) / top_db * mw)
        pk = int(max(0, min(top_db, hh["peak"])) / top_db * mw)
        for r in (2, 3):
            self.put(top + r, 3, self.SHADE[1] * mw, self.a("dim"))
            self.put(top + r, 3, self.FULL * fill, self.lvl_color(v))
            if pk < mw:
                self.put(top + r, 3 + pk, "|", self.a("white") | curses.A_BOLD)
        self.put(top + 2, mw + 5, f"+{v:4.1f} dB", curses.A_BOLD | self.lvl_color(v))
        self.put(top + 3, mw + 5, trend, tcol | curses.A_BOLD)
        self.put(top + 4, 3, f"0 dB{' ' * (mw - 14)}{top_db:.0f} dB", self.a("dim"))
        self.put(top + 4, max(3, min(mw - 10, 3 + pk - 4)), f"peak {hh['peak']:.0f}", self.a("white"))
        # history chart
        ch = h - 8
        if ch >= 3:
            cw = w - 8
            data = hist[-cw:]
            self.put(top + 6, 1, "history (newest at right)", self.a("dim"))
            for c, val in enumerate(data):
                halves = int(round(max(0, min(top_db, val)) / top_db * ch * 2))
                colr = self.lvl_color(val)
                for j in range(halves // 2):
                    self.put(top + 6 + ch - j, 4 + c, self.FULL, colr)
                if halves % 2:
                    self.put(top + 6 + ch - halves // 2, 4 + c, self.HALF, colr)

    def draw_radar(self, top, h, w):
        cfg = self.cfg
        self.title(top, f" AIRCRAFT RADAR  centered on {cfg['place']} ",
                   f"{len(self.planes)} aircraft · {self.adsb_frames} frames" if self.u else
                   f"{len(self.planes)} aircraft - {self.adsb_frames} frames")
        ph = h - 2
        pw = w
        cy, cx = top + 1 + ph // 2, pw // 2
        R_rows = ph // 2 - 1
        R_cols = min(pw // 2 - 2, R_rows * 2)
        if R_rows < 3:
            return
        far = max([p["dist"] for p in self.planes if "dist" in p] + [5])
        rng = next((r for r in (10, 25, 50, 100, 150, 250) if r >= far * 1.05), 250)
        for frac in (0.5, 1.0):
            for k in range(0, 360, 4):
                t = math.radians(k)
                y = cy - int(round(math.cos(t) * R_rows * frac))
                x = cx + int(round(math.sin(t) * R_cols * frac))
                self.put(y, x, self.DOT, self.a("blue"))
            self.put(cy - int(R_rows * frac), cx + 2, f"{rng * frac:g} mi", self.a("dim"))
        self.put(cy - R_rows - 1 if cy - R_rows - 1 > top else cy - R_rows, cx, "N", self.a("white") | curses.A_BOLD)
        self.put(cy, cx + R_cols + 1, "E", self.a("dim"))
        self.put(cy, cx - R_cols - 1, "W", self.a("dim"))
        self.put(cy + R_rows, cx, "S", self.a("dim"))
        self.put(cy, cx, "+", self.a("green") | curses.A_BOLD)
        self.put(cy + 1, cx - 1, "YOU", self.a("green"))
        for i, p in enumerate([p for p in self.planes if "dist" in p][:9], 1):
            t = math.radians(p["brg"])
            d = p["dist"] / rng
            y = cy - int(round(math.cos(t) * R_rows * d))
            x = cx + int(round(math.sin(t) * R_cols * d))
            low = p.get("alt", 99999) < 3000
            self.put(y, x, str(i), (self.a("red") if low else self.a("yellow")) | curses.A_BOLD)
            self.put(y, x + 1, (p.get("callsign") or p["icao"])[:8], self.a("dim"))
        if not self.planes:
            self.put(cy + 3, max(1, cx - 16), "listening... (antenna near a window helps)", self.a("dim"))

    def draw_plane_list(self, top, h, x, w):
        self.put(top, x, " AIRCRAFT ", self.a("cyan") | curses.A_BOLD)
        row = top + 1
        withpos = [p for p in self.planes if "dist" in p]
        for i, p in enumerate(withpos[:9], 1):
            if row + 1 >= top + h:
                break
            alt = f"{p['alt']:,} ft" if "alt" in p else "? ft"
            self.put(row, x + 1, f"{i}", self.a("white") | curses.A_BOLD)
            self.put(row, x + 3, f"{(p.get('callsign') or p['icao']):<8}", curses.A_BOLD)
            self.put(row, x + 12, alt, self.a("red") if p.get("alt", 99999) < 3000 else 0)
            self.put(row + 1, x + 3, f"{p['dist']:.1f} mi  {p['brg']:.0f}°" if self.u else
                     f"{p['dist']:.1f} mi  {p['brg']:.0f} deg", self.a("dim"))
            if "speed" in p:
                self.put(row + 1, x + 18, f"{p['speed']} kt", self.a("dim"))
            row += 2
        extra = len(self.planes) - len(withpos)
        if extra > 0 and row < top + h:
            self.put(row, x + 1, f"+{extra} heard, no position yet", self.a("dim"))

    def draw_peaks(self, top, h, x, w, s):
        self.put(top, x, " SIGNALS ", self.a("cyan") | curses.A_BOLD)
        if not s:
            return
        if not s["peaks"]:
            self.put(top + 2, x + 1, "Nothing above noise floor", self.a("dim")); return
        row = top + 1
        for i, k in enumerate(s["peaks"], 1):
            if row + 1 >= top + h:
                break
            v = k["db"]
            self.put(row, x + 1, f"{i}", self.a("white") | curses.A_BOLD)
            self.put(row, x + 3, f"{k['freq']:8.3f} MHz", self.a("bold"))
            self.put(row, x + 17, f"+{v:.0f} dB", self.lvl_color(v))
            self.put(row + 1, x + 3, k["band"][:w - 4], self.a("dim"))
            row += 2

    def chat_lines(self, w):
        pre = {"you": ("> ", "green"), "ai": ("RFScout: ", "cyan"), "info": ("", "dim"), "good": ("+ ", "green"),
               "warn": ("! ", "yellow"), "error": ("x ", "red"), "alert": ("!! ", "red")}
        out = []
        for role, text in self.chat:
            p, c = pre.get(role, ("", "dim"))
            first = True
            for para in text.split("\n"):
                for chunk in textwrap.wrap(para, max(10, w - len(p) - 2), break_on_hyphens=False,
                                           drop_whitespace=False) or [""]:
                    out.append(((p if first else " " * len(p)) + chunk.rstrip(), c, role))
                    first = False
        return out

    def draw(self):
        self.scr.erase()
        h, w = self.scr.getmaxyx()
        if h < 20 or w < 60:
            self.put(0, 0, "Make the terminal at least 60x20 (Ctrl-C quits).")
            self.scr.refresh(); return
        self.draw_header(w)
        v = self.view
        side = 30 if w >= 100 and v in ("scan", "watch", "adsb") else 0
        big = v in ("adsb", "comms", "hunt", "watch")
        top_h = max(10, min(28 if big else 18, (h - 8) * (58 if big else 45) // 100))
        self.pw = w - side - 1
        if v == "scan":
            self.draw_spectrum(3, top_h, self.pw)
        elif v == "watch":
            self.draw_waterfall(3, top_h, self.pw)
        elif v == "comms":
            self.draw_comms(3, top_h, w)
        elif v == "hunt":
            self.draw_hunt(3, top_h, w)
        elif v == "adsb":
            self.draw_radar(3, top_h, self.pw)
        else:
            self.draw_idle(3, top_h, w)
        if side:
            for y in range(3, 3 + top_h):
                self.put(y, w - side - 1, self.VL, self.a("dim"))
            if v == "adsb":
                self.draw_plane_list(3, top_h, w - side, side)
            else:
                self.draw_peaks(3, top_h, w - side, side, self.wf_res if v == "watch" else self.scan)
        sep = 3 + top_h
        self.put(sep, 0, self.HL * w, self.a("dim"))
        c_top, c_bot = sep + 1, h - 3
        lines = self.chat_lines(w)
        room = c_bot - c_top + 1
        self.scroll = max(0, min(self.scroll, max(0, len(lines) - room)))
        end = len(lines) - self.scroll
        for i, (t, c, role) in enumerate(lines[max(0, end - room):end]):
            self.put(c_top + i, 1, t, self.a(c) | (curses.A_BOLD if role in ("ai", "alert", "good") else 0))
        if self.scroll:
            self.put(c_top, w - 14, f" ^ {self.scroll} more ", self.a("yellow"))
        spin = "|/-\\"[self.spin % 4]
        busy = self.busy or not self.ready
        st = (f" {spin} " if busy else (" ● " if self.u else " * ")) + self.status
        self.put(h - 2, 0, " " * w, curses.A_REVERSE)
        self.put(h - 2, 0, st, curses.A_REVERSE | (self.a("yellow") if busy else 0))
        keys = "Enter send · Esc stop · ↑↓ history · PgUp/PgDn scroll · Ctrl-C quit" if self.u else \
               "Enter send | Esc stop | Up/Down history | PgUp/PgDn scroll | Ctrl-C quit"
        if len(st) + len(keys) + 3 < w:
            self.put(h - 2, w - len(keys) - 1, keys, curses.A_REVERSE)
        vis = self.entry[-(w - 4):]
        self.put(h - 1, 0, "> ", self.a("green") | curses.A_BOLD)
        self.put(h - 1, 2, vis, curses.A_BOLD)
        try:
            self.scr.move(h - 1, min(w - 2, 2 + len(vis)))
        except curses.error:
            pass
        self.scr.refresh()

    # ---------- input ----------
    def key(self, k):
        if isinstance(k, str):
            if k in "\n\r":
                e, self.entry = self.entry, ""; self.submit(e)
            elif k in ("\x03", "\x04"):
                self.running = False
            elif k == "\x1b":
                if self.mode:
                    self.stop_evt.set(); self.status = "Stopping..."
            elif k in ("\x7f", "\b", "\x08"):
                self.entry = self.entry[:-1]
            elif k == "\x15":
                self.entry = ""
            elif k.isprintable() and len(self.entry) < 300:
                self.entry += k
            return
        if k == curses.KEY_ENTER:
            e, self.entry = self.entry, ""; self.submit(e)
        elif k in (curses.KEY_BACKSPACE, curses.KEY_DC):
            self.entry = self.entry[:-1]
        elif k == curses.KEY_UP and self.hist:
            self.hidx = len(self.hist) - 1 if self.hidx is None else max(0, self.hidx - 1)
            self.entry = self.hist[self.hidx]
        elif k == curses.KEY_DOWN and self.hidx is not None:
            self.hidx += 1
            if self.hidx >= len(self.hist):
                self.hidx, self.entry = None, ""
            else:
                self.entry = self.hist[self.hidx]
        elif k == curses.KEY_PPAGE:
            self.scroll += 5
        elif k == curses.KEY_NPAGE:
            self.scroll = max(0, self.scroll - 5)

    def run(self, scr):
        self.scr = scr
        self.pw = scr.getmaxyx()[1]
        self.colors()
        try:
            curses.curs_set(1)
        except curses.error:
            pass
        try:
            curses.set_escdelay(25)
        except Exception:
            pass
        scr.keypad(True)
        scr.timeout(120)
        threading.Thread(target=self.load_model, daemon=True).start()
        while self.running:
            self.drain()
            self.spin += 1
            self.draw()
            try:
                k = scr.get_wch()
            except curses.error:
                continue
            except KeyboardInterrupt:
                break
            self.key(k)
        self.stop_evt.set()


def main():
    ap = argparse.ArgumentParser(description="RFScout terminal dashboard")
    ap.add_argument("--mock", action="store_true", help="simulated SDR data")
    ap.add_argument("--ascii", action="store_true", help="plain ASCII drawing characters")
    a = ap.parse_args()
    if not a.mock and not shutil.which("rtl_power"):
        sys.exit("rtl_power not found. Install rtl-sdr, or run with --mock for simulated data.")
    if not sys.stdout.isatty():
        sys.exit("Run this from an interactive terminal.")
    app = App(a.mock, a.ascii)
    try:
        curses.wrapper(app.run)
    except KeyboardInterrupt:
        pass
    finally:
        app.stop_evt.set()
        if core.server_proc and core.server_proc.poll() is None:
            core.server_proc.terminate()


if __name__ == "__main__":
    main()
