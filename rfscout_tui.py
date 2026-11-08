#!/usr/bin/env python3
"""
RFScout terminal dashboard (curses, Python standard library only).

Runs on the board's HDMI console. Uses the same scan + AI code as rfscout.py.

  python3 rfscout_tui.py           # live SDR
  python3 rfscout_tui.py --mock    # simulated SDR data (labeled SIMULATED)
  python3 rfscout_tui.py --ascii   # if the console font can't draw block characters

Keys: Enter send | Up/Down history | PgUp/PgDn scroll chat | Ctrl-C quit
"""
import argparse, curses, locale, os, queue, re, shutil, sys, textwrap, threading, time
from collections import deque

try:
    import rfscout as core
except ImportError:
    sys.exit("Put rfscout_tui.py in the same folder as rfscout.py and run it from there.")

locale.setlocale(locale.LC_ALL, "")
UTF8 = "utf" in (locale.getpreferredencoding() or "").lower()

DEMO_HINT = "Try: scan fm | what's at 162.55 | between 88 and 108 MHz | disaster | stats | help"


class App:
    def __init__(self, mock, ascii_only):
        self.mock = mock
        u = UTF8 and not ascii_only
        self.FULL, self.HALF, self.MARK = ("█", "▄", "▼") if u else ("#", ".", "v")
        self.ev = queue.Queue()
        self.chat = deque(maxlen=300)          # (role, text)
        self.hist, self.hidx, self.entry = [], None, ""
        self.scroll = 0
        self.scan = None
        self.ready = self.busy = False
        self.running = True
        self.status = "Loading local AI model..."
        self.spin = 0
        self.load_s = None
        self.last = {}                         # latency, tps
        self.checks = [0, 0]                   # passed, total
        self.queries = 0
        self.say("info", "RFScout: offline RF spectrum assistant. Everything runs on this board.")
        if mock:
            self.say("warn", "SIMULATED MODE: spectrum data is fake. The AI is still real and local.")

    # ---------- state ----------
    def say(self, role, text):
        self.chat.append((role, str(text)))
        self.scroll = 0

    def load_model(self):
        try:
            self.ev.put(("ready", core.ensure_server()))
        except SystemExit as e:
            self.ev.put(("fatal", re.sub(r"\x1b\[[0-9;]*m", "", str(e)) or "model failed to start"))
        except Exception as e:
            self.ev.put(("fatal", f"model failed to start: {e}"))

    def work(self, q, r):
        try:
            job = core.plan(r)
            if job["note"]:
                self.ev.put(("say", ("alert" if r[0] == "disaster" else "info", job["note"])))
            facts, total = [], 0
            for label, lo, hi, binw in job["scans"]:
                self.ev.put(("status", f"Sweeping {label} ({lo:g}-{hi:g} MHz)"))
                res = core.analyze(label, lo, hi, binw, self.mock)
                if not res["ok"]:
                    self.ev.put(("say", ("error", f"No data from the SDR for {label}. Is it plugged in?")))
                    continue
                self.ev.put(("scan", res))
                facts.append(res["facts"]); total += len(res["peaks"])
            if not facts:
                return
            if not total:
                self.ev.put(("say", ("ai", job["empty"])))
                return
            self.ev.put(("status", "Local AI is writing a summary"))
            out = core.explain_full(job["prefix"] + "\n".join(facts))
            try:
                core.log_metric(q, out["latency"], out["tps"])
            except Exception:
                pass
            self.ev.put(("reply", out))
        except Exception as e:
            self.ev.put(("say", ("error", f"{type(e).__name__}: {e}")))
        finally:
            self.ev.put(("done", None))

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
        elif q.lower() == "clear":
            self.chat.clear()
        elif k == "help":
            self.say("info", core.HELP.replace("\n  quit", "\n  clear | quit"))
        elif k == "bands":
            self.say("info", "\n".join(f"{lo:8.3f}-{hi:8.3f} MHz  {n}" for lo, hi, n, _ in core.BANDS))
        elif k == "stats":
            self.say("info", self.stats_text())
        elif k == "badfreq":
            self.say("error", f"{r[1]:g} MHz is outside the RTL-SDR range (24-1766 MHz).")
        elif not self.ready:
            self.say("error", "The AI model is still loading.")
        elif self.busy:
            self.say("error", "Still working on the last request.")
        else:
            self.busy = True; self.queries += 1
            threading.Thread(target=self.work, args=(q, r), daemon=True).start()

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
                self.ready = True; self.load_s = v
                self.status = "Ready"
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
            elif kind == "reply":
                self.last = {"latency": v["latency"], "tps": v["tps"]}
                self.checks[1] += 1
                if v["checked"]:
                    self.checks[0] += 1
                else:
                    self.say("warn", "AI reply failed the fact check. Showing the verified summary instead.")
                self.say("ai", v["text"])
            elif kind == "done":
                self.busy = False; self.status = "Ready"

    # ---------- drawing ----------
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
                                    curses.COLOR_RED, curses.COLOR_WHITE, curses.COLOR_MAGENTA], 1):
                curses.init_pair(i, fg, bg)
            curses.init_pair(7, curses.COLOR_BLACK, curses.COLOR_CYAN)
            curses.init_pair(8, curses.COLOR_BLACK, curses.COLOR_YELLOW)
            P = curses.color_pair
            self.A = {"title": P(7) | curses.A_BOLD, "sim": P(8) | curses.A_BOLD,
                      "cyan": P(1), "green": P(2), "yellow": P(3), "red": P(4), "white": P(5),
                      "mag": P(6), "dim": curses.A_DIM, "bold": curses.A_BOLD}

    def a(self, k):
        return self.A.get(k, 0)

    def draw_header(self, w):
        self.put(0, 0, " " * w, self.a("title"))
        self.put(0, 1, "RFScout  ·  Offline RF Spectrum Assistant  ·  TI SK-AM62B" if UTF8 else
                 "RFScout | Offline RF Spectrum Assistant | TI SK-AM62B", self.a("title"))
        tag = " SIMULATED SDR " if self.mock else " LIVE SDR "
        tag2 = " LOCAL AI · NO CLOUD " if UTF8 else " LOCAL AI - NO CLOUD "
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

    def draw_spectrum(self, top, h, w):
        """Spectrum plot in rows top..top+h-1, columns 0..w-1."""
        s = self.scan
        title = f" SPECTRUM  {s['label']}  {s['lo']:g}-{s['hi']:g} MHz " if s else " SPECTRUM "
        self.put(top, 1, title, self.a("cyan") | curses.A_BOLD)
        if not s:
            self.put(top + h // 2, 4, "No scan yet. " + DEMO_HINT[5:], self.a("dim"))
            return
        ph = h - 3                      # plot rows
        y0 = top + 1                    # first plot row
        lab = 6                         # y-axis label width
        pw = w - lab - 2
        if ph < 3 or pw < 10:
            return
        top_db = 30.0
        for db in (0, 10, 20, 30):
            yy = y0 + ph - 1 - int(db / top_db * (ph - 1))
            self.put(yy, 0, f"{db:>3}dB", self.a("dim"))
        pts, floor = s["points"], s["floor"]
        n = len(pts)
        cols = []
        for c in range(pw):
            a, b = c * n // pw, max(c * n // pw + 1, (c + 1) * n // pw)
            seg = pts[a:min(b, n)]
            cols.append(max(p for _, p in seg) - floor if seg else 0)
        for c, v in enumerate(cols):
            v = max(0.0, min(top_db, v))
            halves = int(round(v / top_db * ph * 2))
            colr = (self.a("green") | curses.A_DIM) if v < 6 else self.a("green") if v < 15 else \
                self.a("yellow") if v < 25 else self.a("red")
            for j in range(halves // 2):
                self.put(y0 + ph - 1 - j, lab + 1 + c, self.FULL, colr)
            if halves % 2:
                self.put(y0 + ph - 1 - halves // 2, lab + 1 + c, self.HALF, colr)
        # peak markers
        span = s["hi"] - s["lo"]
        for i, k in enumerate(s["peaks"], 1):
            c = int((k["freq"] - s["lo"]) / span * pw) if span else 0
            c = max(0, min(pw - 1, c))
            v = max(0.0, min(top_db, k["db"]))
            yy = max(y0, y0 + ph - 1 - int(round(v / top_db * ph)) - 1)
            self.put(yy, lab + 1 + c, str(i), self.a("white") | curses.A_BOLD)
        # x axis
        axis = y0 + ph
        self.put(axis, lab, ("─" if UTF8 else "-") * (pw + 1), self.a("dim"))
        for frac in (0, 0.25, 0.5, 0.75, 1.0):
            f = s["lo"] + span * frac
            t = f"{f:.3f}".rstrip("0").rstrip(".")
            x = lab + 1 + int(frac * (pw - 1)) - (len(t) if frac == 1.0 else len(t) // 2 if frac else 0)
            self.put(axis + 1, max(lab, x), t, self.a("dim"))

    def draw_peaks(self, top, h, x, w):
        self.put(top, x, " SIGNALS ", self.a("cyan") | curses.A_BOLD)
        s = self.scan
        if not s:
            return
        if not s["peaks"]:
            self.put(top + 2, x + 1, "Nothing above noise floor", self.a("dim")); return
        row = top + 1
        for i, k in enumerate(s["peaks"], 1):
            if row + 1 >= top + h:
                break
            v = k["db"]
            colr = self.a("green") if v < 15 else self.a("yellow") if v < 25 else self.a("red")
            self.put(row, x + 1, f"{i}", self.a("white") | curses.A_BOLD)
            self.put(row, x + 3, f"{k['freq']:8.3f} MHz", self.a("bold"))
            self.put(row, x + 17, f"+{v:.0f} dB", colr)
            self.put(row + 1, x + 3, k["band"][:w - 4], self.a("dim"))
            row += 2

    def chat_lines(self, w):
        pre = {"you": ("> ", "green"), "ai": ("RFScout: ", "cyan"), "info": ("", "dim"),
               "warn": ("! ", "yellow"), "error": ("x ", "red"), "alert": ("!! ", "red")}
        out = []
        for role, text in self.chat:
            p, c = pre.get(role, ("", "dim"))
            first = True
            for para in text.split("\n"):
                for chunk in textwrap.wrap(para, max(10, w - len(p) - 2), break_on_hyphens=False) or [""]:
                    out.append(((p if first else " " * len(p)) + chunk, c, role))
                    first = False
        return out

    def draw(self):
        self.scr.erase()
        h, w = self.scr.getmaxyx()
        if h < 20 or w < 60:
            self.put(0, 0, "Make the terminal at least 60x20 (Ctrl-C quits).")
            self.scr.refresh(); return
        self.draw_header(w)
        side = 30 if w >= 100 else 0
        spec_h = max(9, min(18, (h - 8) * 45 // 100))
        self.draw_spectrum(3, spec_h, w - side - 1)
        if side:
            vl = "│" if UTF8 else "|"
            for y in range(3, 3 + spec_h):
                self.put(y, w - side - 1, vl, self.a("dim"))
            self.draw_peaks(3, spec_h, w - side, side)
        sep = 3 + spec_h
        self.put(sep, 0, ("─" if UTF8 else "-") * w, self.a("dim"))
        # chat
        c_top, c_bot = sep + 1, h - 3
        lines = self.chat_lines(w)
        room = c_bot - c_top
        self.scroll = max(0, min(self.scroll, max(0, len(lines) - room)))
        end = len(lines) - self.scroll
        for i, (t, c, role) in enumerate(lines[max(0, end - room):end]):
            self.put(c_top + i, 1, t, self.a(c) | (curses.A_BOLD if role == "ai" else 0))
        if self.scroll:
            self.put(c_top, w - 14, f" ↑ {self.scroll} more " if UTF8 else f" ^ {self.scroll} more ", self.a("yellow"))
        # status + input
        spin = "|/-\\"[self.spin % 4]
        busy = self.busy or not self.ready
        st = (f" {spin} " if busy else " ● " if UTF8 else " * ") + self.status
        self.put(h - 2, 0, " " * w, curses.A_REVERSE)
        self.put(h - 2, 0, st, curses.A_REVERSE | (self.a("yellow") if busy else 0))
        keys = "Enter send · ↑↓ history · PgUp/PgDn scroll · Ctrl-C quit" if UTF8 else \
               "Enter send | Up/Down history | PgUp/PgDn scroll | Ctrl-C quit"
        self.put(h - 2, max(len(st) + 2, w - len(keys) - 1), keys, curses.A_REVERSE)
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
        self.colors()
        try:
            curses.curs_set(1)
        except curses.error:
            pass
        scr.keypad(True)
        scr.timeout(150)
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
        if core.server_proc and core.server_proc.poll() is None:
            core.server_proc.terminate()


if __name__ == "__main__":
    main()
