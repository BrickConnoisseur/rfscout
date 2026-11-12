#!/usr/bin/env python3
"""
RFScout dashboard: a full-screen terminal UI for the monitor attached to the SK-AM62B.

  ┌ title bar ─────────────────────────────── OFFLINE ┐
  │ live spectrum plot          │ system panel         │
  │ (last sweep, peaks labeled) │ model, memory, speed │
  ├ conversation ──────────────────────────────────────┤
  │ questions, tool output, model answers              │
  └ RFScout > input line ──────────────────────────────┘

Uses only Python's built-in curses. Every command goes through the same
router and tools as rfscout.py, so the behavior is identical.

Usage:
  python3 dashboard.py           # live SDR
  python3 dashboard.py --mock    # fake sweep data
  python3 dashboard.py --ascii   # if the console font shows boxes or diamonds
"""
import argparse, curses, io, locale, os, re, shutil, statistics, sys, textwrap, threading, time
from contextlib import redirect_stdout, redirect_stderr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rfscout as rf

locale.setlocale(locale.LC_ALL, "")
UTF8 = "utf" in (locale.getpreferredencoding(False) or "").lower() and "--ascii" not in sys.argv
BLOCKS = " ▁▂▃▄▅▆▇█" if UTF8 else " ..::||##"
FULL = BLOCKS[-1]
DOT = "·" if UTF8 else "."
SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏" if UTF8 else "|/-\\"
ANSI = re.compile(r"\x1b\[[0-9;]*m")

# color pair ids
TITLE, OK, WARN, BAD, CYAN, DIM, PEAK, BORDER, USER = range(1, 10)

def init_colors():
    curses.start_color()
    try: curses.use_default_colors(); bg = -1
    except curses.error: bg = curses.COLOR_BLACK
    curses.init_pair(TITLE, curses.COLOR_BLACK, curses.COLOR_CYAN)
    curses.init_pair(OK, curses.COLOR_GREEN, bg)
    curses.init_pair(WARN, curses.COLOR_YELLOW, bg)
    curses.init_pair(BAD, curses.COLOR_RED, bg)
    curses.init_pair(CYAN, curses.COLOR_CYAN, bg)
    curses.init_pair(DIM, curses.COLOR_WHITE, bg)
    curses.init_pair(PEAK, curses.COLOR_YELLOW, bg)
    curses.init_pair(BORDER, curses.COLOR_BLUE, bg)
    curses.init_pair(USER, curses.COLOR_GREEN, bg)

def put(win, y, x, text, attr=0):
    """addstr that never crashes on the edge of the screen."""
    h, w = win.getmaxyx()
    if y < 0 or y >= h or x >= w: return
    text = text[: max(0, w - x - (1 if y == h - 1 else 0))]
    try: win.addstr(y, x, text, attr)
    except curses.error: pass

def box(win, y, x, h, w, title=""):
    a = curses.color_pair(BORDER)
    try:
        win.attron(a)
        win.hline(y, x + 1, curses.ACS_HLINE, w - 2); win.hline(y + h - 1, x + 1, curses.ACS_HLINE, w - 2)
        win.vline(y + 1, x, curses.ACS_VLINE, h - 2); win.vline(y + 1, x + w - 1, curses.ACS_VLINE, h - 2)
        win.addch(y, x, curses.ACS_ULCORNER); win.addch(y, x + w - 1, curses.ACS_URCORNER)
        win.addch(y + h - 1, x, curses.ACS_LLCORNER)
        try: win.addch(y + h - 1, x + w - 1, curses.ACS_LRCORNER)
        except curses.error: pass
        win.attroff(a)
    except curses.error:
        pass
    if title: put(win, y, x + 2, f" {title} ", curses.color_pair(CYAN) | curses.A_BOLD)

class Dashboard:
    def __init__(self, scr, mock):
        self.scr, self.mock = scr, mock
        self.notes = rf.load_knowledge()
        self.log = []                      # (text, attr)
        self.buf, self.hist, self.hpos = "", [], 0
        self.busy, self.busy_label, self.spin = False, "", 0
        self.worker, self.capture, self.consumed = None, None, 0
        self.status, self.status_t = {}, 0
        self.scroll = 0

    # ---------- log ----------
    def add(self, text, attr=0):
        for line in text.split("\n"):
            self.log.append((line, attr))
        self.log = self.log[-500:]
        self.scroll = 0

    def add_output(self, raw):
        for line in raw.split("\n"):
            if not line.strip(): continue
            attr = 0
            if "\x1b[96m" in line or line.startswith("RFScout:"): attr = curses.color_pair(CYAN) | curses.A_BOLD
            elif "\x1b[91m" in line: attr = curses.color_pair(BAD)
            elif "\x1b[93m" in line: attr = curses.color_pair(WARN)
            elif "\x1b[2m" in line: attr = curses.color_pair(DIM) | curses.A_DIM
            elif "\x1b[1m" in line: attr = curses.A_BOLD
            clean = ANSI.sub("", line)
            if "█" in clean and not UTF8: clean = clean.replace("█", "#")
            self.add(clean, attr)

    # ---------- background work ----------
    def run_bg(self, label, fn):
        self.busy, self.busy_label = True, label
        self.capture, self.consumed = io.StringIO(), 0
        def job():
            try:
                with redirect_stdout(self.capture), redirect_stderr(self.capture):
                    fn()
            except SystemExit as e:
                self.capture.write(f"\x1b[91m{e}\x1b[0m\n")
            except Exception as e:
                self.capture.write(f"\x1b[91m  error: {e}\x1b[0m\n")
            finally:
                self.busy = False
        self.worker = threading.Thread(target=job, daemon=True)
        self.worker.start()

    def drain(self):
        if self.capture is None: return
        data = self.capture.getvalue()
        if len(data) > self.consumed:
            new = data[self.consumed:]
            cut = new.rfind("\n") + 1 if self.busy else len(new)
            if cut:
                self.add_output(new[:cut]); self.consumed += cut

    # ---------- status ----------
    def refresh_status(self):
        if time.time() - self.status_t < 2: return
        self.status_t = time.time()
        lat, tps = rf.STATE["latencies"], rf.STATE["tps"]
        pm = rf.peak_mem_mb() if rf.server_proc else None
        t = rf.cpu_temp_c()
        self.status = {
            "net": rf.is_offline(),
            "model": rf.MODEL_ID.replace("-Instruct", ""),
            "quant": rf.quant_from_name(os.path.realpath(rf.MODEL)) if os.path.exists(rf.MODEL) else "?",
            "load": f"{rf.STATE['load_s']:.1f} s" if rf.STATE["load_s"] else "loading",
            "mem": f"{pm:.0f} MB" if pm else "-",
            "free": f"{rf.mem_avail_mb() or 0:.0f} MB",
            "lat": f"{statistics.mean(lat):.1f} s" if lat else "-",
            "last": f"{lat[-1]:.1f} s" if lat else "-",
            "tps": f"{statistics.mean(tps):.1f} tok/s" if tps else "-",
            "n": str(len(lat)),
            "temp": f"{t:.0f} °C" if t else "-",
            "up": rf.uptime_str(),
            "sdr": "mock data" if self.mock else ("attached" if rf.sdr_present() else "NOT FOUND"),
            "gain": f"{rf.SDR_GAIN} dB",
        }

    # ---------- drawing ----------
    def draw(self):
        s = self.scr
        s.erase()
        H, W = s.getmaxyx()
        if H < 18 or W < 60:
            put(s, 0, 0, "Make the terminal larger (min 60x18)"); s.refresh(); return
        self.refresh_status()

        # title bar
        put(s, 0, 0, " " * W, curses.color_pair(TITLE))
        put(s, 0, 1, "RFScout", curses.color_pair(TITLE) | curses.A_BOLD)
        put(s, 0, 9, "· Offline Edge AI Radio Assistant · TI SK-AM62B", curses.color_pair(TITLE))
        net = self.status.get("net")
        tag = " ● OFFLINE " if net else (" ● ONLINE " if net is False else " ? NET ")
        if not UTF8: tag = tag.replace("●", "*")
        put(s, 0, W - len(tag) - 1, tag,
            (curses.color_pair(OK) if net else curses.color_pair(BAD)) | curses.A_REVERSE | curses.A_BOLD)

        side = 30 if W >= 90 else (26 if W >= 72 else 0)
        top_h = max(9, int((H - 3) * 0.45))
        spec_w = W - side
        self.draw_spectrum(1, 0, top_h, spec_w)
        if side: self.draw_side(1, spec_w, top_h, side)
        log_h = H - top_h - 3
        self.draw_log(1 + top_h, 0, log_h, W)
        self.draw_input(H - 2, W)
        s.refresh()

    def draw_spectrum(self, y, x, h, w):
        pts = rf.LAST.get("pts") or []
        label = (rf.LAST.get("label") or "").replace("_", " ").strip()
        title = "SPECTRUM"
        if pts: title += f" · {label} · {pts[0][0]:.3f}–{pts[-1][0]:.3f} MHz"
        box(self.scr, y, x, h, w, title)
        ph, pw = h - 4, w - 9          # plot area
        px, py = x + 7, y + 1
        if not pts:
            msg = "no sweep yet  ·  try:  scan fm   what's at 162.55   measure 433.92"
            put(self.scr, y + h // 2, x + max(2, (w - len(msg)) // 2), msg, curses.color_pair(DIM) | curses.A_DIM)
            return
        # bin to columns (max hold)
        n = len(pts)
        cols = []
        for c in range(pw):
            a, b = c * n // pw, max(c * n // pw + 1, (c + 1) * n // pw)
            seg = pts[a:b]
            cols.append(max(seg, key=lambda t: t[1]))
        powers = [p for _, p in pts]
        floor = statistics.median(powers)
        top = max(powers)
        lo, hi = floor - 4, max(top + 2, floor + 15)
        thr = floor + 10
        def row_of(p): return (p - lo) / (hi - lo) * ph
        # y axis labels
        for r, val in ((0, hi), (ph // 2, (hi + lo) / 2), (ph - 1, lo)):
            put(self.scr, py + r, x + 1, f"{val:5.0f}", curses.color_pair(DIM) | curses.A_DIM)
        # threshold line
        tr = ph - 1 - int(row_of(thr))
        if 0 <= tr < ph:
            put(self.scr, py + tr, px, DOT * pw, curses.color_pair(BORDER))
        # bars
        for c, (f, p) in enumerate(cols):
            hgt = max(0.0, row_of(p))
            full, frac = int(hgt), hgt - int(hgt)
            attr = curses.color_pair(PEAK) | curses.A_BOLD if p >= thr else curses.color_pair(OK)
            for r in range(min(full, ph)):
                put(self.scr, py + ph - 1 - r, px + c, FULL, attr)
            if full < ph and frac > 0.125:
                put(self.scr, py + ph - 1 - full, px + c, BLOCKS[int(frac * 8)], attr)
        # peak labels
        peaks, _ = rf.find_peaks(pts, top=4)
        used = []
        for f, p in sorted(peaks, key=lambda t: -t[1]):
            c = int((f - pts[0][0]) / max(1e-9, pts[-1][0] - pts[0][0]) * (pw - 1))
            lab = f"{f:.3f}" if pts[-1][0] - pts[0][0] < 2 else f"{f:.1f}"
            lx = min(max(px, px + c - len(lab) // 2), px + pw - len(lab))
            if any(abs(lx - u) < len(lab) + 1 for u in used): continue
            used.append(lx)
            r = ph - 1 - min(ph - 1, int(row_of(p))) - 1
            put(self.scr, py + max(0, r), lx, lab, curses.color_pair(PEAK) | curses.A_BOLD)
        # x axis
        ay = py + ph
        put(self.scr, ay, px, ("─" if UTF8 else "-") * pw, curses.color_pair(BORDER))
        f0, f1 = pts[0][0], pts[-1][0]
        for k in range(5):
            c = int(k * (pw - 1) / 4)
            fv = f0 + (f1 - f0) * k / 4
            lab = f"{fv:.3f}" if f1 - f0 < 2 else f"{fv:.1f}"
            lx = min(max(px, px + c - len(lab) // 2), px + pw - len(lab))
            put(self.scr, ay + 1, lx, lab, curses.color_pair(DIM))
        put(self.scr, ay + 1, x + 1, "MHz", curses.color_pair(DIM) | curses.A_DIM)
        note = f" noise floor {floor:.0f} dB {DOT} threshold +10 dB "
        put(self.scr, y + h - 1, x + max(2, w - len(note) - 2), note, curses.color_pair(DIM))

    def draw_side(self, y, x, h, w):
        box(self.scr, y, x, h, w, "SYSTEM")
        st = self.status
        rows = [
            ("Model", st.get("model", ""), CYAN),
            ("Quant", st.get("quant", ""), 0),
            ("Load", st.get("load", ""), 0),
            ("Peak mem", st.get("mem", ""), 0),
            ("Latency", f"{st.get('lat', '')} avg", 0),
            ("Speed", st.get("tps", ""), 0),
            ("SDR", st.get("sdr", ""), BAD if st.get("sdr") == "NOT FOUND" else OK),
            ("Gain", st.get("gain", ""), 0),
            ("CPU temp", st.get("temp", ""), 0),
            ("Answers", st.get("n", ""), 0),
            ("Free RAM", st.get("free", ""), 0),
            ("Runtime", "llama.cpp CPU", 0),
            ("Uptime", st.get("up", ""), 0),
        ]
        for i, (k, v, c) in enumerate(rows[: h - 2]):
            put(self.scr, y + 1 + i, x + 2, f"{k:<9}", curses.color_pair(DIM) | curses.A_DIM)
            put(self.scr, y + 1 + i, x + 12, str(v)[: w - 14], curses.color_pair(c) if c else 0)

    def draw_log(self, y, x, h, w):
        box(self.scr, y, x, h, w, "CONVERSATION")
        inner_w, inner_h = w - 4, h - 2
        lines = []
        for text, attr in self.log:
            wrapped = textwrap.wrap(text, inner_w, subsequent_indent="  ", drop_whitespace=False) or [""]
            lines.extend((l, attr) for l in wrapped)
        end = len(lines) - self.scroll
        view = lines[max(0, end - inner_h): end]
        for i, (l, attr) in enumerate(view):
            put(self.scr, y + 1 + i, x + 2, l, attr)
        if self.scroll:
            put(self.scr, y, x + w - 16, f" ↑ {self.scroll} more " if UTF8 else f" ^ {self.scroll} more ", curses.color_pair(WARN))

    def draw_input(self, y, W):
        if self.busy:
            sp = SPIN[self.spin % len(SPIN)]
            put(self.scr, y, 1, f"{sp} {self.busy_label}", curses.color_pair(WARN) | curses.A_BOLD)
        else:
            put(self.scr, y, 1, "RFScout > ", curses.color_pair(OK) | curses.A_BOLD)
            put(self.scr, y, 11, self.buf[-(W - 13):])
        hint = "help · PgUp/PgDn scroll · quit"
        put(self.scr, y + 1, max(1, W - len(hint) - 2), hint, curses.color_pair(DIM) | curses.A_DIM)
        if not self.busy:
            try: self.scr.move(y, min(W - 2, 11 + len(self.buf)))
            except curses.error: pass

    # ---------- input ----------
    def ask_yes_no(self, prompt):
        self.add(f"  {prompt} [y/N]", curses.color_pair(WARN) | curses.A_BOLD)
        self.draw()
        self.scr.timeout(-1)
        try:
            k = self.scr.get_wch()
        finally:
            self.scr.timeout(150)
        ok = isinstance(k, str) and k.lower() == "y"
        self.add("  y" if ok else "  n", curses.color_pair(DIM))
        return ok

    def submit(self, q):
        q = q.strip()
        if not q: return True
        self.hist.append(q); self.hpos = len(self.hist)
        self.add("")
        self.add(f"> {q}", curses.color_pair(USER) | curses.A_BOLD)
        r = rf.route(q)
        if r[0] == "quit": return False
        if r[0] == "clear": self.log = []; return True
        if r[0] == "gain":
            g = max(0, min(49, r[1]))
            ans = self.ask_yes_no(f"Change SDR gain {rf.SDR_GAIN} -> {g} dB?")
            rf.confirm = lambda prompt, a=ans: a
        labels = {"preset": "sweeping + thinking", "freq": "sweeping + thinking", "disaster": "emergency sweep",
                  "measure": "measuring", "occupancy": "5 sweeps", "selftest": "running self-test",
                  "bench": "benchmarking", "general": "thinking", "evidence": "hashing model"}
        self.run_bg(labels.get(r[0], "working"), lambda: rf.dispatch(r, q, self.mock, self.notes))
        return True

    def key(self, k):
        if k in (curses.KEY_ENTER, "\n", "\r", 10, 13):
            q, self.buf = self.buf, ""
            return self.submit(q)
        if k in (curses.KEY_BACKSPACE, "\b", "\x7f", 127, 8):
            self.buf = self.buf[:-1]
        elif k == curses.KEY_UP and self.hist:
            self.hpos = max(0, self.hpos - 1); self.buf = self.hist[self.hpos]
        elif k == curses.KEY_DOWN and self.hist:
            self.hpos = min(len(self.hist), self.hpos + 1)
            self.buf = self.hist[self.hpos] if self.hpos < len(self.hist) else ""
        elif k == curses.KEY_PPAGE: self.scroll += 5
        elif k == curses.KEY_NPAGE: self.scroll = max(0, self.scroll - 5)
        elif k == "\x1b": self.buf = ""
        elif k == "\x0c": self.scr.clear()            # Ctrl+L
        elif isinstance(k, str) and k.isprintable():
            self.buf += k
        return True

    # ---------- main loop ----------
    def run(self):
        curses.curs_set(1)
        self.scr.keypad(True)
        self.scr.timeout(150)
        self.add("RFScout dashboard. Everything below runs on this board, with no network.", curses.A_BOLD)
        self.add("Try:  scan fm   ·   what's at 162.55   ·   measure 433.92   ·   disaster   ·   status",
                 curses.color_pair(DIM) | curses.A_DIM)
        if self.mock: self.add("(mock SDR data)", curses.color_pair(WARN))
        def load():
            print(f"  loading {rf.MODEL_ID}...")
            rf.STATE["load_s"] = rf.start_server()
            print(f"\x1b[2m  model ready in {rf.STATE['load_s']:.1f}s, {rf.peak_mem_mb() or 0:.0f} MB\x1b[0m")
        self.run_bg("loading model", load)
        alive = True
        while alive:
            self.drain()
            if not self.busy and rf.server_proc is None and rf.STATE["load_s"] is None and self.worker and not self.worker.is_alive():
                self.drain(); self.draw()
                self.scr.timeout(-1); self.scr.get_wch(); break      # model failed to load
            self.spin += 1
            self.draw()
            try: k = self.scr.get_wch()
            except curses.error: continue
            except KeyboardInterrupt: break
            if k == curses.KEY_RESIZE: self.scr.clear(); continue
            if self.busy: continue                               # ignore typing while working
            alive = self.key(k)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="use fake sweep data")
    ap.add_argument("--ascii", action="store_true", help="plain ASCII characters if the console font shows boxes")
    a = ap.parse_args()
    mock = a.mock or not shutil.which("rtl_power")
    os.environ.setdefault("ESCDELAY", "25")
    _route = rf.route
    rf.route = lambda q: ("clear",) if q.strip().lower() in ("clear", "cls") else _route(q)
    try:
        curses.wrapper(lambda scr: (init_colors(), Dashboard(scr, mock).run()))
    finally:
        if rf.server_proc: rf.server_proc.terminate()

if __name__ == "__main__":
    main()
