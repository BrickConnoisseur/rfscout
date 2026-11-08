#!/usr/bin/env python3
"""
RFScout - offline RF spectrum assistant for the TI SK-AM62B.

Flow:  question -> intent router -> rtl_power sweep -> peak detection
       -> band lookup (local table) -> small language model explains it
       -> fact check (model output must match the measured facts).

Everything runs on the board. No internet, no cloud APIs.

Usage:
  python3 rfscout.py           # live SDR, plain command line
  python3 rfscout.py --mock    # fake sweep data (test the AI side without the SDR)
  python3 rfscout_tui.py       # full-screen dashboard (recommended for the demo)
"""
import argparse, csv, json, math, os, random, re, shutil, subprocess, sys, threading, time, urllib.request

HOME = os.path.expanduser("~/rfscout")
MODEL = os.path.join(HOME, "models", "model.gguf")
SERVER_BIN = os.path.join(HOME, "llama.cpp", "build", "bin", "llama-server")
LOG_DIR = os.path.join(HOME, "logs")
CONFIG = os.path.join(HOME, "config.json")
PORT = 8080
URL = f"http://127.0.0.1:{PORT}"

GAIN = "40"          # rtl_power gain in dB; try 49.6 for more sensitivity
ADSB_GAIN = "49.6"   # aircraft are weak, use max gain
FLOOR_PCT = 0.20     # noise floor = 20th percentile (median is inflated in crowded bands)
MARGIN_DB = 10.0     # how far above the floor a peak must be

# ---------- colors (CLI only) ----------
C = {"g": "\033[92m", "y": "\033[93m", "c": "\033[96m", "r": "\033[91m", "d": "\033[2m", "x": "\033[0m"}
def col(s, k): return f"{C[k]}{s}{C['x']}"

# ---------- config (receiver location for the aircraft map) ----------
DEFAULT_CFG = {"lat": 32.9857, "lon": -96.7502, "place": "UT Dallas"}

def load_cfg():
    try:
        with open(CONFIG) as f:
            return {**DEFAULT_CFG, **json.load(f)}
    except Exception:
        return dict(DEFAULT_CFG)

def save_cfg(cfg):
    os.makedirs(HOME, exist_ok=True)
    with open(CONFIG, "w") as f:
        json.dump(cfg, f)

# ---------- local band table (US), most specific first ----------
BANDS = [
    (162.390, 162.560, "NOAA Weather Radio", "continuous weather forecasts and emergency alerts from the National Weather Service"),
    (433.050, 434.790, "433 MHz ISM band", "low-power devices like wireless weather sensors, remote controls, and doorbells"),
    (314.000, 316.000, "315 MHz short-range devices", "US car key fobs, garage door openers, and tire pressure sensors"),
    (462.550, 467.725, "FRS/GMRS", "handheld walkie-talkies used by families, outdoor groups, and people during emergencies"),
    (1089.0, 1091.0, "ADS-B", "aircraft broadcasting their position, altitude, and speed"),
    (1575.0, 1576.0, "GPS L1", "GPS satellite navigation signals"),
    (121.45, 121.55, "Aviation emergency", "the international aircraft distress and emergency voice channel"),
    (88.0, 108.0, "FM broadcast radio", "commercial and public FM radio stations, which also relay emergency alerts"),
    (108.0, 118.0, "Aviation navigation", "VOR and ILS beacons that guide aircraft"),
    (118.0, 137.0, "Aviation voice", "pilots talking to air traffic control"),
    (137.0, 138.0, "Weather satellites", "NOAA and Meteor satellites sending weather images"),
    (144.0, 148.0, "2 m amateur radio", "licensed ham radio operators, who often run emergency nets in disasters"),
    (148.0, 150.8, "Federal/military VHF", "government and military radio and satellite uplinks"),
    (156.0, 162.0, "Marine VHF", "boats and harbor communications"),
    (150.8, 174.0, "VHF land mobile", "public safety, business, and utility radios"),
    (420.0, 450.0, "70 cm amateur radio", "licensed ham radio operators"),
    (450.0, 470.0, "UHF land mobile", "business, public safety, and utility radios"),
    (470.0, 608.0, "UHF TV", "over-the-air digital television"),
    (729.0, 768.0, "700 MHz cell towers", "LTE cell tower downlink signals"),
    (698.0, 806.0, "700 MHz cellular", "LTE cell phone networks and FirstNet public safety broadband"),
    (806.0, 824.0, "800 MHz public safety (mobile)", "police, fire, and EMS radios transmitting to towers"),
    (851.0, 869.0, "800 MHz public safety (towers)", "police, fire, and EMS trunked radio system towers"),
    (869.0, 894.0, "850 MHz cell towers", "cell tower downlink signals"),
    (824.0, 849.0, "850 MHz cell phones", "cell phones transmitting to towers"),
    (902.0, 928.0, "915 MHz ISM band", "LoRa sensors, smart meters, and other IoT devices"),
    (929.0, 932.0, "Paging", "pager networks, still used by hospitals and emergency services"),
]

def lookup(mhz):
    for lo, hi, name, desc in BANDS:
        if lo <= mhz <= hi:
            return name, desc
    return "Unknown band", "no entry in the local band table"

# ---------- scan presets ----------
PRESETS = {
    "fm":    (88.0, 108.0, "50k",  "FM broadcast"),
    "noaa":  (162.35, 162.60, "5k", "NOAA Weather Radio"),
    "air":   (118.0, 137.0, "25k", "Aviation voice"),
    "ism433":(433.0, 434.9, "10k", "433 MHz ISM"),
    "fob":   (314.5, 315.5, "5k",  "315 MHz key fobs"),
    "frs":   (462.5, 467.8, "12.5k", "FRS/GMRS"),
    "ism915":(902.0, 928.0, "100k", "915 MHz ISM"),
    "cell":  (869.0, 894.0, "100k", "850 MHz cell towers"),
    "safety":(851.0, 869.0, "25k", "800 MHz public safety"),
}

# ---------- SDR ----------
def sweep(lo, hi, binw, mock=False):
    """Return list of (freq_MHz, power_dB). Never falls back to fake data silently."""
    if mock:
        return mock_sweep(lo, hi)
    for attempt in range(3):              # first sweep after startup sometimes returns nothing
        pts = _sweep_once(lo, hi, binw)
        if pts: return pts
        time.sleep(1)
    return []

def _sweep_once(lo, hi, binw):
    out = "/tmp/rfscout_sweep.csv"
    if os.path.exists(out): os.remove(out)
    cmd = ["rtl_power", "-f", f"{lo}M:{hi}M:{binw}", "-i", "1", "-g", GAIN, "-1", out]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
    except subprocess.TimeoutExpired:
        return []
    pts = []
    if not os.path.exists(out): return pts
    with open(out) as f:
        for row in csv.reader(f):
            if len(row) < 7: continue
            f_lo, step = float(row[2]), float(row[4])
            for i, v in enumerate(row[6:]):
                try: pts.append(((f_lo + i * step) / 1e6, float(v)))
                except ValueError: pass
    pts.sort()
    return pts

MOCK_T0 = time.time()

def mock_sweep(lo, hi):
    """Fake spectrum. Some signals come and go over time so watch mode has something to show."""
    rnd = random.Random()
    n = 200
    pts = [(lo + (hi - lo) * i / n, -60 + rnd.uniform(-3, 3)) for i in range(n + 1)]
    t = time.time() - MOCK_T0
    fake = [162.400, 162.550, 94.5, 98.7, 102.1, 433.92, 315.0, 121.5]
    if int(t / 12) % 2: fake.append(462.5625)          # walkie-talkie keys up every other 12 s
    if int(t / 20) % 3 != 2: fake.append(105.3)         # an FM station that drops out sometimes
    sharp = 4000 if hi - lo < 2 else 400
    for fc in fake:
        if lo <= fc <= hi:
            pts = [(f, p + max(0, 25 - abs(f - fc) * sharp)) for f, p in pts]
    for a, b in ((870, 880), (885, 891), (731, 746), (853, 856)):   # wideband cell/trunk carriers
        pts = [(f, p + (16 if a <= f <= b else 0)) for f, p in pts]
    return pts

def find_peaks(pts, margin=MARGIN_DB, top=5, min_sep=0.05):
    if len(pts) < 3: return [], None
    powers = sorted(p for _, p in pts)
    floor = powers[int(len(powers) * FLOOR_PCT)]
    peaks = []
    for i in range(1, len(pts) - 1):
        f, p = pts[i]
        if p > floor + margin and p >= pts[i - 1][1] and p >= pts[i + 1][1]:
            if not peaks or abs(f - peaks[-1][0]) > min_sep:
                peaks.append((f, p))
            elif p > peaks[-1][1]:
                peaks[-1] = (f, p)
    peaks.sort(key=lambda x: -x[1])
    return peaks[:top], floor

def occupancy(pts):
    """For wideband signals (cell towers): fraction of the band well above the quietest part."""
    if len(pts) < 10: return 0.0, None
    powers = sorted(p for _, p in pts)
    quiet = powers[max(0, int(len(powers) * 0.02))]
    busy = sum(1 for p in powers if p > quiet + MARGIN_DB)
    return busy / len(powers), quiet

# ---------- SLM ----------
server_proc = None

def start_server():
    global server_proc
    if not os.path.exists(SERVER_BIN): sys.exit(col(f"llama-server not found at {SERVER_BIN}", "r"))
    if not os.path.exists(MODEL): sys.exit(col(f"model not found at {MODEL}", "r"))
    t0 = time.time()
    server_proc = subprocess.Popen(
        [SERVER_BIN, "-m", MODEL, "-c", "1024", "-t", "4", "--host", "127.0.0.1", "--port", str(PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(240):
        try:
            if urllib.request.urlopen(URL + "/health", timeout=1).status == 200:
                return time.time() - t0
        except Exception:
            pass
        time.sleep(0.5)
    sys.exit(col("model server did not start", "r"))

def ensure_server():
    """Reuse a running llama-server if there is one, else start it. Returns load seconds (0 if reused)."""
    try:
        if urllib.request.urlopen(URL + "/health", timeout=1).status == 200:
            return 0.0
    except Exception:
        pass
    return start_server()

def peak_mem_mb():
    try:
        with open(f"/proc/{server_proc.pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM"):
                    return int(line.split()[1]) / 1024
    except Exception:
        return None

SYSTEM = ("You are RFScout, a radio assistant running offline on a small device. "
          "Write ONE short sentence for a non-expert saying how many signals were found "
          "and what the strongest one is used for. Use MHz. Copy numbers exactly from the facts. "
          "Do not invent anything.")

SYSTEM_COMMS = ("You are RFScout, an offline radio assistant used after a disaster. "
                "In ONE or TWO short sentences, tell a non-expert which communication channels have "
                "signals on the air and which do not. A signal on the air does not prove a service works. "
                "Copy numbers exactly. Do not claim any message was decoded. Do not invent anything.")

SYSTEM_ASK = ("You are RFScout, an offline radio assistant. Answer the question in ONE or TWO short "
              "sentences using ONLY the reference facts. Use MHz and copy numbers exactly. "
              "If the facts do not answer it, say you only know the local US band table.")

def dedupe(text):
    """Safety net for small models: drop repeated lines/sentences."""
    seen, out = set(), []
    for line in re.split(r"(?<=[.!?])\s+|\n+", text):
        key = line.strip().lower()
        if key and key not in seen:
            seen.add(key); out.append(line.strip())
    return " ".join(out)

BAD_WORDS = ("ghz", "khz", "decod", " am ", "am/fm", "amplitude", "interfer", "amateur")

def grounded(text, facts):
    """Reject replies that use wrong units, claim decoding, or cite numbers not in the facts."""
    t = " " + text.lower() + " "
    f = facts.lower()
    if not text.strip(): return False
    if any(w in t and w not in f for w in BAD_WORDS): return False
    fact_nums = {float(x) for x in re.findall(r"\d+(?:\.\d+)?", facts)}
    for x in re.findall(r"\d+(?:\.\d+)?", text):
        v = float(x)
        if v < 10: continue                       # small counts like "2 signals" are fine
        if not any(abs(v - n) <= 0.05 for n in fact_nums): return False
    return True

def template(facts):
    """Deterministic summary built straight from the facts (used if the model drifts)."""
    sigs = re.findall(r"- ([\d.]+) MHz, (\d+) dB above noise: ([^(]+)", facts)
    if not sigs: return "No signals stood out in this scan."
    f, db, name = sigs[0]
    return (f"Found {len(sigs)} signal{'s' if len(sigs) > 1 else ''}. "
            f"Strongest is {float(f):.1f} MHz, {db} dB above the noise floor ({name.strip()}).")

def explain_full(facts, system=SYSTEM, fallback=None):
    """Ask the local model. Returns dict(text, latency, tps, checked) - checked=False means fallback used."""
    body = json.dumps({
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": facts}],
        "max_tokens": 120, "temperature": 0.1, "repeat_penalty": 1.3}).encode()
    t0 = time.time()
    req = urllib.request.Request(URL + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=180).read())
    dt = time.time() - t0
    text = dedupe(r["choices"][0]["message"]["content"].strip())
    tps = r.get("timings", {}).get("predicted_per_second")
    ok = grounded(text, facts)
    fb = fallback if fallback is not None else template(facts)
    return {"text": text if ok else fb, "latency": dt, "tps": tps, "checked": ok, "raw": text}

def explain(facts):
    res = explain_full(facts)
    if not res["checked"]:
        print(col("  [AI reply failed fact check, using verified summary]", "y"))
    return res["text"], res["latency"], res["tps"]

# ---------- logging ----------
def log_metric(query, latency, tps):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, "metrics.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["time", "query", "latency_s", "tokens_per_s", "peak_mem_mb"])
        w.writerow([time.strftime("%H:%M:%S"), query, f"{latency:.2f}", tps and f"{tps:.1f}", peak_mem_mb()])

def log_event(kind, **data):
    """Append one record to logs/scans.jsonl (used by the incident report)."""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(os.path.join(LOG_DIR, "scans.jsonl"), "a") as f:
            f.write(json.dumps({"t": time.time(), "kind": kind, **data}) + "\n")
    except Exception:
        pass

# ---------- scanning ----------
SDR_LOCK = threading.Lock()   # one SDR program at a time

def analyze(label, lo, hi, binw, mock, log=True):
    """Sweep + detect. Returns a dict the UIs can draw. ok=False means the SDR gave no data."""
    with SDR_LOCK:
        pts = sweep(lo, hi, binw, mock)
    min_sep = 0.15 if hi - lo > 5 else 0.02     # FM stations are ~200 kHz wide
    peaks, floor = find_peaks(pts, min_sep=min_sep)
    res = {"label": label, "lo": lo, "hi": hi, "points": pts, "floor": floor, "ok": floor is not None,
           "peaks": [{"freq": f, "db": p - floor, "band": lookup(f)[0], "desc": lookup(f)[1]} for f, p in peaks]}
    if not res["ok"]:
        res["facts"] = None
    elif not peaks:
        res["facts"] = f"Scan of {label} ({lo:g}-{hi:g} MHz): no signals above the noise floor."
    else:
        res["facts"] = (f"Scan of {label} ({lo:g}-{hi:g} MHz) found these signals:\n" +
                        "\n".join(f"- {k['freq']:.3f} MHz, {k['db']:.0f} dB above noise: {k['band']} ({k['desc']})"
                                  for k in res["peaks"]))
    if log and res["ok"]:
        log_event("scan", label=label, lo=lo, hi=hi,
                  peaks=[[round(k["freq"], 3), round(k["db"], 1)] for k in res["peaks"]])
    return res

def binwidth_for(lo, hi):
    """~400 bins across the span, at least 5 kHz."""
    k = max(5.0, (hi - lo) * 1000 / 400)
    return f"{k:g}k" if k < 1000 else f"{k/1000:g}M"

def plan(r):
    """Turn a routed scan command into scan jobs. Returns dict(scans, prefix, empty, note) or None."""
    if r[0] == "preset":
        lo, hi, binw, label = PRESETS[r[1]]
        return {"scans": [(label, lo, hi, binw)], "prefix": "", "note": None,
                "empty": f"No signals stood out in {label} ({lo:g}-{hi:g} MHz)."}
    if r[0] == "range":
        lo, hi = r[1], r[2]
        return {"scans": [(f"{lo:g}-{hi:g} MHz", lo, hi, binwidth_for(lo, hi))], "prefix": "", "note": None,
                "empty": f"No signals stood out between {lo:g} and {hi:g} MHz."}
    if r[0] == "freq":
        mhz = r[1]; name, desc = lookup(mhz)
        return {"scans": [(name, max(24.0, mhz - 0.5), min(1766.0, mhz + 0.5), "5k")],
                "prefix": f"Frequency of interest: {mhz:g} MHz, in the {name} band ({desc}).\n",
                "note": f"{mhz:g} MHz is in: {name} — {desc}",
                "empty": f"Nothing active near {mhz:g} MHz right now. That frequency is in the {name} band, used for {desc}."}
    return None

def scan_target(r):
    """(label, lo, hi, binw) for watch mode from a routed preset/range/freq."""
    p = plan(r)
    return p["scans"][0] if p else None

# ---------- communications status report (disaster mode) ----------
# (key, label, lo, hi, bin width, kind)   kind: "peaks" = narrow signals, "wide" = cell-style carriers
COMMS = [
    ("noaa",   "NOAA Weather Radio",           162.35, 162.60, "5k",    "peaks"),
    ("fm",     "FM broadcast (EAS relay)",     88.0,   108.0,  "50k",   "peaks"),
    ("cell7",  "Cell towers 700 MHz",          729.0,  768.0,  "100k",  "wide"),
    ("cell8",  "Cell towers 850 MHz",          869.0,  894.0,  "100k",  "wide"),
    ("safety", "Police/fire/EMS 800 MHz",      851.0,  869.0,  "25k",   "wide"),
    ("frs",    "Walkie-talkies (FRS/GMRS)",    462.5,  467.8,  "12.5k", "peaks"),
    ("ham",    "Ham radio 2 m (emergency nets)", 144.0, 148.0, "12.5k", "peaks"),
]

def comms_report(mock, progress=None):
    """Scan every emergency-relevant channel. Returns dict(items, scans, facts, card)."""
    items, scans = [], []
    for key, label, lo, hi, binw, kind in COMMS:
        if progress: progress(f"Checking {label} ({lo:g}-{hi:g} MHz)")
        res = analyze(label, lo, hi, binw, mock, log=False)
        if not res["ok"]:
            items.append({"key": key, "label": label, "present": None, "detail": "no data from SDR"}); continue
        scans.append(res)
        if kind == "wide":
            occ, _ = occupancy(res["points"])
            present = occ >= 0.10
            detail = f"carrier energy across {occ*100:.0f}% of band" if present else "no carrier energy detected"
        else:
            present = bool(res["peaks"])
            if present:
                fs = ", ".join(f"{k['freq']:.3f}" for k in res["peaks"][:3])
                detail = f"{len(res['peaks'])} signal{'s' if len(res['peaks']) > 1 else ''} ({fs} MHz)"
            else:
                detail = "nothing on the air during the scan"
        items.append({"key": key, "label": label, "present": present, "detail": detail})
    on = [i for i in items if i["present"]]
    off = [i for i in items if i["present"] is False]
    facts = ("Communications check after a disaster.\nOn the air:\n" +
             ("\n".join(f"- {i['label']}: {i['detail']}" for i in on) or "- none") +
             "\nNot detected:\n" + ("\n".join(f"- {i['label']}" for i in off) or "- none"))
    card = "\n".join(("[ON ] " if i["present"] else "[ - ] " if i["present"] is False else "[ ? ] ") +
                     f"{i['label']}: {i['detail']}" for i in items)
    summary = (f"{len(on)} of {len(items)} channels have signals on the air"
               + (f": {', '.join(i['label'] for i in on)}." if on else "."))
    if not any(i["key"] in ("cell7", "cell8") and i["present"] for i in items):
        summary += " No cell tower signal was detected, so phones may not work here."
    log_event("comms", items=[{k: i[k] for k in ("key", "label", "present", "detail")} for i in items])
    return {"items": items, "scans": scans, "facts": facts, "card": card, "summary": summary}

# ---------- watch mode (change detection) ----------
class Watcher:
    """Tracks which signals are present between sweeps and reports NEW / LOST with hysteresis."""
    def __init__(self, tol):
        self.tol = tol
        self.active = {}          # freq -> {"db", "seen", "miss"}
        self.sweeps = 0

    def update(self, res):
        events = []
        self.sweeps += 1
        pts, floor = res["points"], res["floor"]
        def level_at(f):
            near = [p for x, p in pts if abs(x - f) <= self.tol]
            return (max(near) - floor) if near else 0.0
        now = [k for k in res["peaks"] if k["db"] >= MARGIN_DB + 3]
        for k in now:
            match = next((f for f in self.active if abs(f - k["freq"]) <= self.tol), None)
            if match is None:
                self.active[k["freq"]] = {"db": k["db"], "seen": time.time(), "miss": 0}
                if self.sweeps > 1:          # first sweep is the baseline, not news
                    events.append(("new", k["freq"], k["db"], k["band"]))
            else:
                self.active[match].update(db=k["db"], miss=0)
        for f in list(self.active):
            if level_at(f) < MARGIN_DB - 3:
                self.active[f]["miss"] += 1
                if self.active[f]["miss"] >= 2:      # gone for 2 sweeps in a row
                    events.append(("lost", f, self.active[f]["db"], lookup(f)[0]))
                    del self.active[f]
        for e in events:
            log_event("watch", event=e[0], freq=round(e[1], 3), db=round(e[2], 1), band=e[3], label=res["label"])
        return events

def watch_tol(lo, hi):
    """Same spacing analyze() uses to merge peaks, so a jittery peak isn't reported as new."""
    return 0.15 if hi - lo > 5 else 0.02

def waterfall_row(res, cols):
    """Collapse one sweep into `cols` values of dB above floor (for waterfall/spectrum drawing)."""
    pts, floor = res["points"], res["floor"]
    n = len(pts); out = []
    for c in range(cols):
        a, b = c * n // cols, max(c * n // cols + 1, (c + 1) * n // cols)
        seg = pts[a:min(b, n)]
        out.append(max(p for _, p in seg) - floor if seg else 0.0)
    return out

# ---------- hunt mode (find a transmitter by walking) ----------
def hunt_level(mhz, mock, _state={"x": 0.0}):
    """Signal strength (dB above floor) at one frequency. Walk around and watch it change."""
    if mock:
        _state["x"] += 0.15
        return max(0.0, 18 + 10 * math.sin(_state["x"]) + random.uniform(-2, 2))
    lo, hi = max(24.0, mhz - 0.2), min(1766.0, mhz + 0.2)
    with SDR_LOCK:
        pts = sweep(lo, hi, "2k", False)
    if not pts: return None
    powers = sorted(p for _, p in pts)
    floor = powers[int(len(powers) * FLOOR_PCT)]
    near = [p for f, p in pts if abs(f - mhz) <= 0.0125]
    return (max(near) - floor) if near else 0.0

# ---------- ADS-B aircraft decoding (real GPS positions broadcast by planes) ----------
CHARSET = "#ABCDEFGHIJKLMNOPQRSTUVWXYZ##### ###############0123456789######"

def adsb_crc(h):
    data, nbits, gen = int(h, 16), len(h) * 4, 0x1FFF409
    for i in range(nbits - 24):
        if data & (1 << (nbits - 1 - i)):
            data ^= gen << (nbits - 25 - i)
    return data & 0xFFFFFF

def _nl(lat):
    if abs(lat) >= 87: return 2 if abs(lat) == 87 else 1
    if lat == 0: return 59
    a = 1 - math.cos(math.pi / 30)
    b = math.cos(math.pi / 180 * abs(lat)) ** 2
    return int(math.floor(2 * math.pi / math.acos(1 - a / b)))

def cpr_local(lat_cpr, lon_cpr, odd, ref_lat, ref_lon):
    """Decode one position frame using the receiver location as reference (valid within ~180 NM)."""
    dlat = 360.0 / (59 if odd else 60)
    yl = lat_cpr / 131072.0
    j = math.floor(ref_lat / dlat) + math.floor((ref_lat % dlat) / dlat - yl + 0.5)
    lat = dlat * (j + yl)
    nl = _nl(lat) - (1 if odd else 0)
    dlon = 360.0 / max(nl, 1)
    xl = lon_cpr / 131072.0
    m = math.floor(ref_lon / dlon) + math.floor((ref_lon % dlon) / dlon - xl + 0.5)
    lon = dlon * (m + xl)
    return lat, lon

def decode_adsb(h, ref_lat, ref_lon):
    """Decode one 112-bit hex frame. Returns dict with icao and whatever fields it carried, or None."""
    h = h.strip().strip("*;").upper()
    if len(h) != 28 or not re.fullmatch(r"[0-9A-F]{28}", h): return None
    df = int(h[0:2], 16) >> 3
    if df not in (17, 18) or adsb_crc(h) != 0: return None
    icao = h[2:8]
    me = bin(int(h[8:22], 16))[2:].zfill(56)
    tc = int(me[0:5], 2)
    out = {"icao": icao}
    if 1 <= tc <= 4:
        cs = "".join(CHARSET[int(me[8 + 6 * i:14 + 6 * i], 2)] for i in range(8))
        out["callsign"] = cs.replace("#", "").strip()
    elif 9 <= tc <= 18:
        ab = me[8:20]
        if ab[7] == "1":
            n = int(ab[:7] + ab[8:], 2)
            out["alt"] = n * 25 - 1000
        odd = me[21] == "1"
        lat, lon = cpr_local(int(me[22:39], 2), int(me[39:56], 2), odd, ref_lat, ref_lon)
        out["lat"], out["lon"] = lat, lon
    elif tc == 19 and int(me[5:8], 2) in (1, 2):
        vew = int(me[14:24], 2) - 1; vns = int(me[25:35], 2) - 1
        if me[13] == "1": vew = -vew
        if me[24] == "1": vns = -vns
        out["speed"] = round(math.hypot(vew, vns))
        out["heading"] = round(math.degrees(math.atan2(vew, vns)) % 360)
    return out

def dist_bearing(lat1, lon1, lat2, lon2):
    """Miles and compass bearing from point 1 to point 2."""
    r = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    d = 2 * r * math.asin(math.sqrt(a))
    brg = math.degrees(math.atan2(math.sin(dl) * math.cos(p2),
                                  math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl))) % 360
    return d, brg

class AdsbTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.planes = {}      # icao -> dict
        self.frames = 0

    def feed(self, line):
        d = decode_adsb(line, self.cfg["lat"], self.cfg["lon"])
        if not d: return
        self.frames += 1
        p = self.planes.setdefault(d["icao"], {"icao": d["icao"]})
        p.update(d); p["seen"] = time.time()
        if "lat" in d:
            dist, brg = dist_bearing(self.cfg["lat"], self.cfg["lon"], d["lat"], d["lon"])
            if dist > 250:                    # bad decode, ignore
                for k in ("lat", "lon"): p.pop(k, None)
            else:
                p["dist"], p["brg"] = dist, brg

    def current(self, max_age=60):
        now = time.time()
        for k in [k for k, p in self.planes.items() if now - p["seen"] > max_age]:
            del self.planes[k]
        return sorted(self.planes.values(), key=lambda p: p.get("dist", 999))

def adsb_capture(tracker, seconds, mock, stop=None):
    """Listen on 1090 MHz for `seconds` (or until stop is set) and feed frames into tracker."""
    if mock:
        _mock_adsb(tracker, seconds, stop); return True
    if not shutil.which("rtl_adsb"):
        return False
    with SDR_LOCK:
        proc = subprocess.Popen(["rtl_adsb", "-g", ADSB_GAIN], stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True, bufsize=1)
        end = time.time() + seconds
        def reader():
            for line in proc.stdout:
                tracker.feed(line)
        th = threading.Thread(target=reader, daemon=True); th.start()
        try:
            while time.time() < end and proc.poll() is None and not (stop and stop.is_set()):
                time.sleep(0.2)
        finally:
            proc.terminate()
            try: proc.wait(timeout=3)
            except Exception: proc.kill()
    log_event("adsb", planes=len(tracker.planes), frames=tracker.frames)
    return True

_MOCK_PLANES = [("A1B2C3", "AAL1234", 34000, 12, 40, 250), ("A4D5E6", "SWA88", 9000, 6, 200, 120),
                ("A7F8A9", "N911HT", 1200, 3, 310, 90), ("AB12CD", "UAL57", 37000, 38, 95, 270)]

def _mock_adsb(tracker, seconds, stop):
    cfg = tracker.cfg
    t = time.time() - MOCK_T0
    for icao, cs, alt, dist, brg, hdg in _MOCK_PLANES:
        b = math.radians(brg + t * 0.8)
        lat = cfg["lat"] + dist / 69.0 * math.cos(b)
        lon = cfg["lon"] + dist / (69.0 * math.cos(math.radians(cfg["lat"]))) * math.sin(b)
        d, br = dist_bearing(cfg["lat"], cfg["lon"], lat, lon)
        tracker.planes[icao] = {"icao": icao, "callsign": cs, "alt": alt, "lat": lat, "lon": lon,
                                "dist": d, "brg": br, "speed": 140 + alt // 200, "heading": hdg, "seen": time.time()}
        tracker.frames += 7
    end = time.time() + min(seconds, 3)
    while time.time() < end and not (stop and stop.is_set()):
        time.sleep(0.2)

# ---------- incident report ----------
def make_report(hours=12):
    """Summarize logs/scans.jsonl into a text report and save it. Returns (text, path)."""
    path = os.path.join(LOG_DIR, "scans.jsonl")
    since = time.time() - hours * 3600
    recs = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r["t"] >= since: recs.append(r)
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    ts = lambda t: time.strftime("%H:%M:%S", time.localtime(t))
    cfg = load_cfg()
    lines = [f"RFScout incident report  {time.strftime('%Y-%m-%d %H:%M')}",
             f"Location: {cfg['place']} ({cfg['lat']:.4f}, {cfg['lon']:.4f})   Window: last {hours} h",
             f"Records: {len(recs)}", ""]
    if not recs:
        lines.append("No scans logged yet. Run some scans, then 'report' again.")
    comms = [r for r in recs if r["kind"] == "comms"]
    if comms:
        last = comms[-1]
        lines.append(f"LATEST COMMUNICATIONS CHECK ({ts(last['t'])})")
        for i in last["items"]:
            mark = "ON " if i["present"] else " - " if i["present"] is False else " ? "
            lines.append(f"  [{mark}] {i['label']}: {i['detail']}")
        if len(comms) > 1:
            first = {i["key"]: i["present"] for i in comms[0]["items"]}
            changed = [i["label"] + (" came ON" if i["present"] else " went OFF")
                       for i in last["items"] if first.get(i["key"]) != i["present"]]
            lines.append(f"  Changes since {ts(comms[0]['t'])}: " + (", ".join(changed) or "none"))
        lines.append("")
    watch = [r for r in recs if r["kind"] == "watch"]
    if watch:
        lines.append(f"SIGNAL CHANGES SEEN IN WATCH MODE ({len(watch)})")
        for r in watch[-15:]:
            lines.append(f"  {ts(r['t'])}  {'NEW ' if r['event'] == 'new' else 'LOST'}  "
                         f"{r['freq']:.3f} MHz  {r['band']}")
        lines.append("")
    scans = [r for r in recs if r["kind"] == "scan"]
    if scans:
        lines.append(f"BANDS SCANNED ({len(scans)} scans)")
        by = {}
        for r in scans:
            b = by.setdefault(r["label"], {"n": 0, "first": r["t"], "last": r["t"], "sig": {}})
            b["n"] += 1; b["last"] = r["t"]
            for f, db in r["peaks"]:
                key = round(f * 20) / 20
                b["sig"][key] = max(db, b["sig"].get(key, 0))
        for label, b in by.items():
            top = sorted(b["sig"].items(), key=lambda x: -x[1])[:4]
            lines.append(f"  {label}: {b['n']} scan(s) {ts(b['first'])}-{ts(b['last'])}, "
                         f"{len(b['sig'])} distinct signal(s)" +
                         (": " + ", ".join(f"{f:.2f} MHz" for f, _ in top) if top else ""))
        lines.append("")
    adsb = [r for r in recs if r["kind"] == "adsb"]
    if adsb:
        lines.append(f"AIRCRAFT: up to {max(r['planes'] for r in adsb)} tracked at once "
                     f"({sum(r['frames'] for r in adsb)} position/ID frames decoded)")
        lines.append("")
    lines.append("Note: RFScout detects signal energy. A signal on the air does not prove a service works,")
    lines.append("and no messages are decoded except aircraft ADS-B broadcasts.")
    text = "\n".join(lines)
    os.makedirs(LOG_DIR, exist_ok=True)
    out = os.path.join(LOG_DIR, f"report_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    with open(out, "w") as f:
        f.write(text + "\n")
    return text, out

# ---------- open-ended questions, answered from the local band table ----------
ALIASES = {
    "weather": "NOAA", "noaa": "NOAA", "storm": "NOAA", "tornado": "NOAA", "hurricane": "NOAA",
    "walkie": "FRS", "walkie-talkie": "FRS", "gmrs": "FRS", "frs": "FRS",
    "police": "public safety", "fire": "public safety", "ems": "public safety", "ambulance": "public safety",
    "cop": "public safety", "first responder": "public safety", "responders": "public safety",
    "phone": "cell", "cell": "cell", "lte": "cell", "mobile": "cell",
    "plane": "Aviation", "airplane": "Aviation", "pilot": "Aviation", "aircraft": "ADS-B",
    "ham": "amateur", "amateur": "amateur", "boat": "Marine", "ship": "Marine",
    "car": "key fobs", "garage": "key fobs", "tire": "key fobs", "doorbell": "433", "sensor": "433",
    "lora": "915", "iot": "915", "smart meter": "915", "tv": "TV", "television": "TV", "gps": "GPS",
    "pager": "Paging", "radio station": "FM broadcast", "music": "FM broadcast", "satellite": "satellites",
    "distress": "Aviation emergency", "mayday": "Aviation emergency",
}
STOP = set("a an the is are was what which where who how do does can i me my any there near here of in on "
           "for to and or it this that radio frequency frequencies channel channels band bands mhz use used "
           "about tell explain show find give".split())

def ask_facts(q):
    """Pick the band-table rows most related to a free-form question."""
    ql = q.lower()
    hints = [v.lower() for k, v in ALIASES.items() if re.search(rf"\b{re.escape(k)}\b", ql)]
    words = [w for w in re.findall(r"[a-z0-9/]+", ql) if w not in STOP and len(w) > 2]
    scored = []
    for lo, hi, name, desc in BANDS:
        text = f"{name} {desc}".lower()
        s = sum(3 for h in hints if h in text) + sum(1 for w in words if w in text)
        if s: scored.append((s, lo, hi, name, desc))
    scored.sort(key=lambda x: -x[0])
    return [x[1:] for x in scored[:3]]

def answer_question(q, use_model=True):
    """Free-form question -> answer grounded in the band table. Returns dict(text, checked, model, latency, tps)."""
    rows = ask_facts(q)
    if not rows:
        return {"text": "I only know the local US band table and what the radio can measure. "
                        "Try 'bands', a frequency like 'what's at 162.55', or 'emergency check'.",
                "checked": True, "model": False, "latency": None, "tps": None}
    facts_lines = [f"- {name}: {lo:g} to {hi:g} MHz, used for {desc}." for lo, hi, name, desc in rows]
    fallback = "From the local band table: " + " ".join(l[2:] for l in facts_lines)
    if not use_model:
        return {"text": fallback, "checked": True, "model": False, "latency": None, "tps": None}
    out = explain_full(f"Question: {q}\nReference facts:\n" + "\n".join(facts_lines),
                       system=SYSTEM_ASK, fallback=fallback)
    out["model"] = True
    return out

# ---------- intent router ----------
HELP = """Ask things like:
  what's at 162.55            scan around one frequency
  between 88 and 108 MHz      scan any range from 24 to 1766 MHz
  scan fm | noaa | air | frs | cell | safety | ism433 | ism915 | fob
  emergency check             disaster mode: which emergency channels are on the air
  watch fm  (or watch 462.5 to 467.8)   keep scanning, alert on new/lost signals
  hunt 462.5625               signal-strength meter to find a transmitter by walking
  aircraft                    map of planes overhead (decoded ADS-B GPS positions)
  report                      save an incident report from everything logged
  what frequency do police use?   open questions, answered from the US band table
  bands | stats | location 32.98 -96.75 | stop | clear | quit"""

NUM = r"(\d{2,4}(?:\.\d+)?)\s*(?:mhz|m)?"
RANGE_RE = re.compile(NUM + r"\s*(?:-|–|to|and|through|:)\s*" + NUM)
QUESTION_RE = re.compile(r"^(what|which|where|who|how|why|is|are|can|does|do|tell|explain|list)\b")

def preset_match(q):
    for key in PRESETS:
        pat = re.sub(r"([a-z]+)(\d+)", r"\1\\s*\2", key)   # ism433 -> ism\s*433
        if re.search(rf"\b{pat}\b", q):
            return key
    return None

def has(q, *words):
    return any(re.search(rf"\b{re.escape(w)}", q) for w in words)

def route_scan(q):
    """Preset / range / single-frequency part of the router. Returns a route tuple or None."""
    key = preset_match(q)
    if key: return ("preset", key)
    m = RANGE_RE.search(q)
    if m:
        lo, hi = sorted((float(m.group(1)), float(m.group(2))))
        if lo < 24 or hi > 1766: return ("badfreq", lo if lo < 24 else hi)
        if hi - lo < 0.1: return ("freq", lo)
        return ("range", lo, hi)
    m = re.search(r"(?<![\d.])(\d{2,4}(?:\.\d+)?)(?![\d.])", q)
    if m:
        mhz = float(m.group(1))
        if 24 <= mhz <= 1766: return ("freq", mhz)
        return ("badfreq", mhz)
    return None

def route(q):
    q = q.lower().strip()
    if q in ("quit", "exit", "q"): return ("quit",)
    if q in ("help", "?", ""): return ("help",)
    if q in ("stop", "s", "esc"): return ("stop",)
    if q == "clear": return ("clear",)
    if q in ("stat", "stats") or q.startswith("stats "): return ("stats",)
    if q.startswith("report") or has(q, "incident report", "make a report", "save report"): return ("report",)
    m = re.match(r"^(?:set\s+)?location\s+(-?\d+(?:\.\d+)?)[ ,]+(-?\d+(?:\.\d+)?)", q)
    if m: return ("setloc", float(m.group(1)), float(m.group(2)))
    if has(q, "aircraft", "airplane", "planes", "plane overhead", "flights", "adsb", "ads-b", "radar",
           "helicopter", "map"):
        return ("adsb",)
    if re.match(r"^(watch|monitor)\b", q):
        inner = route_scan(re.sub(r"^(watch|monitor)\s*", "", q)) or ("preset", "fm")
        if inner[0] == "badfreq": return inner
        return ("watch", inner)
    if re.match(r"^(hunt|locate|track|find|direction)\b", q):
        inner = route_scan(re.sub(r"^\w+\s*", "", q))
        if inner and inner[0] == "freq": return ("hunt", inner[1])
        if inner and inner[0] == "badfreq": return inner
        return ("hunthelp",)
    if has(q, "where is the signal", "where are the signal", "where is it coming", "location of",
           "direction of", "triangulat"):
        return ("hunthelp",)
    if has(q, "disaster", "emergency", "comms", "communication", "outage", "what works", "status check",
           "are phones", "cell service", "is there service", "first responder", "responders"):
        return ("disaster",)
    if has(q, "band table", "us table", "us band", "allocation", "band plan", "all bands", "list bands",
           "list frequencies", "frequency table") or q.startswith("band"):
        return ("bands",)
    if has(q, "weather radio", "noaa"): return ("preset", "noaa")
    if re.match(r"^(scan|check|sweep)\b", q):
        if has(q, "weather"): return ("preset", "noaa")
        if has(q, "walkie", "gmrs"): return ("preset", "frs")
        if has(q, "key fob", "keyfob", "garage"): return ("preset", "fob")
        if has(q, "police", "fire", "public safety"): return ("preset", "safety")
        if has(q, "cell", "phone", "lte"): return ("preset", "cell")
        if has(q, "pilot", "air traffic", "aviation"): return ("preset", "air")
    r = route_scan(q)
    if r: return r
    if QUESTION_RE.match(q) or len(q.split()) >= 3:
        return ("ask", q)
    return ("help",)

HUNT_HELP = ("One antenna can't tell where a signal comes from (that needs several receivers or a directional "
             "antenna). Two things RFScout can do: 'aircraft' maps planes from the GPS positions they broadcast, "
             "and 'hunt <MHz>' shows a live strength meter so you can walk toward a transmitter "
             "(shield the antenna with your body to find the direction).")

# ---------- CLI ----------
def run_scan(label, lo, hi, binw, mock):
    """CLI: print one scan. Returns (facts, n_peaks); facts None if the SDR returned nothing."""
    print(col(f"  [sweeping {lo:.3f}–{hi:.3f} MHz ...]", "d"))
    res = analyze(label, lo, hi, binw, mock)
    if not res["ok"]:
        print(col("  no data from SDR (is it plugged in?)", "r")); return None, 0
    if not res["peaks"]:
        print(f"  nothing above the noise floor ({res['floor']:.1f} dB)")
    for k in res["peaks"]:
        n = max(1, int(k["db"] / 1.5))
        print(f"  {k['freq']:9.3f} MHz |{col('█' * min(n, 30), 'g')} +{k['db']:4.0f} dB  {col('← ' + k['band'], 'd')}")
    return res["facts"], len(res["peaks"])

def answer(facts, query, system=SYSTEM, fallback=None):
    out = explain_full(facts, system, fallback)
    if not out["checked"]:
        print(col("  [AI reply failed fact check, using verified summary]", "y"))
    print(col("RFScout: ", "c") + out["text"])
    print(col(f"  [{out['latency']:.1f}s" + (f", {out['tps']:.1f} tok/s" if out["tps"] else "") + "]", "d"))
    log_metric(query, out["latency"], out["tps"])

def say(text):
    print(col("RFScout: ", "c") + text)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="use fake sweep data")
    a = ap.parse_args()
    mock = a.mock
    if not mock and not shutil.which("rtl_power"):
        sys.exit(col("rtl_power not found. Install rtl-sdr, or run with --mock for simulated data.", "r"))

    print(col("RFScout", "c") + " — offline RF spectrum assistant on TI SK-AM62B")
    if mock: print(col("  (SIMULATED SDR data)", "y"))
    print(col("  loading model...", "d"))
    load_s = start_server()
    print(col(f"  model ready in {load_s:.1f}s — type 'help'  (rfscout_tui.py has the full dashboard)", "d"))

    try:
        while True:
            try: q = input(col("\nRFScout > ", "g"))
            except EOFError: break
            r = route(q)
            k = r[0]
            try:
                if k == "quit": break
                elif k in ("help", "clear", "stop"): print(HELP)
                elif k == "bands":
                    for lo, hi, name, _ in BANDS: print(f"  {lo:8.3f}–{hi:8.3f} MHz  {name}")
                elif k == "stats":
                    print(f"  model: {os.path.basename(MODEL)}  load: {load_s:.1f}s  peak mem: {peak_mem_mb() or 0:.0f} MB")
                elif k == "badfreq":
                    print(f"  {r[1]:g} MHz is outside the RTL-SDR range (24–1766 MHz).")
                elif k == "hunthelp": say(HUNT_HELP)
                elif k == "setloc":
                    cfg = load_cfg(); cfg.update(lat=r[1], lon=r[2], place="custom"); save_cfg(cfg)
                    say(f"Location set to {r[1]:.4f}, {r[2]:.4f}.")
                elif k == "report":
                    text, path = make_report(); print(text); print(col(f"  saved to {path}", "d"))
                elif k == "ask":
                    out = answer_question(q)
                    if out.get("model") and not out["checked"]:
                        print(col("  [AI reply failed fact check, using band table]", "y"))
                    say(out["text"])
                elif k == "disaster":
                    print(col("  EMERGENCY CHECK: scanning emergency communication channels", "r"))
                    rep = comms_report(mock, progress=lambda s: print(col("  " + s + " ...", "d")))
                    print(rep["card"])
                    answer(rep["facts"], q, SYSTEM_COMMS, rep["summary"])
                elif k == "watch":
                    label, lo, hi, binw = scan_target(r[1])
                    w = Watcher(watch_tol(lo, hi))
                    print(col(f"  watching {label} ({lo:g}-{hi:g} MHz), Ctrl-C to stop", "y"))
                    try:
                        while True:
                            res = analyze(label, lo, hi, binw, mock, log=False)
                            if not res["ok"]: print(col("  no data from SDR", "r")); break
                            for ev, f, db, band in w.update(res):
                                print((col("  NEW  ", "g") if ev == "new" else col("  LOST ", "r")) +
                                      f"{time.strftime('%H:%M:%S')}  {f:.3f} MHz  +{db:.0f} dB  {band}")
                            if mock: time.sleep(1)
                    except KeyboardInterrupt:
                        print(col("  stopped", "d"))
                elif k == "hunt":
                    print(col(f"  hunting {r[1]:g} MHz, Ctrl-C to stop. Walk around and watch the level.", "y"))
                    try:
                        while True:
                            v = hunt_level(r[1], mock)
                            if v is None: print(col("  no data from SDR", "r")); break
                            print(f"  {time.strftime('%H:%M:%S')} |{col('█' * int(max(0, v)), 'g')} +{v:.0f} dB")
                            if mock: time.sleep(0.5)
                    except KeyboardInterrupt:
                        print(col("  stopped", "d"))
                elif k == "adsb":
                    cfg = load_cfg(); tr = AdsbTracker(cfg)
                    print(col("  listening for aircraft on 1090 MHz for 30 s ...", "d"))
                    if not adsb_capture(tr, 30, mock): print(col("  rtl_adsb not found", "r")); continue
                    planes = tr.current()
                    if not planes: say("No aircraft decoded. Try an antenna near a window, or listen longer.")
                    for p in planes:
                        print(f"  {p.get('callsign') or p['icao']:8} " +
                              (f"{p['alt']:>6} ft " if "alt" in p else "     ? ft ") +
                              (f"{p['dist']:5.1f} mi @ {p['brg']:3.0f}°" if "dist" in p else "  no position yet"))
                else:
                    job = plan(r)
                    if job["note"]: print(col("  " + job["note"], "y"))
                    facts, total = [], 0
                    for scan in job["scans"]:
                        f, n = run_scan(*scan, mock)
                        if f is None: continue
                        facts.append(f); total += n
                    if not facts: continue
                    if total: answer(job["prefix"] + "\n".join(facts), q)
                    else: say(job["empty"])
            except (urllib.error.URLError, ConnectionError) as e:
                print(col(f"  AI model not reachable: {e}", "r"))
    finally:
        if server_proc: server_proc.terminate()

if __name__ == "__main__":
    main()
