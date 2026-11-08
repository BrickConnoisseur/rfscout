#!/usr/bin/env python3
"""
RFScout - offline RF spectrum assistant for the TI SK-AM62B.

Flow:  question -> intent router -> rtl_power sweep -> peak detection
       -> band lookup (local table) -> small language model explains it.

Everything runs on the board. No internet, no cloud APIs.

Usage:
  python3 rfscout.py           # live SDR
  python3 rfscout.py --mock    # fake sweep data (test the AI side without the SDR)
"""
import argparse, csv, json, os, re, shutil, statistics, subprocess, sys, time, urllib.request

HOME = os.path.expanduser("~/rfscout")
MODEL = os.path.join(HOME, "models", "model.gguf")
SERVER_BIN = os.path.join(HOME, "llama.cpp", "build", "bin", "llama-server")
LOG_DIR = os.path.join(HOME, "logs")
PORT = 8080
URL = f"http://127.0.0.1:{PORT}"

GAIN = "40"          # rtl_power gain in dB; try 49.6 for more sensitivity
FLOOR_PCT = 0.20     # noise floor = 20th percentile (median is inflated in crowded bands)
MARGIN_DB = 10.0     # how far above the floor a peak must be

# ---------- colors ----------
C = {"g": "\033[92m", "y": "\033[93m", "c": "\033[96m", "r": "\033[91m", "d": "\033[2m", "x": "\033[0m"}
def col(s, k): return f"{C[k]}{s}{C['x']}"

# ---------- local band table (US), most specific first ----------
BANDS = [
    (162.390, 162.560, "NOAA Weather Radio", "continuous weather forecasts and emergency alerts from the National Weather Service"),
    (433.050, 434.790, "433 MHz ISM band", "low-power devices like wireless weather sensors, remote controls, and doorbells"),
    (314.000, 316.000, "315 MHz short-range devices", "US car key fobs, garage door openers, and tire pressure sensors"),
    (462.550, 467.725, "FRS/GMRS", "handheld walkie-talkies used by families and outdoor groups"),
    (1089.0, 1091.0, "ADS-B", "aircraft broadcasting their position, altitude, and speed"),
    (1575.0, 1576.0, "GPS L1", "GPS satellite navigation signals"),
    (88.0, 108.0, "FM broadcast radio", "commercial and public FM radio stations"),
    (108.0, 118.0, "Aviation navigation", "VOR and ILS beacons that guide aircraft"),
    (118.0, 137.0, "Aviation voice", "pilots talking to air traffic control"),
    (137.0, 138.0, "Weather satellites", "NOAA and Meteor satellites sending weather images"),
    (144.0, 148.0, "2 m amateur radio", "licensed ham radio operators"),
    (156.0, 162.0, "Marine VHF", "boats and harbor communications"),
    (150.8, 174.0, "VHF land mobile", "public safety, business, and utility radios"),
    (420.0, 450.0, "70 cm amateur radio", "licensed ham radio operators"),
    (450.0, 470.0, "UHF land mobile", "business, public safety, and utility radios"),
    (470.0, 608.0, "UHF TV", "over-the-air digital television"),
    (698.0, 806.0, "700 MHz cellular", "LTE cell phone networks"),
    (806.0, 824.0, "800 MHz public safety", "police, fire, and EMS trunked radio"),
    (824.0, 894.0, "850 MHz cellular", "cell phone networks"),
    (902.0, 928.0, "915 MHz ISM band", "LoRa sensors, smart meters, and other IoT devices"),
    (929.0, 932.0, "Paging", "pager networks"),
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
}

# ---------- SDR ----------
def sweep(lo, hi, binw, mock=False):
    """Return list of (freq_MHz, power_dB). Never falls back to fake data silently."""
    if mock:
        return mock_sweep(lo, hi)
    out = "/tmp/rfscout_sweep.csv"
    if os.path.exists(out): os.remove(out)
    cmd = ["rtl_power", "-f", f"{lo}M:{hi}M:{binw}", "-i", "1", "-g", GAIN, "-1", out]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
    except subprocess.TimeoutExpired:
        print(col("  sweep timed out", "r")); return []
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

def mock_sweep(lo, hi):
    import random
    random.seed(int(lo * 100))
    n = 200
    pts = [(lo + (hi - lo) * i / n, -60 + random.uniform(-3, 3)) for i in range(n + 1)]
    fake = [162.400, 162.550, 94.5, 98.7, 102.1, 433.92, 315.0, 121.5]
    for fc in fake:
        if lo <= fc <= hi:
            pts = [(f, p + max(0, 25 - abs(f - fc) * 400)) for f, p in pts]
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

def bars(peaks, floor):
    for f, p in peaks:
        above = p - floor
        n = max(1, int(above / 1.5))
        name, _ = lookup(f)
        print(f"  {f:9.3f} MHz |{col('█' * min(n, 30), 'g')} +{above:4.0f} dB  {col('← ' + name, 'd')}")

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

def peak_mem_mb():
    try:
        with open(f"/proc/{server_proc.pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM"):
                    return int(line.split()[1]) / 1024
    except Exception:
        return None

SYSTEM = ("You are RFScout, a radio assistant running offline on a small device. "
          "Summarize the scan for a non-expert in 1 or 2 short sentences. "
          "Only use the facts given. Do not invent frequencies or stations. "
          "Do not repeat the facts word for word and do not repeat yourself.")

def dedupe(text):
    """Safety net for small models: drop repeated lines/sentences."""
    seen, out = set(), []
    for line in re.split(r"(?<=[.!?])\s+|\n+", text):
        key = line.strip().lower()
        if key and key not in seen:
            seen.add(key); out.append(line.strip())
    return " ".join(out)

def explain(facts):
    body = json.dumps({
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": facts}],
        "max_tokens": 120, "temperature": 0.2, "repeat_penalty": 1.3}).encode()
    t0 = time.time()
    req = urllib.request.Request(URL + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=180).read())
    dt = time.time() - t0
    text = dedupe(r["choices"][0]["message"]["content"].strip())
    tps = r.get("timings", {}).get("predicted_per_second")
    return text, dt, tps

def log_metric(query, latency, tps):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, "metrics.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["time", "query", "latency_s", "tokens_per_s", "peak_mem_mb"])
        w.writerow([time.strftime("%H:%M:%S"), query, f"{latency:.2f}", tps and f"{tps:.1f}", peak_mem_mb()])

# ---------- intent router ----------
HELP = """Ask things like:
  what's at 162.55          scan around a frequency
  between 88 and 108 MHz    scan a range (also 88-108, 88 to 108)
  scan fm | noaa | air | ism433 | fob | frs | ism915
  disaster                  emergency mode: NOAA + FM check
  bands                     list known bands
  stats                     model + memory stats
  quit"""

NUM = r"(\d{2,4}(?:\.\d+)?)\s*(?:mhz|m)?"
RANGE_RE = re.compile(NUM + r"\s*(?:-|–|to|and|through|:)\s*" + NUM)

def preset_match(q):
    for key in PRESETS:
        pat = re.sub(r"([a-z]+)(\d+)", r"\1\\s*\2", key)   # ism433 -> ism\s*433
        if re.search(rf"\b{pat}\b", q):
            return key
    return None

def route(q):
    q = q.lower().strip()
    if q in ("quit", "exit", "q"): return ("quit",)
    if q in ("help", "?", ""): return ("help",)
    if q.startswith("band"): return ("bands",)
    if q.startswith("stat"): return ("stats",)
    if "disaster" in q or "emergency" in q: return ("disaster",)
    key = preset_match(q)
    if key: return ("preset", key)
    if "weather" in q: return ("preset", "noaa")
    if "key fob" in q or "keyfob" in q or "garage" in q: return ("preset", "fob")
    if "aircraft" in q or "plane" in q: return ("preset", "air")
    m = RANGE_RE.search(q)
    if m:
        lo, hi = sorted((float(m.group(1)), float(m.group(2))))
        if lo < 24 or hi > 1766: return ("badfreq", lo if lo < 24 else hi)
        if hi - lo < 0.1: return ("freq", lo)
        return ("range", lo, hi)
    m = re.search(r"(\d{2,4}(?:\.\d+)?)", q)
    if m:
        mhz = float(m.group(1))
        if 24 <= mhz <= 1766: return ("freq", mhz)
        return ("badfreq", mhz)
    return ("help",)

def binwidth_for(lo, hi):
    """~400 bins across the span, at least 5 kHz."""
    k = max(5.0, (hi - lo) * 1000 / 400)
    return f"{k:g}k" if k < 1000 else f"{k/1000:g}M"

def run_scan(label, lo, hi, binw, mock):
    """Returns (facts, n_peaks). facts is None if the SDR returned nothing."""
    print(col(f"  [sweeping {lo:.3f}–{hi:.3f} MHz ...]", "d"))
    pts = sweep(lo, hi, binw, mock)
    min_sep = 0.15 if hi - lo > 5 else 0.02     # FM stations are ~200 kHz wide
    peaks, floor = find_peaks(pts, min_sep=min_sep)
    if floor is None:
        print(col("  no data from SDR (is it plugged in?)", "r")); return None, 0
    if not peaks:
        print(f"  nothing above the noise floor ({floor:.1f} dB)")
        return f"Scan of {label} ({lo:g}-{hi:g} MHz): no signals above the noise floor.", 0
    bars(peaks, floor)
    lines = []
    for f, p in peaks:
        name, desc = lookup(f)
        lines.append(f"- {f:.3f} MHz, {p - floor:.0f} dB above noise: {name} ({desc})")
    return f"Scan of {label} ({lo:g}-{hi:g} MHz) found these signals:\n" + "\n".join(lines), len(peaks)

def answer(facts, query):
    text, dt, tps = explain(facts)
    print(col("RFScout: ", "c") + text)
    print(col(f"  [{dt:.1f}s" + (f", {tps:.1f} tok/s" if tps else "") + "]", "d"))
    log_metric(query, dt, tps)

def say(text):
    """Deterministic reply, no model call (used when there is nothing to explain)."""
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
    print(col(f"  model ready in {load_s:.1f}s — type 'help'", "d"))

    try:
        while True:
            try: q = input(col("\nRFScout > ", "g"))
            except EOFError: break
            r = route(q)
            if r[0] == "quit": break
            elif r[0] == "help": print(HELP)
            elif r[0] == "bands":
                for lo, hi, name, _ in BANDS: print(f"  {lo:8.3f}–{hi:8.3f} MHz  {name}")
            elif r[0] == "stats":
                print(f"  model: {os.path.basename(MODEL)}  load: {load_s:.1f}s  peak mem: {peak_mem_mb() or 0:.0f} MB")
            elif r[0] == "badfreq":
                print(f"  {r[1]:g} MHz is outside the RTL-SDR range (24–1766 MHz).")
            elif r[0] == "preset":
                lo, hi, binw, label = PRESETS[r[1]]
                facts, n = run_scan(label, lo, hi, binw, mock)
                if facts is None: continue
                if n: answer(facts, q)
                else: say(f"No signals stood out in {label} ({lo:g}-{hi:g} MHz).")
            elif r[0] == "range":
                lo, hi = r[1], r[2]
                label = f"{lo:g}-{hi:g} MHz"
                facts, n = run_scan(label, lo, hi, binwidth_for(lo, hi), mock)
                if facts is None: continue
                if n: answer(facts, q)
                else: say(f"No signals stood out between {lo:g} and {hi:g} MHz.")
            elif r[0] == "freq":
                mhz = r[1]
                name, desc = lookup(mhz)
                print(f"  {mhz:g} MHz is in: {col(name, 'y')} — {desc}")
                facts, n = run_scan(name, max(24.0, mhz - 0.5), min(1766.0, mhz + 0.5), "5k", mock)
                if facts is None: continue
                if n: answer(f"Frequency of interest: {mhz:g} MHz, in the {name} band ({desc}).\n" + facts, q)
                else: say(f"Nothing active near {mhz:g} MHz right now. That frequency is in the {name} band, used for {desc}.")
            elif r[0] == "disaster":
                print(col("  DISASTER MODE: checking emergency broadcast channels", "r"))
                f1, n1 = run_scan("NOAA Weather Radio", 162.35, 162.60, "5k", mock)
                f2, n2 = run_scan("FM broadcast", 88.0, 108.0, "50k", mock)
                if not (n1 or n2):
                    say("No NOAA Weather Radio or FM broadcast signals detected. Check the antenna and location.")
                    continue
                answer("Emergency check. Say which of these information sources have signals present. "
                       "Do not claim any alert or message was decoded.\n"
                       + (f1 or "") + "\n" + (f2 or ""), q)
    finally:
        if server_proc: server_proc.terminate()

if __name__ == "__main__":
    main()
