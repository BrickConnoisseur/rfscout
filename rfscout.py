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
    """Return list of (freq_MHz, power_dB)."""
    if mock or not shutil.which("rtl_power"):
        return mock_sweep(lo, hi)
    out = "/tmp/rfscout_sweep.csv"
    if os.path.exists(out): os.remove(out)
    cmd = ["rtl_power", "-f", f"{lo}M:{hi}M:{binw}", "-i", "1", "-g", "40", "-1", out]
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

def find_peaks(pts, margin=10.0, top=5):
    if not pts: return [], None
    floor = statistics.median(p for _, p in pts)
    peaks = []
    for i in range(1, len(pts) - 1):
        f, p = pts[i]
        if p > floor + margin and p >= pts[i - 1][1] and p >= pts[i + 1][1]:
            if not peaks or abs(f - peaks[-1][0]) > 0.05:
                peaks.append((f, p))
            elif p > peaks[-1][1]:
                peaks[-1] = (f, p)
    peaks.sort(key=lambda x: -x[1])
    return peaks[:top], floor

def bars(peaks, floor):
    for f, p in peaks:
        n = max(1, int((p - floor) / 1.5))
        name, _ = lookup(f)
        print(f"  {f:9.3f} MHz |{col('█' * min(n, 30), 'g')} {p:6.1f} dB  {col('← ' + name, 'd')}")

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
          "Explain radio scan results to a non-expert in 2 short sentences. "
          "Only use the facts given. Do not invent frequencies or stations.")

def explain(facts):
    body = json.dumps({
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": facts}],
        "max_tokens": 80, "temperature": 0.2}).encode()
    t0 = time.time()
    req = urllib.request.Request(URL + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=180).read())
    dt = time.time() - t0
    text = r["choices"][0]["message"]["content"].strip()
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
  scan fm | noaa | air | ism433 | fob | frs | ism915
  disaster                  emergency mode: NOAA + FM check
  bands                     list known bands
  stats                     model + memory stats
  quit"""

def route(q):
    q = q.lower().strip()
    if q in ("quit", "exit", "q"): return ("quit",)
    if q in ("help", "?", ""): return ("help",)
    if q.startswith("band"): return ("bands",)
    if q.startswith("stat"): return ("stats",)
    if "disaster" in q or "emergency" in q: return ("disaster",)
    for key in PRESETS:
        if key in q.replace(" ", ""): return ("preset", key)
    if "weather" in q: return ("preset", "noaa")
    if "key fob" in q or "keyfob" in q or "garage" in q: return ("preset", "fob")
    if "aircraft" in q or "plane" in q: return ("preset", "air")
    m = re.search(r"(\d{2,4}(?:\.\d+)?)", q)
    if m:
        mhz = float(m.group(1))
        if 24 <= mhz <= 1766: return ("freq", mhz)
        return ("badfreq", mhz)
    return ("help",)

def run_scan(label, lo, hi, binw, mock):
    print(col(f"  [sweeping {lo:.3f}–{hi:.3f} MHz ...]", "d"))
    pts = sweep(lo, hi, binw, mock)
    peaks, floor = find_peaks(pts)
    if floor is None:
        print(col("  no data from SDR (is it plugged in?)", "r")); return None
    if not peaks:
        print(f"  nothing above the noise floor ({floor:.1f} dB)")
        return f"Scan of {label} ({lo}-{hi} MHz): no signals above the noise floor."
    bars(peaks, floor)
    lines = []
    for f, p in peaks:
        name, desc = lookup(f)
        lines.append(f"- {f:.3f} MHz, {p - floor:.0f} dB above noise: {name} ({desc})")
    return f"Scan of {label} ({lo}-{hi} MHz) found these signals:\n" + "\n".join(lines)

def answer(facts, query):
    text, dt, tps = explain(facts)
    print(col("RFScout: ", "c") + text)
    print(col(f"  [{dt:.1f}s" + (f", {tps:.1f} tok/s" if tps else "") + "]", "d"))
    log_metric(query, dt, tps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="use fake sweep data")
    a = ap.parse_args()
    mock = a.mock or not shutil.which("rtl_power")

    print(col("RFScout", "c") + " — offline RF spectrum assistant on TI SK-AM62B")
    if mock: print(col("  (mock SDR data)", "y"))
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
                print(f"  {r[1]} MHz is outside the RTL-SDR range (24–1766 MHz).")
            elif r[0] == "preset":
                lo, hi, binw, label = PRESETS[r[1]]
                facts = run_scan(label, lo, hi, binw, mock)
                if facts: answer(facts, q)
            elif r[0] == "freq":
                mhz = r[1]
                name, desc = lookup(mhz)
                print(f"  {mhz} MHz is in: {col(name, 'y')} — {desc}")
                facts = run_scan(name, mhz - 0.5, mhz + 0.5, "5k", mock)
                if facts: answer(f"User asked about {mhz} MHz ({name}: {desc}).\n" + facts, q)
            elif r[0] == "disaster":
                print(col("  DISASTER MODE: checking emergency broadcast channels", "r"))
                f1 = run_scan("NOAA Weather Radio", 162.35, 162.60, "5k", mock) or ""
                f2 = run_scan("FM broadcast", 88.0, 108.0, "50k", mock) or ""
                answer("Emergency check. Tell the user which emergency info sources are on the air.\n"
                       + f1 + "\n" + f2, q)
    finally:
        if server_proc: server_proc.terminate()

if __name__ == "__main__":
    main()
