#!/usr/bin/env python3
"""
RFScout - an offline Edge AI radio assistant on the TI SK-AM62B.

A small language model (SmolLM2-135M, 4-bit GGUF, llama.cpp on the Cortex-A53 CPU)
answers questions about the radio spectrum around the board. An RTL-SDR measures
real signals; the model only explains facts that local code has already verified.

Flow:  question -> intent router -> vetted tool (SDR sweep / band table / device status)
       -> facts -> SLM explanation.   Anything else -> local knowledge note -> SLM.

No internet, no cloud APIs. Every answer is timed and logged as evidence.

Usage:
  python3 rfscout.py           # live SDR
  python3 rfscout.py --mock    # fake sweep data (no SDR attached)
"""
import argparse, csv, json, os, platform, re, shutil, statistics, subprocess, sys, time, urllib.request

HOME = os.path.expanduser("~/rfscout")
MODEL = os.path.join(HOME, "models", "model.gguf")
MODEL_ID = "SmolLM2-135M-Instruct"
SERVER_BIN = os.path.join(HOME, "llama.cpp", "build", "bin", "llama-server")
KNOWLEDGE = os.path.join(HOME, "knowledge.txt")
LOG_DIR = os.path.join(HOME, "logs")
PORT = 8080
URL = f"http://127.0.0.1:{PORT}"
CTX = 1024
THREADS = 4

# ---------- colors ----------
C = {"g": "\033[92m", "y": "\033[93m", "c": "\033[96m", "r": "\033[91m", "d": "\033[2m", "b": "\033[1m", "x": "\033[0m"}
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
    "noaa":  (162.335, 162.640, "5k", "NOAA Weather Radio"),
    "air":   (118.0, 137.0, "25k", "Aviation voice"),
    "ism433":(433.0, 434.9, "10k", "433 MHz ISM"),
    "fob":   (314.6, 315.6, "5k",  "315 MHz key fobs"),
    "frs":   (462.5, 467.8, "12.5k", "FRS/GMRS"),
    "ism915":(902.0, 928.0, "100k", "915 MHz ISM"),
}

# ---------- device status (read-only allowlist) ----------
def is_offline():
    """True if the kernel has no default route, i.e. no path to the internet."""
    try:
        with open("/proc/net/route") as f:
            for line in f.readlines()[1:]:
                parts = line.split()
                if len(parts) > 1 and parts[1] == "00000000":
                    return False
        return True
    except Exception:
        return None

def cpu_temp_c():
    temps = []
    base = "/sys/class/thermal"
    try:
        for z in os.listdir(base):
            p = os.path.join(base, z, "temp")
            if os.path.exists(p):
                with open(p) as f: temps.append(int(f.read().strip()) / 1000)
    except Exception:
        pass
    return max(temps) if temps else None

def uptime_str():
    try:
        with open("/proc/uptime") as f: s = int(float(f.read().split()[0]))
        return f"{s // 3600}h {(s % 3600) // 60}m"
    except Exception:
        return "?"

def mem_avail_mb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"): return int(line.split()[1]) / 1024
    except Exception:
        return None

def sdr_present():
    try:
        for d in os.listdir("/sys/bus/usb/devices"):
            p = os.path.join("/sys/bus/usb/devices", d)
            try:
                v = open(os.path.join(p, "idVendor")).read().strip()
                pr = open(os.path.join(p, "idProduct")).read().strip()
                if v == "0bda" and pr in ("2838", "2832"): return True
            except Exception:
                pass
    except Exception:
        pass
    return False

def quant_from_name(path):
    m = re.search(r"(q\d_k_[sml]|q\d_\d|q\d_k|f16|bf16|q8_0)", os.path.basename(path).lower())
    return m.group(1).upper() if m else "Q4_K_M"

# ---------- SDR ----------
SDR_GAIN = 40

def sweep(lo, hi, binw, mock=False):
    """Return list of (freq_MHz, power_dB)."""
    if mock:
        return mock_sweep(lo, hi)
    out = "/tmp/rfscout_sweep.csv"
    if os.path.exists(out): os.remove(out)
    cmd = ["rtl_power", "-f", f"{lo}M:{hi}M:{binw}", "-i", "1", "-g", str(SDR_GAIN), "-1", out]
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
            vals = []
            for v in row[6:]:
                try: vals.append(float(v))
                except ValueError: vals.append(None)
            remove_dc_spike(vals, step)
            for i, v in enumerate(vals):
                if v is not None: pts.append(((f_lo + i * step) / 1e6, v))
    pts.sort()
    return pts

def remove_dc_spike(vals, step_hz):
    """The RTL-SDR shows a false spike at the exact frequency it is tuned to (the middle of
    each rtl_power hop). Replace those few bins with the median of their neighbors."""
    n = len(vals)
    if n < 12: return
    mid = n // 2
    w = max(1, round(3000 / max(step_hz, 1)))
    lo, hi = max(0, mid - w), min(n, mid + w + 1)
    nb = [v for v in vals[max(0, lo - 6):lo] + vals[hi:hi + 6] if v is not None]
    if nb:
        fill = statistics.median(nb)
        for i in range(lo, hi): vals[i] = fill

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

LAST = {"pts": [], "label": None}

# ---------- EE measurement tools (all math done in code, not by the model) ----------
def measure(mhz, mock):
    """Peak frequency, relative power, SNR, -3/-10 dB bandwidth, frequency offset and ppm error."""
    binw = "1k"
    pts = sweep(mhz - 0.2, mhz + 0.3, binw, mock)        # tuned 50 kHz above target, away from the spike
    LAST["pts"], LAST["label"] = pts, f"measure_{mhz}"
    if not pts: return None
    floor = statistics.median(p for _, p in pts)
    i = max(range(len(pts)), key=lambda k: pts[k][1])
    fpk, ppk = pts[i]
    def width(drop):
        lo = hi = i
        while lo > 0 and pts[lo - 1][1] >= ppk - drop: lo -= 1
        while hi < len(pts) - 1 and pts[hi + 1][1] >= ppk - drop: hi += 1
        return (pts[hi][0] - pts[lo][0]) * 1000
    res_khz = (pts[1][0] - pts[0][0]) * 1000 if len(pts) > 1 else 0
    off_khz = (fpk - mhz) * 1000
    r = {"target": mhz, "peak": fpk, "power": ppk, "floor": floor, "snr": ppk - floor,
         "bw3": width(3), "bw10": width(10), "off": off_khz, "ppm": off_khz * 1000 / mhz, "res": res_khz}
    band, _ = lookup(fpk)
    print(col(f"  ── measurement @ {mhz} MHz ({band}) ──", "b"))
    print(f"  peak freq     {fpk:.4f} MHz   (offset {off_khz:+.1f} kHz, {r['ppm']:+.1f} ppm)")
    print(f"  peak power    {ppk:.1f} dB (relative, uncalibrated)")
    print(f"  noise floor   {floor:.1f} dB    SNR {r['snr']:.1f} dB")
    print(f"  bandwidth     {r['bw3']:.1f} kHz @ -3 dB    {r['bw10']:.1f} kHz @ -10 dB")
    print(col(f"  resolution    {res_khz:.2f} kHz/bin  (±{res_khz / 2 * 1000 / mhz:.1f} ppm)", "d"))
    if r["snr"] < 10:
        print(col("  no clear signal (SNR < 10 dB): bandwidth and offset are not meaningful", "y"))
    return r

def occupancy(key, mock, n=5):
    """Repeat sweeps and report how often each channel is active (duty cycle)."""
    lo, hi, binw, label = PRESETS[key]
    seen, maxp = {}, {}
    print(col(f"  [{n} sweeps of {label} {lo}–{hi} MHz ...]", "d"))
    for k in range(n):
        pts = sweep(lo, hi, binw, mock)
        LAST["pts"], LAST["label"] = pts, f"occupancy_{key}"
        peaks, _ = find_peaks(pts, top=10)
        for f, p in peaks:
            ch = round(f / 0.025) * 0.025
            seen[ch] = seen.get(ch, 0) + 1
            maxp[ch] = max(maxp.get(ch, -999), p)
    if not seen:
        print("  no channels active above the noise floor"); return None
    print(col(f"  ── channel occupancy, {label} ──", "b"))
    rows = sorted(seen, key=lambda c: (-seen[c], -maxp[c]))[:8]
    for ch in rows:
        duty = 100 * seen[ch] / n
        print(f"  {ch:9.3f} MHz  active {seen[ch]}/{n} sweeps ({duty:3.0f}%)  {col('█' * int(duty / 10), 'g')}  max {maxp[ch]:.1f} dB")
    return [(ch, seen[ch], n) for ch in rows]

def antenna(mhz):
    lam = 299.792458 / mhz
    vf = 0.95
    print(col(f"  ── antenna for {mhz} MHz ──", "b"))
    print(f"  wavelength          {lam:.3f} m")
    print(f"  quarter-wave        {lam / 4 * 100:.1f} cm free space, {lam / 4 * vf * 100:.1f} cm cut length (VF {vf})")
    print(f"  dipole: set EACH telescopic arm to ~{lam / 4 * vf * 100:.1f} cm")
    if lam / 4 * vf > 1.0:
        print(col("  longer than the kit's 1 m arms: extend fully (still works, less efficient)", "y"))

def fspl(mhz, km):
    loss = 20 * __import__("math").log10(km) + 20 * __import__("math").log10(mhz) + 32.44
    print(f"  free-space path loss at {mhz} MHz over {km} km = {col(f'{loss:.1f} dB', 'y')}")
    print(col("  FSPL(dB) = 20·log10(d_km) + 20·log10(f_MHz) + 32.44", "d"))

def dbm_convert(arg):
    import math
    m = re.match(r"(-?\d+(?:\.\d+)?)\s*(mw|uw|w|dbm)?$", arg.strip())
    if not m:
        print("  usage: dbm 5w | dbm 100mw | dbm 30"); return
    v, unit = float(m.group(1)), (m.group(2) or "dbm")
    if unit == "dbm":
        w = 10 ** ((v - 30) / 10)
        print(f"  {v} dBm = {w * 1000:.4g} mW = {w:.4g} W")
    else:
        w = v * {"w": 1, "mw": 1e-3, "uw": 1e-6}[unit]
        if w <= 0: print("  power must be > 0"); return
        print(f"  {v} {unit} = {10 * math.log10(w * 1000):.2f} dBm = {10 * math.log10(w):.2f} dBW")

def export_last():
    if not LAST["pts"]:
        print("  nothing to export yet; run a scan first"); return
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, time.strftime(f"sweep_{LAST['label']}_%H%M%S.csv"))
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["freq_mhz", "power_db_relative"])
        for fr, p in LAST["pts"]: w.writerow([f"{fr:.6f}", f"{p:.2f}"])
    print(f"  saved {len(LAST['pts'])} points to {path}")

def bars(peaks, floor):
    for f, p in peaks:
        n = max(1, int((p - floor) / 1.5))
        name, _ = lookup(f)
        print(f"  {f:9.3f} MHz |{col('█' * min(n, 30), 'g')} {p:6.1f} dB  {col('← ' + name, 'd')}")

# ---------- local knowledge note (keyword passage selection with provenance) ----------
def load_knowledge():
    if not os.path.exists(KNOWLEDGE): return []
    with open(KNOWLEDGE) as f:
        return [(i + 1, l.strip()) for i, l in enumerate(f) if l.strip() and not l.startswith("#")]

STOP = set("the a an is are what why how to of in on for and or it this that do does i you me my with be can".split())

def select_passages(q, notes, k=2):
    words = {w for w in re.findall(r"[a-z0-9]+", q.lower()) if w not in STOP and len(w) > 2}
    scored = []
    for n, line in notes:
        lw = set(re.findall(r"[a-z0-9]+", line.lower()))
        s = len(words & lw)
        if s: scored.append((s, n, line))
    scored.sort(key=lambda x: -x[0])
    return [(n, line) for _, n, line in scored[:k]]

# ---------- SLM ----------
server_proc = None
STATE = {"load_s": None, "latencies": [], "tps": []}

def start_server():
    global server_proc
    if not os.path.exists(SERVER_BIN): sys.exit(col(f"llama-server not found at {SERVER_BIN}", "r"))
    if not os.path.exists(MODEL): sys.exit(col(f"model not found at {MODEL}", "r"))
    t0 = time.time()
    server_proc = subprocess.Popen(
        [SERVER_BIN, "-m", MODEL, "-c", str(CTX), "-t", str(THREADS), "--host", "127.0.0.1", "--port", str(PORT)],
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

SYS_RF = ("You are RFScout, a radio assistant running offline on a small device. "
          "Summarize the scan for a non-expert in at most 2 short sentences. "
          "Do not repeat the list. Only use the facts given. Do not invent frequencies or stations.")
RF_EXAMPLE = [
    {"role": "user", "content": "Scan of FM broadcast: 3 signals. Strongest: 96.300 MHz (FM broadcast radio, 24 dB above noise); "
                                "101.100 MHz (FM broadcast radio, 18 dB above noise). FM broadcast radio is used for commercial and public FM radio stations."},
    {"role": "assistant", "content": "I found 3 FM radio stations on the air, and the strongest is at 96.3 MHz. "
                                     "These are ordinary commercial or public radio broadcasts."},
]

def clean(text):
    """Keep the model's answer short: drop echoed list lines and stop after 2 sentences."""
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith(("-", "*", "•"))]
    text = " ".join(lines)
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = " ".join(parts[:2]).strip()
    return out if out else text.strip()
SYS_GENERAL = ("You are RFScout, a helpful assistant running fully offline on a TI SK-AM62B edge computer. "
               "Answer briefly and follow the user's instructions exactly.")

def llm(system, user, max_tokens=60, examples=None):
    msgs = [{"role": "system", "content": system}] + (examples or []) + [{"role": "user", "content": user}]
    body = json.dumps({
        "messages": msgs, "max_tokens": max_tokens, "temperature": 0.2,
        "stop": ["\n\n", "\n-", "\nUser", "\nScan of"]}).encode()
    t0 = time.time()
    req = urllib.request.Request(URL + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=180).read())
    dt = time.time() - t0
    text = clean(r["choices"][0]["message"]["content"])
    tps = r.get("timings", {}).get("predicted_per_second")
    STATE["latencies"].append(dt)
    if tps: STATE["tps"].append(tps)
    return text, dt, tps

def log_metric(kind, query, answer, latency, tps):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, "metrics.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["timestamp", "kind", "query", "answer", "latency_s", "tokens_per_s", "peak_mem_mb", "offline"])
        pm = peak_mem_mb()
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), kind, query, answer, f"{latency:.2f}",
                    tps and f"{tps:.1f}", pm and f"{pm:.0f}", is_offline()])

def show(text, dt, tps):
    print(col("RFScout: ", "c") + text)
    print(col(f"  [{dt:.1f}s" + (f", {tps:.1f} tok/s" if tps else "") + "]", "d"))

def answer_rf(facts, query):
    text, dt, tps = llm(SYS_RF, facts, examples=RF_EXAMPLE)
    show(text, dt, tps)
    log_metric("rf", query, text, dt, tps)

def answer_general(q, notes):
    hits = select_passages(q, notes)
    if hits:
        ctx = "\n".join(f"[note line {n}] {line}" for n, line in hits)
        user = f"Use these local notes if relevant:\n{ctx}\n\nQuestion: {q}"
    else:
        user = q
    text, dt, tps = llm(SYS_GENERAL, user)
    show(text, dt, tps)
    if hits:
        print(col("  source: knowledge.txt lines " + ", ".join(str(n) for n, _ in hits), "d"))
    log_metric("general", q, text, dt, tps)

# ---------- built-in acceptance test (guide Step 19) ----------
SELFTEST = [
    ("A", "Explain edge AI in one sentence.", "relevant short definition"),
    ("B", "List two reasons to use offline inference.", "two grounded advantages"),
    ("C", "Reply with only the word READY.", "exact format"),
]

def selftest():
    print(col("  Running Track 3 acceptance prompts (offline: %s)" % is_offline(), "y"))
    for tid, prompt, cond in SELFTEST:
        print(col(f"\n  Test {tid}: {prompt}", "b") + col(f"  (pass if: {cond})", "d"))
        text, dt, tps = llm(SYS_GENERAL, prompt, max_tokens=64)
        show(text, dt, tps)
        if tid == "C":
            ok = text.strip().strip(".").upper() == "READY"
            print("  " + (col("PASS", "g") if ok else col("FAIL (small model did not follow exact format)", "r")))
        log_metric("selftest-" + tid, prompt, text, dt, tps)

def bench(n=5):
    prompt = "In one sentence, what does a software defined radio do?"
    print(col(f"  benchmarking {n} runs...", "d"))
    lat, tps = [], []
    for _ in range(n):
        _, dt, t = llm(SYS_GENERAL, prompt, max_tokens=48)
        lat.append(dt)
        if t: tps.append(t)
    print(f"  latency  mean {statistics.mean(lat):.2f}s  min {min(lat):.2f}s  max {max(lat):.2f}s")
    if tps: print(f"  decode   mean {statistics.mean(tps):.1f} tok/s")
    print(f"  peak mem {peak_mem_mb() or 0:.0f} MB (llama-server VmHWM)")
    log_metric("bench", prompt, f"n={n}", statistics.mean(lat), tps and statistics.mean(tps))

def print_status(mock):
    off = is_offline()
    t = cpu_temp_c()
    pm = peak_mem_mb()
    lat = STATE["latencies"]
    print(col("  ── RFScout on TI SK-AM62B ──", "b"))
    print(f"  board      {platform.machine()}  kernel {platform.release()}")
    print(f"  network    " + (col("OFFLINE (no default route)", "g") if off else col("ONLINE", "r") if off is False else "?"))
    print(f"  model      {MODEL_ID}  quant {quant_from_name(os.path.realpath(MODEL))}  "
          f"size {os.path.getsize(MODEL) / 1e6:.0f} MB  ctx {CTX}  threads {THREADS}")
    print(f"  runtime    llama.cpp (CPU, Cortex-A53)")
    print(f"  load time  {STATE['load_s']:.1f}s")
    print(f"  peak mem   {pm or 0:.0f} MB model server   free {mem_avail_mb() or 0:.0f} MB")
    if lat:
        print(f"  latency    mean {statistics.mean(lat):.2f}s over {len(lat)} answers"
              + (f"   decode {statistics.mean(STATE['tps']):.1f} tok/s" if STATE["tps"] else ""))
    print(f"  cpu temp   {t:.0f}°C" if t else "  cpu temp   ?")
    print(f"  uptime     {uptime_str()}")
    print(f"  SDR        " + ("mock data" if mock else ("RTL-SDR attached" if sdr_present() else col("not detected", "r")))
          + f"   gain {SDR_GAIN} dB")

def write_evidence(mock):
    """Snapshot everything a judge asks for into logs/evidence_<time>.txt."""
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, time.strftime("evidence_%Y%m%d_%H%M%S.txt"))
    import hashlib
    h = hashlib.sha256()
    with open(MODEL, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    lat = STATE["latencies"]
    lines = [
        f"timestamp        {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"arch             {platform.machine()}",
        f"kernel           {platform.release()}",
        f"offline          {is_offline()}",
        f"model            {MODEL_ID}",
        f"model file       {os.path.realpath(MODEL)}",
        f"quantization     {quant_from_name(os.path.realpath(MODEL))}",
        f"model sha256     {h.hexdigest()}",
        f"model size MB    {os.path.getsize(MODEL) / 1e6:.1f}",
        f"context / thr    {CTX} / {THREADS}",
        f"load time s      {STATE['load_s']:.2f}",
        f"peak mem MB      {peak_mem_mb() or 0:.0f}",
        f"answers          {len(lat)}",
        f"mean latency s   {statistics.mean(lat):.2f}" if lat else "mean latency s   n/a",
        f"mean tok/s       {statistics.mean(STATE['tps']):.1f}" if STATE["tps"] else "mean tok/s       n/a",
        f"cpu temp C       {cpu_temp_c()}",
        f"sdr              {'mock' if mock else sdr_present()}",
    ]
    with open(path, "w") as f: f.write("\n".join(lines) + "\n")
    print("\n".join("  " + l for l in lines))
    print(col(f"  saved {path}", "d"))

# ---------- intent router ----------
HELP = """RF tools (vetted functions, no model-generated commands):
  what's at 162.55          scan around a frequency
  scan fm | noaa | air | ism433 | fob | frs | ism915
  disaster                  emergency mode: NOAA + FM check
  bands                     list known bands
EE tools (computed in code, model only explains):
  measure 433.92            peak freq, SNR, -3/-10 dB bandwidth, offset in kHz and ppm
  occupancy fm              5 sweeps: how often each channel is active (duty cycle)
  antenna 162.55            wavelength and dipole arm length to set
  fspl 162.55 10            free-space path loss (MHz, km)
  dbm 5w | dbm 30           power unit conversion
  export                    save the last sweep to CSV
Device (read-only allowlist):
  status                    model, quant, load time, memory, latency, temp, offline check
  gain 30                   change SDR gain (asks for confirmation)
Evaluation:
  selftest                  Track 3 acceptance prompts A/B/C
  bench                     5-run latency and tok/s benchmark
  evidence                  save a metrics snapshot to logs/
Anything else is answered by the local model.  quit to exit."""

def route(q):
    q = q.lower().strip()
    if q in ("quit", "exit", "q"): return ("quit",)
    if q in ("help", "?", ""): return ("help",)
    if q.startswith("band"): return ("bands",)
    if q in ("status", "stats", "stat", "temp", "temperature", "uptime"): return ("status",)
    if q.startswith("selftest") or q == "test": return ("selftest",)
    if q.startswith("bench"): return ("bench",)
    if q.startswith("evidence"): return ("evidence",)
    m = re.match(r"gain\s+(\d{1,2})", q)
    if m: return ("gain", int(m.group(1)))
    if q.startswith("export"): return ("export",)
    m = re.match(r"(?:measure|meas)\s+(\d+(?:\.\d+)?)", q)
    if m:
        mhz = float(m.group(1))
        return ("measure", mhz) if 24 <= mhz <= 1766 else ("badfreq", mhz)
    m = re.match(r"(?:occupancy|occ|watch)\s+(\w+)", q)
    if m: return ("occupancy", m.group(1)) if m.group(1) in PRESETS else ("help",)
    m = re.match(r"(?:antenna|wavelength|lambda)\s+(\d+(?:\.\d+)?)", q)
    if m and float(m.group(1)) > 0: return ("antenna", float(m.group(1)))
    m = re.match(r"fspl\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)", q)
    if m and float(m.group(1)) > 0 and float(m.group(2)) > 0: return ("fspl", float(m.group(1)), float(m.group(2)))
    m = re.match(r"dbm\s+(.+)", q)
    if m: return ("dbm", m.group(1))
    if "disaster" in q or "emergency" in q: return ("disaster",)
    for key in PRESETS:
        if key in q.replace(" ", ""): return ("preset", key)
    rf_words = ("mhz", "scan", "signal", "frequency", "freq", "radio", "what's at", "whats at", "spectrum")
    if "weather radio" in q or ("weather" in q and any(w in q for w in rf_words)): return ("preset", "noaa")
    if "key fob" in q or "keyfob" in q or "garage" in q: return ("preset", "fob")
    if "aircraft" in q or "plane" in q: return ("preset", "air")
    m = re.search(r"(\d{2,4}(?:\.\d+)?)", q)
    if m and any(w in q for w in rf_words + ("at ", "on ")):
        mhz = float(m.group(1))
        if 24 <= mhz <= 1766: return ("freq", mhz)
        return ("badfreq", mhz)
    if re.fullmatch(r"\d{2,4}(?:\.\d+)?", q):
        mhz = float(q)
        return ("freq", mhz) if 24 <= mhz <= 1766 else ("badfreq", mhz)
    return ("general", q)

def run_scan(label, lo, hi, binw, mock):
    print(col(f"  [sweeping {lo:.3f}–{hi:.3f} MHz ...]", "d"))
    pts = sweep(lo, hi, binw, mock)
    LAST["pts"], LAST["label"] = pts, re.sub(r"\W+", "_", label.lower())
    peaks, floor = find_peaks(pts)
    if floor is None:
        print(col("  no data from SDR (is it plugged in?)", "r")); return None
    if not peaks:
        print(f"  nothing above the noise floor ({floor:.1f} dB)")
        return f"Scan of {label} ({lo}-{hi} MHz): no signals above the noise floor."
    bars(peaks, floor)
    lines, notes = [], {}
    for f, p in peaks[:3]:
        name, desc = lookup(f)
        lines.append(f"{f:.3f} MHz ({name}, {p - floor:.0f} dB above noise)")
        notes[name] = desc
    more = f" plus {len(peaks) - 3} weaker" if len(peaks) > 3 else ""
    return (f"Scan of {label}: {len(peaks)} signals. Strongest: " + "; ".join(lines) + ". "
            + " ".join(f"{k} is used for {v}." for k, v in notes.items()))

def confirm(prompt):
    """Ask before any action that changes hardware state. The dashboard replaces this."""
    return input(f"  {prompt} [y/N] ").strip().lower() == "y"

def dispatch(r, q, mock, notes):
    """Run one routed command. Returns False when the user quits."""
    global SDR_GAIN
    if r[0] == "quit": return False
    elif r[0] == "help": print(HELP)
    elif r[0] == "bands":
        for lo, hi, name, _ in BANDS: print(f"  {lo:8.3f}–{hi:8.3f} MHz  {name}")
    elif r[0] == "status": print_status(mock)
    elif r[0] == "selftest": selftest()
    elif r[0] == "bench": bench()
    elif r[0] == "evidence": write_evidence(mock)
    elif r[0] == "gain":
        g = max(0, min(49, r[1]))
        if confirm(f"Change SDR gain {SDR_GAIN} -> {g} dB?"):
            SDR_GAIN = g; print(f"  gain set to {g} dB")
        else:
            print("  cancelled")
    elif r[0] == "export": export_last()
    elif r[0] == "antenna": antenna(r[1])
    elif r[0] == "fspl": fspl(r[1], r[2])
    elif r[0] == "dbm": dbm_convert(r[1])
    elif r[0] == "measure":
        m = measure(r[1], mock)
        if m and m["snr"] >= 10:
            name, desc = lookup(m["peak"])
            answer_rf(f"Measured a signal at {m['peak']:.4f} MHz in the {name} band ({desc}). "
                      f"SNR {m['snr']:.0f} dB, -3 dB bandwidth {m['bw3']:.1f} kHz, "
                      f"offset {m['off']:+.1f} kHz from {m['target']} MHz. "
                      "Explain what this signal likely is and what the bandwidth suggests.", q)
    elif r[0] == "occupancy":
        rows = occupancy(r[1], mock)
        if rows:
            lo, hi, _, label = PRESETS[r[1]]
            facts = "\n".join(f"- {ch:.3f} MHz active in {s} of {n} sweeps ({lookup(ch)[0]})" for ch, s, n in rows)
            answer_rf(f"Channel occupancy over repeated sweeps of {label}:\n{facts}\n"
                      "Explain which channels are continuous broadcasters and which are intermittent.", q)
    elif r[0] == "badfreq":
        print(f"  {r[1]} MHz is outside the RTL-SDR range (24–1766 MHz).")
    elif r[0] == "preset":
        lo, hi, binw, label = PRESETS[r[1]]
        facts = run_scan(label, lo, hi, binw, mock)
        if facts: answer_rf(facts, q)
    elif r[0] == "freq":
        mhz = r[1]
        name, desc = lookup(mhz)
        print(f"  {mhz} MHz is in: {col(name, 'y')} — {desc}")
        facts = run_scan(name, mhz - 0.4, mhz + 0.6, "5k", mock)
        if facts: answer_rf(f"User asked about {mhz} MHz ({name}: {desc}).\n" + facts, q)
    elif r[0] == "disaster":
        print(col("  EMERGENCY MODE: checking broadcast sources that work when internet is down", "r"))
        f1 = run_scan("NOAA Weather Radio", 162.335, 162.640, "5k", mock) or ""
        f2 = run_scan("FM broadcast", 88.0, 108.0, "50k", mock) or ""
        answer_rf("Emergency check. Tell the user which emergency info sources are on the air.\n"
                  + f1 + "\n" + f2, q)
    elif r[0] == "general":
        answer_general(q, notes)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="use fake sweep data")
    a = ap.parse_args()
    mock = a.mock or not shutil.which("rtl_power")
    notes = load_knowledge()

    print(col("RFScout", "c") + " — offline Edge AI radio assistant on TI SK-AM62B")
    off = is_offline()
    print("  network: " + (col("OFFLINE", "g") if off else col("online (unplug for the demo)", "y")))
    if mock: print(col("  (mock SDR data)", "y"))
    print(col("  loading model...", "d"))
    STATE["load_s"] = start_server()
    print(col(f"  {MODEL_ID} ({quant_from_name(os.path.realpath(MODEL))}) ready in {STATE['load_s']:.1f}s"
              f", {peak_mem_mb() or 0:.0f} MB — type 'help'", "d"))

    try:
        while True:
            try: q = input(col("\nRFScout > ", "g"))
            except EOFError: break
            try:
                if not dispatch(route(q), q, mock, notes): break
            except KeyboardInterrupt:
                print(col("  cancelled", "d"))
            except Exception as e:
                print(col(f"  error: {e}", "r"))
    finally:
        if server_proc: server_proc.terminate()

if __name__ == "__main__":
    main()
