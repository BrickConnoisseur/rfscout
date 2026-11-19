RFScout

An offline Edge AI radio spectrum assistant on the TI SK-AM62B.

Ask about the radio spectrum around you in plain English. RFScout measures real signals with an RTL-SDR, deterministic code verifies what it found, and a small language model running on the board explains it. No internet, no cloud API: everything runs on a quad Cortex-A53 with 2 GB of RAM.

Built for UTiD HackEdge 2026, Track 3: Offline Edge AI Assistant using Small Language Models.

How it works
question ─► SLM tool selector (GBNF JSON) ─► code validation ─► vetted tool ─► verified facts ─► SLM ─► answer
              │ invalid / no match                                • SDR sweep (rtl_power)
              ▼                                                   • peak / SNR / bandwidth
          regex intent router (fallback)                          • band table lookup
                                                                  • EE calculators
                                                                  • device status (read-only)
empty scan or pure calculation ─► answered directly by code (model skipped)
anything else ─► local knowledge note (knowledge.txt, with line sources) ─► SLM ─► answer

Key design rule: the model never reads raw radio samples, never does math, and never executes anything directly. It picks a tool through grammar-constrained JSON, code validates that call (allowed tool, frequency range, span and gain limits), code measures, and the model only explains verified facts. Actions that change hardware state (SDR gain) ask for confirmation first.

Hardware
Part	Role
TI SK-AM62B	Quad Cortex-A53, 2 GB DDR4, Debian Trixie (TI Processor SDK 12.01)
RTL-SDR (RTL2832U + R828D) with dipole kit	Receive-only, about 24–1766 MHz
microSD (32 GB)	OS, model, and code
USB extension cable (recommended)	Moves the dongle away from board noise
Software and dependencies
Component	Detail
Model (default)	SmolLM2-360M-Instruct, Q4_K_M GGUF, Apache 2.0
Model (fallback)	SmolLM2-135M-Instruct, Q4_K_M GGUF (~100 MB), Apache 2.0
Runtime	llama.cpp llama-server, CPU only, GBNF grammar for tool calls
SDR driver	rtl-sdr-blog (R828D support), rtl_power for sweeps
App	rfscout.py (Python 3, standard library only)
Dashboard	dashboard.py (Python curses, full-screen terminal UI)
Build tools	git, cmake, build-essential, pkg-config, libusb-1.0-0-dev, curl
Repository layout
rfscout.py         main app: router, tools, signal processing, SLM calls
dashboard.py       full-screen terminal dashboard with live spectrum
knowledge.txt      local notes used for general questions
setup.sh           one-time install (packages, llama.cpp, SDR driver, models)
run_assistant.sh   launcher (starts llama-server + RFScout)
.gitignore         keeps models, builds, and logs out of the repo
Setup (one time, needs internet)
Flash the SK-AM62B Debian image to a microSD card, boot the board, and log in.
Clone and install:
bash
git clone https://github.com/<your-username>/rfscout.git ~/rfscout
cd ~/rfscout
./setup.sh
sudo reboot

setup.sh installs build tools, builds llama.cpp (CPU, -j2 to fit in 2 GB RAM), builds and installs the rtl-sdr-blog driver, blacklists the dvb_usb_rtl28xxu kernel module so librtlsdr can claim the dongle, and downloads the GGUF model(s) into models/.

Verify the SDR after reboot:
bash
rtl_test -t

If you set things up as root, the project lives in /root/rfscout. Run sudo -i first.

Run (offline)

Unplug Ethernet / tethering, then:

bash
~/rfscout/run_assistant.sh          # live SDR, chat mode
~/rfscout/run_assistant.sh --mock   # recorded-style data, no SDR needed
python3 ~/rfscout/dashboard.py      # full-screen dashboard with live spectrum

Check it is really offline at any time with ip -br addr.

Commands

Radio

scan fm | noaa | air | ism433 | fob | frs | ism915 — sweep a band and explain the strongest signals
what's at 162.55 — identify the band, scan around it, explain
disaster — emergency mode: which broadcast and weather sources are on air
bands — list the local band table

EE tools (computed in code; the model only explains)

measure 433.92 — peak frequency, SNR, −3/−10 dB bandwidth, offset in kHz and ppm
occupancy fm — 5 sweeps showing how often each channel is active
antenna 162.55 — wavelength and dipole arm length
fspl 162.55 10 — free-space path loss (MHz, km)
dbm 5w / dbm 30 — power unit conversion
export — save the last sweep to CSV

Device and evaluation

status — model, quantization, load time, peak memory, latency, CPU temp, offline check
gain 30 — change SDR gain (asks y/N)
selftest — Track 3 acceptance prompts
bench — 5-run latency and tokens/sec benchmark
evidence — save a timestamped metrics snapshot (including model SHA-256) to logs/

Plain-English questions ("is anything transmitting near the weather band?") go to the SLM tool selector.

Interference handling
DC-spike removal: suppresses the RTL-SDR's center spike in each tuned segment.
Baseline subtraction: record a background sweep, then subtract it so persistent local emitters (including the board's own noise near 315 MHz) are not reported as signals.
Results (measured on the SK-AM62B)
Metric	SmolLM2-135M	SmolLM2-360M
Quantization	Q4_K_M	Q4_K_M
Load time	2.2 s	fill in
Peak memory	179 MB	fill in
Decode speed	~9.8 tok/s	fill in
Mean answer latency	~7 s	fill in
Network required	None	None
Limitations
RTL-SDR range excludes 2.4 / 5.8 GHz (Wi-Fi, Bluetooth, drones).
Receive-only; demos need a transmitter such as a key fob.
~2.4 MHz instantaneous bandwidth; wide views use stepped sweeps.
Power readings are relative (uncalibrated), not absolute dBm.
Signal labels come from US band allocations, not demodulation.
Next steps
Multi-step questions (model plans several validated tool calls)
On-device signal classifier
Wider-range SDR or downconverter for 2.4 / 5.8 GHz
Credits
llama.cpp (MIT)
SmolLM2 (Apache 2.0)
rtl-sdr-blog (GPL-2.0)
