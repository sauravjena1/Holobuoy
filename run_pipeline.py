"""
run_pipeline.py — one command, no intervention.

  1. SSH to the Pi and run the capture (the session closes when it finishes)
  2. rsync the new TIFFs down to the laptop
  3. Reconstruct + detect on the laptop
  4. Launch the dashboard

    python run_pipeline.py              # capture, download, process, dashboard
    python run_pipeline.py --no-capture # skip step 1, use what is on the Pi
    python run_pipeline.py --local-only # skip 1 and 2, reprocess what is here
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import config as C


def run(cmd, label):
    print(f"\n{'='*62}\n  {label}\n{'='*62}")
    print("$ " + " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"\nFAILED at: {label}  (exit {r.returncode})")
    return r


def remote_capture(kind, num):
    """Run a capture on the Pi. SSH exits by itself when it completes."""
    remote_cmd = (
        f"python3 {C.PI_SCRIPT} "
        f"--kind {kind} "
        f"--num {num} "
        f"--interval {C.CAP_INTERVAL} "
        f"--shutter {C.CAP_SHUTTER} "
        f"--gain {C.CAP_GAIN}"
    )
    run(["ssh", f"{C.PI_USER}@{C.PI_HOST}", remote_cmd],
        f"Capturing {num} {kind.upper()} frame(s) on the Pi")


def download():
    C.CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    src = f"{C.PI_USER}@{C.PI_HOST}:{C.PI_CAPTURE_DIR}/"
    # Trailing slashes: contents of captures/ land directly in RAW_DIR.
    # rsync skips files already present, so re-runs only fetch new frames.
    # Pulls every session folder (BG_* and Sample_*) down intact.
    run(["rsync", "-avh", "--progress", src, str(C.CAPTURES_DIR) + "/"],
        "Downloading session folders to the laptop")


def purge_after_download():
    """Delete Pi-side frames verified present locally with a matching size.

    Nothing is removed on trust: each file must exist locally AND match
    byte-for-byte before it is named for deletion. Deletion is one-way,
    so a mismatch means the Pi keeps its copy.
    """
    total = 0
    for local_session in sorted(C.CAPTURES_DIR.iterdir()):
        if not local_session.is_dir():
            continue
        name = local_session.name
        listing = subprocess.run(
            ["ssh", "-n", f"{C.PI_USER}@{C.PI_HOST}",
             f"cd ~/captures/{name} 2>/dev/null && stat -c '%s %n' *.tiff 2>/dev/null || true"],
            capture_output=True, text=True, timeout=30).stdout

        remote = {}
        for line in listing.splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) == 2 and parts[0].isdigit():
                remote[parts[1]] = int(parts[0])
        if not remote:
            continue

        verified = [n for n, size in remote.items()
                    if (local_session / n).exists()
                    and (local_session / n).stat().st_size == size]
        if not verified:
            continue

        for i in range(0, len(verified), 100):
            batch = " ".join(f"'{n}'" for n in verified[i:i+100])
            subprocess.run(["ssh", "-n", f"{C.PI_USER}@{C.PI_HOST}",
                            f"cd ~/captures/{name} && rm -f {batch}"],
                           capture_output=True, timeout=60)
        print(f"  purged {len(verified)} verified frame(s) from {name}")
        total += len(verified)

    if total == 0:
        print("  nothing to purge")
    return total


def process():
    print(f"\n{'='*62}\n  Processing on the laptop\n{'='*62}")
    import holo_process
    return holo_process.process_all()


def dashboard():
    print(f"\n{'='*62}\n  Launching dashboard\n{'='*62}")
    here = Path(__file__).parent / "dashboard.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(here)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-capture", action="store_true",
                   help="Do not trigger a new capture on the Pi.")
    p.add_argument("--with-bg", action="store_true",
                   help="Also capture a fresh background run before the sample. "
                        "Prompts you to empty the sample cell first.")
    p.add_argument("--bg-num", type=int, default=5,
                   help="Number of background frames when using --with-bg.")
    p.add_argument("--local-only", action="store_true",
                   help="Skip capture and download; reprocess local files.")
    p.add_argument("--no-dashboard", action="store_true",
                   help="Process only; do not launch the dashboard.")
    p.add_argument("--host", type=str, default=None,
                   help="Pi IP or hostname, overriding config.PI_HOST.")
    p.add_argument("--no-purge", dest="purge", action="store_false",
                   help="Keep frames on the Pi after downloading them.")
    p.set_defaults(purge=True)
    a = p.parse_args()

    if a.host:
        C.PI_HOST = a.host

    t0 = time.time()

    if not a.local_only:
        if not a.no_capture:
            if a.with_bg:
                input("\n>>> Empty the sample cell, then press Enter for the "
                      "BACKGROUND capture...")
                remote_capture("bg", a.bg_num)
                input("\n>>> Load the water sample, then press Enter for the "
                      "SAMPLE capture...")
            remote_capture("sample", C.CAP_NUM)
        download()
        if a.purge:
            print(f"\n{'='*62}\n  Purging downloaded frames from the Pi\n{'='*62}")
            purge_after_download()

    res = process()

    print(f"\n{'='*62}")
    print(f"  Pipeline finished in {time.time()-t0:.1f}s")
    print(f"  {res['n_frames']} frame(s), {res['total_detections']} detection(s)")
    print(f"{'='*62}")

    if not a.no_dashboard:
        dashboard()


if __name__ == "__main__":
    main()
