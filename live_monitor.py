"""
live_monitor.py — continuous capture, score, keep the best, process.

  1. Starts an unlimited capture on the Pi (background, survives disconnect)
  2. Every cycle: rsync new frames down, score them, keep the best N
  3. Reconstructs only the keepers, appending to results.json
  4. Dashboard picks up new results on refresh

    python live_monitor.py                  # run until Ctrl-C
    python live_monitor.py --cycles 5       # stop after 5 cycles
    python live_monitor.py --keep 3         # reconstruct 3 best per cycle
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import tifffile

import config as C
import holo_process as HP


# ───────────────────────────────────────────────────────────────────────
#  Frame quality
# ───────────────────────────────────────────────────────────────────────
def score_frame(path):
    """Cheap quality score for a raw hologram. Higher is better.

    Fringe contrast is what matters: a good inline hologram has strong
    high-frequency interference structure. A blurred, empty or clipped
    frame does not. Laplacian variance measures exactly that, and runs
    in milliseconds — so we can score everything and reconstruct little.
    """
    img = tifffile.imread(path).astype(np.float32)
    full_scale = float(img.max()) if img.max() > 1023 else 1023.0

    sat_frac = float((img >= 65472).mean())
    mean_fs  = float(img.mean()) / 65472.0

    # 4-neighbour Laplacian, no OpenCV needed
    lap = (img[:-2, 1:-1] + img[2:, 1:-1] +
           img[1:-1, :-2] + img[1:-1, 2:] - 4 * img[1:-1, 1:-1])
    contrast = float(lap.var())

    # Penalise clipped frames hard: clipped pixels destroy the division
    # by the background, and no amount of fringe contrast makes up for it.
    penalty = 1.0 - min(sat_frac * 20.0, 1.0)

    # Penalise frames that are too dark or too bright to carry signal
    if   mean_fs < 0.05: penalty *= 0.2
    elif mean_fs > 0.90: penalty *= 0.2

    return {
        "score": contrast * penalty,
        "contrast": contrast,
        "sat_frac": round(sat_frac, 5),
        "mean_fs": round(mean_fs, 4),
    }


# ───────────────────────────────────────────────────────────────────────
#  Pi control
# ───────────────────────────────────────────────────────────────────────
def ssh(cmd, check=True):
    # -n redirects stdin from /dev/null so ssh never waits on the terminal.
    r = subprocess.run(["ssh", "-n", f"{C.PI_USER}@{C.PI_HOST}", cmd],
                       capture_output=True, text=True, timeout=30)
    if check and r.returncode != 0:
        sys.exit(f"SSH failed: {r.stderr.strip()}")
    return r.stdout.strip()


def ssh_launch(cmd):
    """Fire a remote command and do not wait for it at all.

    ssh can keep a channel open for a backgrounded remote process no
    matter how its streams are redirected, so any call that waits risks
    hanging. We do not need the return value: whether the capture really
    started is confirmed afterwards with a separate pgrep, which is the
    honest check anyway.
    """
    p = subprocess.Popen(
        ["ssh", "-n", f"{C.PI_USER}@{C.PI_HOST}", cmd],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return p


def start_capture(session, shutter, gain, interval, settle=2.0):
    """Launch an unlimited capture that keeps running after we disconnect."""
    # The bracket stops the pattern matching the shell running it: the
    # literal string "capture_series.py" appears in this very command line,
    # so an unbracketed pkill -f kills its own SSH session and we hang.
    ssh("pkill -f '[c]apture_series.py' || true", check=False)
    time.sleep(1)
    # setsid puts the capture in its own session so it outlives this
    # connection; ssh_launch avoids the pipes that would otherwise keep
    # ssh waiting on it.
    cmd = (f"cd ~ && setsid nohup python3 {C.PI_SCRIPT} "
           f"--kind sample --session {session} --num 0 "
           f"--interval {interval} --shutter {shutter} --gain {gain} "
           f"--settle {settle} "
           f"< /dev/null > ~/capture_live.log 2>&1 &")
    launcher = ssh_launch(cmd)

    # Poll for the process rather than trusting the launch call.
    alive = ""
    for _ in range(10):
        time.sleep(2)
        alive = ssh("pgrep -f '[c]apture_series.py' | head -1", check=False)
        if alive:
            break
    if not alive:
        log = ssh("tail -30 ~/capture_live.log", check=False)
        sys.exit(f"Capture failed to start on the Pi. Log:\n{log}")

    print(f"Capture running on the Pi (pid {alive}) -> ~/captures/{session}")
    return launcher


def stop_capture():
    ssh("pkill -f '[c]apture_series.py' || true", check=False)
    print("Capture stopped on the Pi.")


def sync(session):
    src = f"{C.PI_USER}@{C.PI_HOST}:~/captures/{session}/"
    dst = C.CAPTURES_DIR / session
    dst.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-az", "--exclude", "metadata.csv", src, str(dst) + "/"],
                   check=False)
    return dst


def purge_remote(session, keep_recent=5):
    """Delete Pi-side frames that are verified present locally, same size.

    Deletion is one-way, so nothing is removed on trust. Each candidate
    must exist locally AND match byte-for-byte in size before it is
    named for deletion. The newest few frames are always kept: one of
    them may still be mid-write, and a half-written file would compare
    as a size mismatch anyway.

    Only individual .tiff files are removed. The session folder,
    metadata.csv, and every BG_* folder are left alone.
    """
    local_dir = C.CAPTURES_DIR / session
    if not local_dir.exists():
        return 0

    # Remote inventory: "size name", one per line.
    listing = ssh(f"cd ~/captures/{session} 2>/dev/null && "
                  f"stat -c '%s %n' *.tiff 2>/dev/null || true", check=False)
    if not listing:
        return 0

    remote = {}
    for line in listing.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0].isdigit():
            remote[parts[1]] = int(parts[0])

    if len(remote) <= keep_recent:
        return 0

    # Never touch the newest few; sorted names are chronological here.
    candidates = sorted(remote)[:-keep_recent]

    verified = []
    for name in candidates:
        lf = local_dir / name
        if lf.exists() and lf.stat().st_size == remote[name]:
            verified.append(name)

    if not verified:
        return 0

    # Delete in batches so the command line cannot overflow.
    for i in range(0, len(verified), 100):
        batch = " ".join(f"'{n}'" for n in verified[i:i+100])
        ssh(f"cd ~/captures/{session} && rm -f {batch}", check=False)

    return len(verified)


# ───────────────────────────────────────────────────────────────────────
#  Main loop
# ───────────────────────────────────────────────────────────────────────
def append_results(new_frames, session, bg_session, seen=0, kept=0):
    """Merge newly processed frames into results.json.

    seen/kept are counts, but older call sites passed collections. Coerce
    so either works rather than failing mid-run and losing the cycle.
    """
    if not isinstance(seen, int):
        seen = len(seen) if hasattr(seen, "__len__") else 0
    if not isinstance(kept, int):
        kept = len(kept) if hasattr(kept, "__len__") else 0
    if C.RESULTS_JSON.exists():
        with open(C.RESULTS_JSON) as f:
            res = json.load(f)
    else:
        res = {"frames": [], "params": {}}

    seen = {f["frame"] for f in res["frames"]}
    res["frames"].extend(f for f in new_frames if f["frame"] not in seen)

    all_d = [d["diameter_mm"] for fr in res["frames"] for d in fr["detections"]]
    res.update({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "frames_captured": int(res.get("frames_captured") or 0) + seen,
        "frames_processed": len(res["frames"]),
        "sample_session": session,
        "bg_session": bg_session,
        "n_frames": len(res["frames"]),
        "n_failed": 0,
        "failures": [],
        "total_detections": len(all_d),
        "mean_diameter_mm": round(float(np.mean(all_d)), 5) if all_d else None,
        "params": {
            "wavelength_m": C.WAVELENGTH, "pixel_size_m": C.PIXEL_SIZE,
            "z_near_m": C.Z_NEAR, "z_far_m": C.Z_FAR,
            "min_blob_area": C.MIN_BLOB_AREA, "max_blob_area": C.MAX_BLOB_AREA,
        },
    })

    C.RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(C.RESULTS_JSON, "w") as f:
        json.dump(res, f, indent=2)
    return res


def write_session_record(session):
    """Write one self-contained CSV + JSON for this session only.

    Each session gets its own folder, so runs never mix. Written after
    every cycle (not just at the end) so an interrupted run still leaves
    a complete record of what it processed.
    """
    if not C.RESULTS_JSON.exists():
        return None
    with open(C.RESULTS_JSON) as f:
        res = json.load(f)

    frames = [fr for fr in res.get("frames", [])]
    if not frames:
        return None

    sdir = C.SESSIONS_DIR / session
    sdir.mkdir(parents=True, exist_ok=True)

    csv_path = sdir / f"{session}_particles.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "frame", "particle_id", "date", "time",
                    "x_mm", "y_mm", "z_mm", "diameter_mm", "diameter_um",
                    "size_class", "area_px"])
        for fr in frames:
            # Frame names carry the capture timestamp: NAME_IIII_YYYYMMDD_HHMMSS_mmm
            parts = fr["frame"].split("_")
            date = time_ = ""
            if len(parts) >= 4:
                date, time_ = parts[-3], parts[-2]
                date = f"{date[:4]}-{date[4:6]}-{date[6:]}" if len(date) == 8 else date
                time_ = f"{time_[:2]}:{time_[2:4]}:{time_[4:]}" if len(time_) == 6 else time_
            for i, d in enumerate(fr["detections"], 1):
                um = d["diameter_mm"] * 1000.0
                w.writerow([session, fr["frame"], i, date, time_,
                            d["x_mm"], d["y_mm"], d["z_mm"],
                            d["diameter_mm"], round(um, 2),
                            C.SIZE_LABELS[C.size_class_index(um)], d["area_px"]])

    summary = {
        "session": session,
        "written_at": datetime.now().isoformat(timespec="seconds"),
        "bg_session": res.get("bg_session"),
        "frames_processed": len(frames),
        "frames_captured": res.get("frames_captured"),
        "total_particles": sum(len(fr["detections"]) for fr in frames),
        "params": res.get("params", {}),
        "per_frame": [{"frame": fr["frame"], "n": len(fr["detections"])}
                      for fr in frames],
    }
    with open(sdir / f"{session}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    return csv_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cycles", type=int, default=0, help="0 = run until Ctrl-C")
    p.add_argument("--keep", type=int, default=2,
                   help="Frames to reconstruct per cycle (best-scoring).")
    p.add_argument("--cycle-seconds", type=float, default=30.0,
                   help="Seconds of capture to collect before each processing pass.")
    p.add_argument("--shutter", type=int, default=C.CAP_SHUTTER)
    p.add_argument("--gain", type=float, default=C.CAP_GAIN)
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--settle", type=float, default=2.0,
                   help="Seconds the camera runs before the first frame is "
                        "kept. Raise it if early frames look worse than later "
                        "ones - exposure and gain need time to latch.")
    p.add_argument("--session", type=str, default=None)
    p.add_argument("--host", type=str, default=None,
                   help="Pi IP or hostname, overriding config.PI_HOST. "
                        "Useful because DHCP reassigns the address on "
                        "every new network.")
    p.add_argument("--no-capture", action="store_true",
                   help="Do not start a capture; just watch an existing folder.")
    p.add_argument("--no-purge", dest="purge_remote", action="store_false",
                   help="Keep frames on the Pi. By default they are deleted "
                        "after each sync, once verified present locally with "
                        "a matching byte size.")
    p.set_defaults(purge_remote=True)
    a = p.parse_args()

    # Override before anything reads C.PI_HOST.
    if a.host:
        C.PI_HOST = a.host
    print(f"Pi         : {C.PI_USER}@{C.PI_HOST}")

    session = a.session or f"Sample_live_{datetime.now().strftime('%d_%m_%y_%H%M')}"
    bg_dir  = C.bg_dir()
    print(f"Background : {bg_dir.name}")

    C.OUT_MIP.mkdir(parents=True, exist_ok=True)
    C.OUT_DET.mkdir(parents=True, exist_ok=True)
    background = HP.load_background(bg_dir)
    z_depths   = np.arange(C.Z_NEAR, C.Z_FAR + C.Z_STEP, C.Z_STEP)
    z_mm       = z_depths * 1000
    print(f"Z-scan     : {len(z_depths)} planes\n")

    if not a.no_capture:
        start_capture(session, a.shutter, a.gain, a.interval, a.settle)

    processed, cycle = set(), 0
    try:
        while a.cycles == 0 or cycle < a.cycles:
            cycle += 1
            print(f"--- cycle {cycle}: collecting {a.cycle_seconds:.0f}s ---")
            time.sleep(a.cycle_seconds)

            local = sync(session)
            if a.purge_remote:
                n = purge_remote(session)
                if n:
                    print(f"  purged {n} verified frame(s) from the Pi")
            files = [f for f in HP.list_tiffs(local) if f.name not in processed]
            if not files:
                print("  no new frames yet")
                continue

            scored = []
            for f in files:
                try:
                    scored.append((f, score_frame(f)))
                except Exception as e:
                    print(f"  skip {f.name}: {e}")
            scored.sort(key=lambda t: t[1]["score"], reverse=True)

            print(f"  {len(scored)} new frame(s); reconstructing best {a.keep}")
            keepers = scored[:a.keep]
            for f, s in scored[a.keep:]:
                processed.add(f.name)          # scored, deliberately not reconstructed

            new = []
            for f, s in keepers:
                print(f"  {f.name}  contrast={s['contrast']:.0f} "
                      f"sat={s['sat_frac']*100:.2f}% mean={s['mean_fs']*100:.0f}%FS")
                try:
                    t0 = time.time()
                    r  = HP.process_frame(f, background, z_depths, z_mm)
                    r["quality"] = s
                    new.append(r)
                    print(f"    -> {r['n_detected']} object(s) in {time.time()-t0:.1f}s")
                except Exception as e:
                    print(f"    FAILED: {e}")
                processed.add(f.name)

            if new:
                res = append_results(new, session, bg_dir.name,
                                     seen=len(scored), kept=len(new))
                print(f"  total: {res['frames_processed']} processed / "
                      f"{res['frames_captured']} captured, "
                      f"{res['total_detections']} detections")
                rec = write_session_record(session)
                if rec:
                    print(f"  session record -> {rec}")

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        if not a.no_capture:
            stop_capture()
        rec = write_session_record(session)
        if rec:
            print(f"\nSession record written:\n  {rec}")
            print(f"  {rec.parent}")


if __name__ == "__main__":
    main()
