#!/usr/bin/env python3
"""
capture_series.py

Timed raw capture for a monochrome IMX296 (Raspberry Pi Global Shutter Camera).

- Saves UNPROCESSED sensor data as 16-bit single-channel TIFF.
- No JPG, no debayer, no gamma, no white balance, no denoise, no auto-exposure.
- Exposure / gain / interval / count are all set from the command line.
- Interval is scheduled against an absolute clock so it does not drift.

Typical use over SSH:
    python3 capture_series.py --num 50 --interval 2.0 --shutter 3000 --gain 1.0

Long runs (survives the SSH session dropping):
    nohup python3 capture_series.py --num 500 --interval 5 > run.log 2>&1 &
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime

import numpy as np
import tifffile
from picamera2 import Picamera2


def parse_args():
    p = argparse.ArgumentParser(
        description="Capture a timed series of raw mono TIFFs from the IMX296.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num", type=int, default=10,
                   help="Number of images to capture. Use 0 for unlimited (Ctrl-C to stop).")
    p.add_argument("--interval", type=float, default=1.0,
                   help="Seconds between the START of consecutive captures.")
    p.add_argument("--shutter", type=int, default=1500,
                   help="Exposure time in microseconds.")
    p.add_argument("--gain", type=float, default=1.0,
                   help="Analogue gain. 1.0 = no amplification.")
    p.add_argument("--frame-duration", type=int, default=None,
                   help="Frame duration in microseconds. Caps the maximum exposure. "
                        "Default: auto (shutter + 2000us, minimum 33333us).")
    p.add_argument("--kind", type=str, default="sample",
                   choices=["sample", "bg"],
                   help="What this run is. 'bg' = background/no-sample reference, "
                        "'sample' = actual water sample. Sets the folder name.")
    p.add_argument("--basedir", type=str,
                   default=os.path.join(os.path.expanduser("~"), "captures"),
                   help="Parent directory that dated session folders are created in.")
    p.add_argument("--session", type=str, default=None,
                   help="Session folder name, overriding the auto-generated "
                        "'BG_DD_MM_YY' / 'Sample_DD_MM_YY'.")
    p.add_argument("--outdir", type=str, default=None,
                   help="Full output path, overriding --basedir and --session entirely.")
    p.add_argument("--prefix", type=str, default=None,
                   help="Filename prefix. Default: matches --kind (BG or Sample).")
    p.add_argument("--settle", type=float, default=2.0,
                   help="Seconds to wait after starting the camera before the first frame.")
    p.add_argument("--width", type=int, default=None,
                   help="Optional sensor width override (must match a real sensor mode).")
    p.add_argument("--height", type=int, default=None,
                   help="Optional sensor height override (must match a real sensor mode).")
    p.add_argument("--bit-depth", type=int, default=None,
                   help="Preferred sensor bit depth, e.g. 10 or 12. Default: highest available.")
    p.add_argument("--list-modes", action="store_true",
                   help="Print the available sensor modes and exit.")
    p.add_argument("--list-controls", action="store_true",
                   help="Print the controls this sensor advertises and exit.")
    p.add_argument("--no-log", action="store_true",
                   help="Do not write metadata.csv.")
    return p.parse_args()


def pick_sensor_mode(picam2, args):
    """Choose a sensor mode and return it with its UNPACKED raw format."""
    modes = picam2.sensor_modes
    if not modes:
        sys.exit("No sensor modes reported. Is the camera detected? Try: libcamera-hello --list-cameras")

    candidates = modes

    if args.width and args.height:
        want = (args.width, args.height)
        candidates = [m for m in candidates if tuple(m["size"]) == want]
        if not candidates:
            sizes = sorted({tuple(m["size"]) for m in modes})
            sys.exit(f"No sensor mode with size {want}. Available: {sizes}")

    if args.bit_depth:
        exact = [m for m in candidates if m.get("bit_depth") == args.bit_depth]
        if not exact:
            depths = sorted({m.get("bit_depth") for m in candidates})
            sys.exit(f"No sensor mode with bit depth {args.bit_depth}. Available: {depths}")
        candidates = exact

    # Prefer the deepest bit depth, then the largest area.
    mode = max(candidates,
               key=lambda m: (m.get("bit_depth", 0), m["size"][0] * m["size"][1]))

    # "unpacked" is the key bit: packed formats (R10_CSI2P) squeeze 4 pixels
    # into 5 bytes and are meaningless if written straight to a TIFF.
    raw_format = mode.get("unpacked") or mode["format"]
    return mode, raw_format


def resolve_outdir(args):
    """Work out where frames go, and what to call them.

    Precedence: --outdir wins outright; otherwise --session names the folder
    inside --basedir; otherwise the folder is auto-named from --kind and
    today's date, e.g. BG_03_09_26 or Sample_03_09_26.
    """
    tag = "BG" if args.kind == "bg" else "Sample"
    prefix = args.prefix or tag

    if args.outdir:
        return args.outdir, prefix

    session = args.session or f"{tag}_{datetime.now().strftime('%d_%m_%y')}"
    return os.path.join(args.basedir, session), prefix


def to_uint16(arr, width, bit_depth):
    """Turn the raw buffer into a clean (height, width) uint16 array."""
    if arr.ndim == 3:                       # (h, w, 1) -> (h, w)
        arr = arr[:, :, 0]
    if arr.dtype == np.uint8 and bit_depth > 8:
        # Buffer arrives as bytes; reinterpret as little-endian 16-bit words.
        arr = arr.view(np.uint16)
    # Rows are padded to a hardware stride; drop the padding columns.
    if arr.shape[1] > width:
        arr = arr[:, :width]
    return np.ascontiguousarray(arr)


def main():
    args = parse_args()

    picam2 = Picamera2()

    if args.list_modes:
        for i, m in enumerate(picam2.sensor_modes):
            print(f"[{i}] size={tuple(m['size'])} format={m['format']} "
                  f"unpacked={m.get('unpacked')} bit_depth={m.get('bit_depth')} "
                  f"fps={m.get('fps')}")
        picam2.close()
        return

    if args.list_controls:
        for name, limits in sorted(picam2.camera_controls.items()):
            print(f"{name}: {limits}")
        picam2.close()
        return

    outdir, prefix = resolve_outdir(args)
    os.makedirs(outdir, exist_ok=True)

    mode, raw_format = pick_sensor_mode(picam2, args)
    raw_size = tuple(mode["size"])
    bit_depth = mode.get("bit_depth", 10)

    # The main (ISP) stream is unused — we never save it — so keep it tiny
    # to save memory and bandwidth. Only the raw stream matters.
    config = picam2.create_still_configuration(
        main={"size": (640, 480)},
        raw={"format": raw_format, "size": raw_size},
        buffer_count=2,
        queue=False,
    )
    picam2.configure(config)

    # Everything auto is switched off so successive frames are comparable.
    # A mono sensor's tuning file has no white-balance block, so libcamera
    # never advertises AwbEnable — send only what this sensor reports.
    #
    # FrameDurationLimits matters: exposure cannot exceed the frame period.
    # The IMX296 defaults to (33333, 33333) = 30fps, which silently clamps any
    # shutter above ~33ms. Widen it to fit the requested exposure.
    exp_limits = picam2.camera_controls.get("ExposureTime")
    if exp_limits and args.shutter > exp_limits[1]:
        print(f"WARNING: --shutter {args.shutter}us exceeds the sensor maximum "
              f"({exp_limits[1]}us) and will be clamped.")

    frame_us = args.frame_duration
    if frame_us is None:
        # Headroom for readout overhead, and never shorter than the mode default.
        frame_us = max(33333, args.shutter + 2000)

    wanted = {
        "ExposureTime": args.shutter,
        "AnalogueGain": args.gain,
        "ExposureTimeMode": 1,      # 1 = Manual
        "AnalogueGainMode": 1,      # 1 = Manual
        "AeEnable": False,          # older libcamera path, harmless alongside
        "AwbEnable": False,
        "FrameDurationLimits": (frame_us, frame_us),
    }
    available = picam2.camera_controls
    controls = {k: v for k, v in wanted.items() if k in available}
    skipped = [k for k in wanted if k not in available]
    if skipped:
        print(f"Note: not advertised by this sensor, skipping: {', '.join(skipped)}")
    picam2.set_controls(controls)

    picam2.start()
    time.sleep(args.settle)          # let exposure/gain actually latch

    actual = picam2.capture_metadata()
    print(f"Sensor mode : {raw_size[0]}x{raw_size[1]}  {raw_format}  {bit_depth}-bit")
    print(f"Requested   : shutter={args.shutter}us gain={args.gain}")
    print(f"Applied     : shutter={actual.get('ExposureTime')}us "
          f"analogue_gain={actual.get('AnalogueGain')} "
          f"digital_gain={actual.get('DigitalGain')} "
          f"frame_duration={actual.get('FrameDuration')}us")
    print(f"Interval    : {args.interval}s")
    print(f"Run type    : {args.kind.upper()}")
    print(f"Output      : {outdir}")
    print(f"Frames      : {'unlimited' if args.num == 0 else args.num}\n")

    log_file = None
    log_writer = None
    if not args.no_log:
        log_path = os.path.join(outdir, "metadata.csv")
        new_file = not os.path.exists(log_path)
        log_file = open(log_path, "a", newline="")
        log_writer = csv.writer(log_file)
        if new_file:
            log_writer.writerow(["filename", "wall_clock", "sensor_timestamp_ns",
                                 "exposure_us", "analogue_gain", "digital_gain",
                                 "bit_depth", "min", "max", "mean"])

    t0 = time.monotonic()
    i = 0
    try:
        while args.num == 0 or i < args.num:
            # Absolute schedule: frame i fires at t0 + i*interval, so a slow
            # disk write does not push every later frame further behind.
            target = t0 + i * args.interval
            wait = target - time.monotonic()
            if wait > 0:
                time.sleep(wait)

            request = picam2.capture_request()
            try:
                arr = request.make_array("raw")     # MUST happen before release()
                meta = request.get_metadata()
            finally:
                request.release()

            img = to_uint16(arr, raw_size[0], bit_depth)

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            fname = f"{prefix}_{i:04d}_{stamp}.tiff"
            fpath = os.path.join(outdir, fname)

            # No compression, no photometric guessing, no scaling.
            tifffile.imwrite(
                fpath,
                img,
                photometric="minisblack",
                compression=None,
                metadata=None,
                description=(f"raw_format={raw_format} bit_depth={bit_depth} "
                             f"exposure_us={meta.get('ExposureTime')} "
                             f"analogue_gain={meta.get('AnalogueGain')} "
                             f"digital_gain={meta.get('DigitalGain')}"),
            )

            lo, hi, mean = int(img.min()), int(img.max()), float(img.mean())
            sat = " *** SATURATED ***" if hi >= (2 ** bit_depth) - 1 else ""
            print(f"[{i:04d}] {fname}  min={lo} max={hi} mean={mean:.1f}{sat}")

            if log_writer:
                log_writer.writerow([fname, datetime.now().isoformat(timespec="milliseconds"),
                                     meta.get("SensorTimestamp"), meta.get("ExposureTime"),
                                     meta.get("AnalogueGain"), meta.get("DigitalGain"),
                                     bit_depth, lo, hi, f"{mean:.3f}"])
                log_file.flush()

            i += 1

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if log_file:
            log_file.close()
        picam2.stop()
        picam2.close()
        print(f"Done. {i} frame(s) written to {outdir}")


if __name__ == "__main__":
    main()
