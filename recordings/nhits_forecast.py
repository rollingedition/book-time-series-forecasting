"""Record NHITS forecasts for the book's electricity data (W-48, B-7).

Run from any directory, with the pinned runtime in recordings/requirements-nhits.txt:

    python recordings/nhits_forecast.py

Inputs: examples/data/recorded/chronos-2/inputs.csv.gz, the exact contexts fed to Chronos-2
(30 clients, 1,344 hours before each standard cutoff; NaN as an empty cell).
Outputs, written to examples/data/recorded/nhits/:
    forecasts.csv.gz  quantile forecasts, 24 hours per client and cutoff
    manifest.json     model, model version, prompt hash, package versions, output hash

One NHITS model is trained per cutoff, on the 30 contexts of that cutoff. Its weights are trained
here and not shipped. The release service never runs this script; the examples read the outputs.
"""
import gzip
import hashlib
import importlib.metadata as md
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BOOK = Path(__file__).resolve().parent.parent
INPUTS = BOOK / "examples" / "data" / "recorded" / "chronos-2" / "inputs.csv.gz"
OUT_DIR = BOOK / "examples" / "data" / "recorded" / "nhits"
SCRIPT = "recordings/nhits_forecast.py"
CREATED = "2026-11-10"

from neuralforecast import NeuralForecast  # noqa: E402
from neuralforecast.losses.pytorch import MQLoss, quantiles_to_outputs  # noqa: E402
from neuralforecast.models import NHITS  # noqa: E402

HORIZON = 24
INPUT_SIZE = 168  # one week of hourly history
QUANTILES = [0.025, 0.1, 0.5, 0.9, 0.975]  # the same levels as the Chronos-2 columns
CUTOFFS = ["2014-12-29 00:00", "2014-12-30 00:00", "2014-12-31 00:00"]
MAX_STEPS = 500
BATCH_SIZE = 32
SCALER = "standard"  # per-series z-score from each input window (neuralforecast's "standard")
SEED = 0
THREADS = 2
DECIMALS = 3


def gz_bytes(text):
    """Deterministic gzip: mtime 0, no filename in the header."""
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as gz:
        gz.write(text.encode("utf-8"))
    return buf.getvalue()


def fill_missing(ctx_y):
    """B-2's rule: each NaN becomes the mean of the hour before and after, from this context only.

    The neighbours are read from the context as it was given (before any fill), and only the
    non-missing ones count. Returns the filled array and the number of hours filled.
    """
    y = ctx_y.copy()
    missing = np.flatnonzero(np.isnan(ctx_y))
    for i in missing:
        nb = [ctx_y[j] for j in (i - 1, i + 1) if 0 <= j < len(ctx_y) and not np.isnan(ctx_y[j])]
        if not nb:
            raise ValueError("a missing hour has no reading on either side in its context")
        y[i] = float(np.mean(nb))
    return y, len(missing)


def read_contexts(raw):
    """Long frame of the Chronos-2 contexts, with the missing hours filled. Returns (frame, hours filled)."""
    df = pd.read_csv(io.BytesIO(gzip.decompress(raw)), dtype={"unique_id": str, "cutoff": str, "ds": str})
    df["ds"] = pd.to_datetime(df["ds"])
    df["cutoff"] = pd.to_datetime(df["cutoff"])
    assert sorted(df["cutoff"].dt.strftime("%Y-%m-%d %H:%M").unique()) == sorted(CUTOFFS)
    parts, n_filled = [], 0
    for (uid, cutoff), g in df.groupby(["unique_id", "cutoff"], sort=False):
        g = g.sort_values("ds").reset_index(drop=True)
        assert len(g) == INPUT_SIZE * 8, (uid, cutoff, len(g))  # 1,344 hours per context
        y, k = fill_missing(g["y"].to_numpy(float))
        n_filled += k
        parts.append(pd.DataFrame({"unique_id": uid, "cutoff": cutoff, "ds": g["ds"], "y": y}))
    return pd.concat(parts, ignore_index=True), n_filled


def fit_and_predict(frame, cutoff):
    """Train one NHITS on the contexts of one cutoff and return its quantile forecasts."""
    _, names = quantiles_to_outputs(QUANTILES)
    columns = [f"NHITS{n}" for n in names]
    train = frame[frame["cutoff"] == cutoff][["unique_id", "ds", "y"]].copy()
    model = NHITS(
        h=HORIZON,
        input_size=INPUT_SIZE,
        loss=MQLoss(quantiles=QUANTILES),
        max_steps=MAX_STEPS,
        batch_size=BATCH_SIZE,
        scaler_type=SCALER,
        random_seed=SEED,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    nf = NeuralForecast(models=[model], freq="h")
    t0 = time.perf_counter()
    nf.fit(df=train)
    elapsed = time.perf_counter() - t0
    fc = nf.predict()
    assert all(c in fc.columns for c in columns), (columns, list(fc.columns))
    return fc, columns, elapsed


def hparams():
    return {
        "h": HORIZON,
        "input_size": INPUT_SIZE,
        "loss": "MQLoss",
        "quantiles": QUANTILES,
        "max_steps": MAX_STEPS,
        "batch_size": BATCH_SIZE,
        "scaler_type": SCALER,
        "random_seed": SEED,
        "learning_rate": 1e-3,
        "note": "other NHITS arguments at neuralforecast 3.3.0 defaults",
    }


def main():
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(THREADS)

    raw = INPUTS.read_bytes()
    frame, n_filled = read_contexts(raw)
    clients = sorted(frame["unique_id"].unique())
    assert len(clients) == 30, len(clients)

    nf_version = md.version("neuralforecast")
    config = {
        "model": "NHITS (neuralforecast)",
        "model_version": nf_version,
        "hyperparameters": hparams(),
        "clients": "ALL30",
        "cutoffs": CUTOFFS,
        "context_hours": INPUT_SIZE * 8,
        "horizon": HORIZON,
        "quantiles": QUANTILES,
        "seed": SEED,
        "threads": THREADS,
        "device": "cpu",
        "nan_handling": "clock-change NaN filled as the mean of the hours either side, from the context only",
        "start_rule": "leading rows flagged before_first_reading removed (already applied in inputs)",
    }
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    prompt_sha = hashlib.sha256(raw + canonical.encode("utf-8")).hexdigest()

    out_rows, timings = [], {}
    for cutoff_s in CUTOFFS:
        cutoff = pd.Timestamp(cutoff_s)
        fc, columns, elapsed = fit_and_predict(frame, cutoff)
        timings[cutoff_s] = round(elapsed, 1)
        fc = fc.sort_values(["unique_id", "ds"]).reset_index(drop=True)
        assert fc.groupby("unique_id").size().eq(HORIZON).all()
        assert fc.groupby("unique_id")["ds"].first().eq(cutoff).all(), "forecasts must start at the cutoff"
        vals = fc[columns].to_numpy(float)
        assert np.isfinite(vals).all()
        for (_, row), v in zip(fc.iterrows(), vals):
            ds = row["ds"]
            cells = ",".join(f"{round(float(x), DECIMALS):.{DECIMALS}f}" for x in v)
            out_rows.append(f"{row['unique_id']},{cutoff:%Y-%m-%d %H:%M:%S},{ds:%Y-%m-%d %H:%M:%S},{cells}")

    out_rows.sort()
    header = "unique_id,cutoff,ds,q0.025,q0.1,q0.5,q0.9,q0.975\n"
    forecasts_gz = gz_bytes(header + "\n".join(out_rows) + "\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "forecasts.csv.gz").write_bytes(forecasts_gz)
    outputs = {"forecasts.csv.gz": hashlib.sha256(forecasts_gz).hexdigest()}

    _, names = quantiles_to_outputs(QUANTILES)
    manifest = {
        "script": SCRIPT,
        "model": "NHITS (neuralforecast)",
        "model_version": f"neuralforecast {nf_version}; NHITS " + json.dumps(hparams(), sort_keys=True,
                                                                           separators=(",", ":")),
        "prompt_sha256": prompt_sha,
        "created": CREATED,
        "outputs": outputs,
        "packages": {n: md.version(n) for n in ["neuralforecast", "torch", "pytorch-lightning", "pandas", "numpy"]},
        "machine": f"a {os.cpu_count()}-vCPU cloud Linux container, CPU only, no GPU",
        "quantiles": {
            "requested": QUANTILES,
            "mapping": {str(q): f"NHITS{n}" for q, n in zip(QUANTILES, names)},
            "mapping_source": "neuralforecast.losses.pytorch.quantiles_to_outputs: MQLoss(quantiles=...) trains "
                              "each level directly; no interpolation",
        },
        "rows": len(out_rows),
        "note": (f"prompt_sha256 is SHA-256 of inputs.csv.gz's bytes (the Chronos-2 contexts, "
                 f"examples/data/recorded/chronos-2/inputs.csv.gz) followed by the canonical JSON "
                 f"(sorted keys, no spaces) of the call configuration: {canonical}. "
                 f"Missing values: {n_filled} hours were NaN in the contexts (clock-change hours) and were filled, "
                 "each as the mean of the hour before and after from that context only (B-2), because NHITS cannot "
                 "take NaN. "
                 "The weights are trained here, one model per cutoff, and are not shipped. "
                 f"Scaler: neuralforecast '{SCALER}'. Output is rounded to {DECIMALS} decimals. "
                 "The release service never runs this script; examples read the recorded outputs."),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    # Timings go to stdout for the report only, never into a file.
    print(f"rows={len(out_rows)} filled={n_filled} sha_forecasts={outputs['forecasts.csv.gz'][:12]} "
          f"train_s={timings}", file=sys.stderr)


if __name__ == "__main__":
    main()
