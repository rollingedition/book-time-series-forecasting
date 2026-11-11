"""Record Chronos-2 forecasts for the book's electricity data (W-48).

Run from any directory, with the pinned runtime in recordings/requirements.txt:

    python recordings/chronos2_forecast.py

Inputs: examples/data/electricity_hourly.csv.gz (read through examples/lib/electricity.py).
Outputs, written to examples/data/recorded/chronos-2/:
    inputs.csv.gz     the exact contexts fed to the model (NaN as an empty cell)
    forecasts.csv.gz  quantile forecasts, 24 hours per client and cutoff
    manifest.json     model, revision, prompt hash, package versions, output hashes

The release service never runs this script. The examples read the recorded outputs.
"""
import gzip
import hashlib
import importlib.metadata as md
import io
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import pandas as pd
import torch

BOOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BOOK / "examples"))
from lib.electricity import CLIENTS, load  # noqa: E402

MODEL = "amazon/chronos-2"
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
SCRIPT = "recordings/chronos2_forecast.py"
OUT_DIR = BOOK / "examples" / "data" / "recorded" / "chronos-2"
CREATED = "2026-11-10"

CONTEXT_HOURS = 1344  # 8 weeks
HORIZON = 24
CUTOFFS = ["2014-12-29 00:00", "2014-12-30 00:00", "2014-12-31 00:00"]
QUANTILES = [0.025, 0.1, 0.5, 0.9, 0.975]
SEED = 0
THREADS = 2
BATCH_SIZE = 256
CLOCK_FLAGS = ("clock_change_gap", "clock_change_repeat")
NAN_NOTE = ("NaN is passed to the model as is: chronos/chronos2/model.py builds the context mask from "
            "isnan, and the scaling ignores masked values. No fill is applied.")


def gz_bytes(text):
    """Deterministic gzip: mtime 0, no filename in the header."""
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as gz:
        gz.write(text.encode("utf-8"))
    return buf.getvalue()


def build_contexts(df):
    """One context per (client, cutoff): 1,344 hours before the cutoff, start rule, clock-change NaN."""
    contexts = []
    for cutoff_s in CUTOFFS:
        cutoff = pd.Timestamp(cutoff_s)
        start = cutoff - pd.Timedelta(hours=CONTEXT_HOURS)
        for uid in CLIENTS:
            s = df[(df.unique_id == uid) & (df.ds >= start) & (df.ds < cutoff)].sort_values("ds")
            assert len(s) == CONTEXT_HOURS, (uid, cutoff_s, len(s))
            pre = s.quirk.eq("before_first_reading")
            # Start rule: a series begins at its first reading, so flagged leading rows are dropped.
            assert not pre.values[np.argmin(pre.values) :].any(), "flagged rows are not leading"
            s = s[~pre]
            y = s.y.astype(float).copy()
            y[s.quirk.isin(CLOCK_FLAGS).values] = np.nan
            contexts.append({"unique_id": uid, "cutoff": cutoff, "ds": s.ds.values, "y": y.values})
    return contexts


def write_inputs(contexts):
    rows = []
    for c in contexts:
        for ds, v in zip(c["ds"], c["y"]):
            if np.isnan(v):
                val = ""
            else:
                assert float(f"{v:.3f}") == v, "source values are not 3-decimal"
                val = f"{v:.3f}"
            rows.append(f"{c['unique_id']},{c['cutoff']:%Y-%m-%d %H:%M:%S},{pd.Timestamp(ds):%Y-%m-%d %H:%M:%S},{val}")
    return gz_bytes("unique_id,cutoff,ds,y\n" + "\n".join(rows) + "\n")


def run_model(contexts):
    from chronos import Chronos2Pipeline

    pipe = Chronos2Pipeline.from_pretrained(MODEL, revision=REVISION, device_map="cpu")
    pipe.model.eval()
    trained = [float(q) for q in pipe.quantiles]
    inputs = [{"target": c["y"].astype(np.float32)} for c in contexts]
    quants, _ = pipe.predict_quantiles(inputs, prediction_length=HORIZON, quantile_levels=QUANTILES,
                                       batch_size=BATCH_SIZE)
    return trained, [q[0].detach().cpu().numpy() for q in quants]


def write_forecasts(contexts, preds):
    rows = []
    for c, p in zip(contexts, preds):
        assert p.shape == (HORIZON, len(QUANTILES)), p.shape
        ds = pd.date_range(c["cutoff"], periods=HORIZON, freq="h")
        for h in range(HORIZON):
            vals = ",".join(f"{round(float(x), 3):.3f}" for x in p[h])
            rows.append(f"{c['unique_id']},{c['cutoff']:%Y-%m-%d %H:%M:%S},{ds[h]:%Y-%m-%d %H:%M:%S},{vals}")
    rows.sort()
    header = "unique_id,cutoff,ds,q0.025,q0.1,q0.5,q0.9,q0.975\n"
    return gz_bytes(header + "\n".join(rows) + "\n"), len(rows)


def pkg_versions():
    names = ["chronos-forecasting", "torch", "transformers", "pandas", "numpy"]
    return {n: md.version(n) for n in names}


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(THREADS)

    df = load(clients=CLIENTS)
    contexts = build_contexts(df)
    inputs_gz = write_inputs(contexts)
    n_nan = int(sum(np.isnan(c["y"]).sum() for c in contexts))

    config = {
        "model": MODEL,
        "revision": REVISION,
        "clients": "ALL30",
        "cutoffs": CUTOFFS,
        "context_hours": CONTEXT_HOURS,
        "horizon": HORIZON,
        "quantiles": QUANTILES,
        "seed": SEED,
        "threads": THREADS,
        "batch_size": BATCH_SIZE,
        "device": "cpu",
        "nan_handling": "NaN passed to the model, no fill",
        "clock_change_hours": "set to NaN",
        "start_rule": "leading rows flagged before_first_reading removed",
    }
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    prompt_sha = hashlib.sha256(inputs_gz + canonical.encode("utf-8")).hexdigest()

    trained, preds = run_model(contexts)
    forecasts_gz, n_rows = write_forecasts(contexts, preds)
    interpolated = [q for q in QUANTILES if q not in trained]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "inputs.csv.gz").write_bytes(inputs_gz)
    (OUT_DIR / "forecasts.csv.gz").write_bytes(forecasts_gz)

    outputs = {name: hashlib.sha256((OUT_DIR / name).read_bytes()).hexdigest()
               for name in ["inputs.csv.gz", "forecasts.csv.gz"]}
    manifest = {
        "script": SCRIPT,
        "model": MODEL,
        "model_version": REVISION,
        "prompt_sha256": prompt_sha,
        "created": CREATED,
        "outputs": outputs,
        "packages": pkg_versions(),
        "machine": f"a {os.cpu_count()}-vCPU cloud Linux container, CPU only, no GPU",
        "quantiles": {
            "requested": QUANTILES,
            "trained_by_model": trained,
            "predicted": [q for q in QUANTILES if q in trained],
            "interpolated": interpolated,
            "interpolation": ("chronos/utils.py interpolate_quantiles: each interpolated level is linear between "
                              "the two nearest trained levels that bracket it (the edge padding applies only "
                              "outside the trained range)"),
        },
        "rows": n_rows,
        "note": (f"prompt_sha256 is SHA-256 of inputs.csv.gz's bytes followed by the canonical JSON "
                 f"(sorted keys, no spaces) of the call configuration: {canonical}. "
                 f"Missing values: {NAN_NOTE} NaN cells in the contexts: {n_nan}. "
                 "Output is rounded to 3 decimals. "
                 "The release service never runs this script; examples read the recorded outputs."),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"rows={n_rows} interpolated={interpolated} sha_forecasts={outputs['forecasts.csv.gz'][:12]}")


if __name__ == "__main__":
    main()
