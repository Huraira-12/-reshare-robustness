"""
Benchmark: for every attack level, can each method find the original?

Setup
  - originals:   N images sampled from COCO val2017 (fixed seed => reproducible)
  - distractors: M other COCO images, never attacked. They make the search
                 realistic: the gallery is originals + distractors.
  - queries:     attack(original) for each original.
  A query is correct if its own original is ranked highly in the gallery.

Metrics (exactly ONE correct gallery item per query)
  - Recall@k: fraction of queries whose original is in the top k.
  - mAP:      with a single relevant item, AP = 1/rank, so mAP equals the
              Mean Reciprocal Rank (MRR).
  Ties: phash scores are integers, so ties are common. We use the
  PESSIMISTIC rank (any other item scoring >= the true one counts as ahead),
  so results are never flattered by lucky tie-breaking.

Usage:
  python src/evaluate.py --methods phash --n-orig 500 --n-distract 2000
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from methods import METHODS
from transforms import ATTACKS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "val2017"
RESULTS = ROOT / "results"


def load_split(n_orig: int, n_distract: int, seed: int):
    files = sorted(DATA.glob("*.jpg"))
    random.Random(seed).shuffle(files)
    assert len(files) >= n_orig + n_distract, "not enough images"
    return files[:n_orig], files[n_orig:n_orig + n_distract]


def pessimistic_ranks(scores: np.ndarray, true_idx: np.ndarray) -> np.ndarray:
    """scores [Q,G]; true_idx [Q] = gallery index of each query's original.
    rank = 1 + number of OTHER gallery items scoring >= the true item."""
    true_scores = scores[np.arange(len(true_idx)), true_idx][:, None]
    ahead = (scores >= true_scores).sum(axis=1) - 1   # minus the true item itself
    return ahead + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["phash"])
    ap.add_argument("--n-orig", type=int, default=500)
    ap.add_argument("--n-distract", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    orig_files, distract_files = load_split(args.n_orig, args.n_distract, args.seed)
    print(f"{len(orig_files)} originals, {len(distract_files)} distractors")

    # Load once. Gallery order: originals first, so original i is gallery index i.
    originals = [Image.open(f).convert("RGB") for f in tqdm(orig_files, desc="load orig")]
    distract = [Image.open(f).convert("RGB") for f in tqdm(distract_files, desc="load distract")]
    true_idx = np.arange(len(originals))

    rows = []
    for mname in args.methods:
        method = METHODS[mname]()
        t0 = time.time()
        gallery = method.encode(originals + distract)
        print(f"[{mname}] gallery encoded in {time.time() - t0:.1f}s")

        for attack, levels in ATTACKS.items():
            for label, fn in levels:
                queries = method.encode([fn(im) for im in originals])
                ranks = pessimistic_ranks(method.similarity(queries, gallery), true_idx)
                rows.append({
                    "method": mname, "attack": attack, "level": label,
                    "R@1": float((ranks <= 1).mean()),
                    "R@5": float((ranks <= 5).mean()),
                    "mAP": float((1.0 / ranks).mean()),
                    "median_rank": float(np.median(ranks)),
                })
                r = rows[-1]
                print(f"  {attack:12s} {label:6s}  R@1={r['R@1']:.3f}  R@5={r['R@5']:.3f}  mAP={r['mAP']:.3f}")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
