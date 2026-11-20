"""
Real WhatsApp re-upload chain vs our simulated attacks.

Data (collected by hand, phone app, upload quality = Standard, "Message yourself"):
  data/whatsapp/hop0  20 originals (from the benchmark's original set, so they are in the gallery)
  data/whatsapp/hop1  hop0 sent as new photos, received copies saved
  data/whatsapp/hop2  hop1 re-uploaded the same way
  data/whatsapp/hop3  hop2 re-uploaded the same way

Steps
  1. WhatsApp renames files, so match every hop-k file back to its hop0 original:
     DINOv2 similarity among the 20 originals + one-to-one assignment (Hungarian),
     cross-checked with pHash. A contact sheet is saved for visual confirmation.
  2. Describe what WhatsApp did: resolution, file size, estimated JPEG quality, PSNR.
  3. Retrieval: query each hop against the SAME 2,500-image gallery as the main benchmark.

Usage: python src/whatsapp_eval.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

from evaluate import load_split, pessimistic_ranks
from methods import METHODS

ROOT = Path(__file__).resolve().parent.parent
WA = ROOT / "data" / "whatsapp"
RESULTS = ROOT / "results"

# Standard IJG (libjpeg) luminance table at quality 50 - used to estimate the
# quality setting WhatsApp's encoder used, from the file's own quantisation table.
IJG_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99])


def jpeg_quality(path: Path):
    """Estimate libjpeg-style quality from the luminance quantisation table.
    (Zig-zag vs raster order doesn't matter: we only use the mean ratio.)"""
    tables = getattr(Image.open(path), "quantization", None)
    if not tables:
        return None
    scale = 100 * np.mean(np.array(tables[0][:64]) / IJG_LUMA)
    q = (200 - scale) / 2 if scale <= 100 else 5000 / scale
    return float(np.clip(q, 1, 100))


def psnr(a: Image.Image, b: Image.Image) -> float:
    """Peak signal-to-noise ratio of b vs a (b resized to a's size if needed).
    ~40 dB = visually identical, ~30 dB = visible artefacts, <25 dB = heavy damage."""
    if b.size != a.size:
        b = b.resize(a.size, Image.BICUBIC)
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    mse = np.mean((x - y) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def load_dir(d: Path):
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    return files, [Image.open(p).convert("RGB") for p in files]


def main():
    hop0_files, hop0 = load_dir(WA / "hop0")
    # hop0 names look like "07_000000123456.jpg" -> original COCO file name
    hop0_coco = [p.name.split("_", 1)[1] for p in hop0_files]

    orig_files, distract_files = load_split(500, 2000, seed=0)
    gallery_names = [p.name for p in orig_files + distract_files]
    gallery_imgs = [Image.open(p).convert("RGB") for p in orig_files + distract_files]
    true_gallery_idx = np.array([gallery_names.index(n) for n in hop0_coco])

    methods = {name: cls() for name, cls in METHODS.items()}
    hops = {0: (hop0_files, hop0)}
    for h in (1, 2, 3):
        hops[h] = load_dir(WA / f"hop{h}")

    # ---- 1. match renamed WhatsApp files back to their originals ----------
    dino, ph = methods["dinov2"], methods["phash"]
    ref_d, ref_p = dino.encode(hop0), ph.encode(hop0)
    order = {0: np.arange(len(hop0))}          # order[h][j] = index in hop h of original j
    for h in (1, 2, 3):
        files, imgs = hops[h]
        sim = dino.similarity(dino.encode(imgs), ref_d)          # [20 wa, 20 orig]
        rows, cols = linear_sum_assignment(-sim)                  # maximise total similarity
        ph_best = ph.similarity(ph.encode(imgs), ref_p).argmax(axis=1)
        agree = int((ph_best[rows] == cols).sum())
        print(f"hop{h}: matched {len(rows)} files, pHash agrees on {agree}/{len(rows)}, "
              f"min DINOv2 sim of a match = {sim[rows, cols].min():.3f}")
        inv = np.empty(len(cols), dtype=int)
        inv[cols] = rows
        order[h] = inv

    # contact sheet: one row per original, columns hop0..hop3
    th = 120
    sheet = Image.new("RGB", (4 * th, len(hop0) * th), "white")
    for j in range(len(hop0)):
        for h in range(4):
            im = hops[h][1][order[h][j]].copy()
            im.thumbnail((th - 4, th - 4))
            sheet.paste(im, (h * th + 2, j * th + 2))
    ImageDraw.Draw(sheet).text((4, 4), "hop0 | hop1 | hop2 | hop3", fill="red")
    RESULTS.mkdir(exist_ok=True)
    sheet.save(RESULTS / "whatsapp_contact_sheet.jpg", quality=90)

    # ---- 2. what did WhatsApp actually do? -------------------------------
    stats = []
    for h in range(4):
        files, imgs = hops[h]
        for j in range(len(hop0)):
            k = order[h][j]
            stats.append({
                "hop": h, "orig": hop0_coco[j],
                "w": imgs[k].size[0], "h": imgs[k].size[1],
                "kb": files[k].stat().st_size / 1024,
                "jpeg_q_est": jpeg_quality(files[k]),
                "psnr_vs_orig": psnr(hop0[j], imgs[k]) if h else np.nan,
            })
    st = pd.DataFrame(stats)
    st.to_csv(RESULTS / "whatsapp_stats.csv", index=False)
    print("\nWhat WhatsApp did (means over 20 images):")
    print(st.groupby("hop")[["w", "h", "kb", "jpeg_q_est", "psnr_vs_orig"]].mean().round(1).to_string())

    # ---- 3. retrieval against the full benchmark gallery ------------------
    rows = []
    for mname, m in methods.items():
        g = m.encode(gallery_imgs)
        for h in range(4):
            imgs = [hops[h][1][order[h][j]] for j in range(len(hop0))]
            ranks = pessimistic_ranks(m.similarity(m.encode(imgs), g), true_gallery_idx)
            rows.append({"method": mname, "hop": h,
                         "R@1": float((ranks <= 1).mean()),
                         "mAP": float((1 / ranks).mean()),
                         "worst_rank": int(ranks.max())})
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "whatsapp_results.csv", index=False)
    print("\nRetrieval of real WhatsApp copies (20 queries, 2,500-image gallery):")
    print(res.pivot(index="hop", columns="method", values="R@1").to_string())


if __name__ == "__main__":
    main()
