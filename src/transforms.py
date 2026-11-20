"""
"Re-share" attacks: the kinds of damage an image suffers when it is
forwarded around social media / WhatsApp.

Every attack is a function  PIL.Image -> PIL.Image.
ATTACKS maps an attack name to a list of (level_label, function) pairs,
ordered from mildest to strongest, so we can plot how each retrieval
method degrades as the attack gets harder.
"""
import io
import math
from functools import partial

from PIL import Image, ImageDraw, ImageFont, ImageOps


# ---------------------------------------------------------------- basics

def to_rgb(img: Image.Image) -> Image.Image:
    # JPEG has no alpha channel and some COCO images are greyscale,
    # so normalise everything to 3-channel RGB first.
    return img.convert("RGB")


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    """Re-encode as JPEG. Lower quality = stronger quantisation of the
    8x8 DCT blocks = more blocking/ringing artefacts and lost detail."""
    buf = io.BytesIO()
    to_rgb(img).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def downscale(img: Image.Image, factor: float) -> Image.Image:
    """Shrink each side by `factor`. Platforms do this to save bandwidth;
    fine detail (high spatial frequencies) is permanently lost."""
    w, h = img.size
    new = (max(1, round(w * factor)), max(1, round(h * factor)))
    return img.resize(new, Image.BICUBIC)


def center_crop(img: Image.Image, keep_area: float) -> Image.Image:
    """Keep the central `keep_area` fraction of the image AREA.
    Each side is scaled by sqrt(keep_area), e.g. 0.5 area -> ~0.707 side."""
    w, h = img.size
    s = math.sqrt(keep_area)
    cw, ch = round(w * s), round(h * s)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


def hflip(img: Image.Image) -> Image.Image:
    """Mirror horizontally - a common trick to dodge naive duplicate checks."""
    return ImageOps.mirror(img)


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def caption_bar(img: Image.Image, text: str = "BREAKING: missile strike footage",
                bar_frac: float = 0.18) -> Image.Image:
    """Add a black caption bar above the image, like a forwarded news post.
    This ENLARGES the canvas, so the original pixels now fill only part of it.
    TODO (later): Urdu/Hindi captions. Correct Urdu/Devanagari rendering in
    Pillow needs libraqm for text shaping - check features.check('raqm')."""
    img = to_rgb(img)
    w, h = img.size
    bar_h = max(20, round(h * bar_frac))
    canvas = Image.new("RGB", (w, h + bar_h), "black")
    canvas.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, bar_h * 0.25), text, fill="white", font=_font(round(bar_h * 0.45)))
    return canvas


def text_overlay(img: Image.Image, text: str = "VIRAL VIDEO") -> Image.Image:
    """Stamp text directly ON the image content (covers real pixels)."""
    img = to_rgb(img).copy()
    w, h = img.size
    draw = ImageDraw.Draw(img)
    size = max(12, round(h * 0.12))
    draw.text((w * 0.05, h * 0.75), text, fill="red", font=_font(size),
              stroke_width=max(1, size // 12), stroke_fill="white")
    return img


def reshare_chain(img: Image.Image, hops: int) -> Image.Image:
    """Simulate `hops` forwards. Each hop: downscale to <=1024px long side,
    then JPEG at quality 70. The first hop also adds a caption bar.
    Repeated lossy re-encoding compounds the damage (generation loss)."""
    out = to_rgb(img)
    for i in range(hops):
        w, h = out.size
        scale = min(1.0, 1024 / max(w, h))
        if scale < 1.0:
            out = downscale(out, scale)
        if i == 0:
            out = caption_bar(out)
        out = jpeg(out, 70)
    return out


# --------------------------------------------------------------- registry

ATTACKS = {
    "none":        [("orig", lambda im: to_rgb(im))],
    "jpeg":        [(f"q{q}", partial(jpeg, quality=q)) for q in (90, 70, 50, 30, 10)],
    "downscale":   [(f"x{f}", partial(downscale, factor=f)) for f in (0.75, 0.5, 0.25, 0.1)],
    "crop":        [(f"{int(a*100)}%", partial(center_crop, keep_area=a)) for a in (0.9, 0.7, 0.5, 0.3)],
    "hflip":       [("flip", hflip)],
    "caption_bar": [("bar", caption_bar)],
    "text_overlay": [("text", text_overlay)],
    "reshare":     [(f"{n}hop", partial(reshare_chain, hops=n)) for n in (1, 3, 5)],
}


if __name__ == "__main__":
    # Quick visual check: python src/transforms.py <some_image.jpg>
    # Writes one file per attack/level into results/attack_examples/.
    import sys
    from pathlib import Path

    src = Image.open(sys.argv[1])
    out_dir = Path("results/attack_examples")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, levels in ATTACKS.items():
        for label, fn in levels:
            fn(src).save(out_dir / f"{name}_{label}.jpg", quality=95)
    print(f"wrote examples to {out_dir}")
