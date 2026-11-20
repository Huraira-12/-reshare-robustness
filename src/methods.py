"""
Retrieval methods. Each method turns a list of images into a feature
matrix, and defines how to score a query against the gallery.

Common interface (so evaluate.py doesn't care which method it runs):
    features = method.encode(list_of_PIL_images)       -> np.ndarray [N, D]
    scores   = method.similarity(query_feats, gallery_feats) -> np.ndarray [Q, G]
Higher score = more similar.

Step 1 (this file): perceptual hash baseline.
Step 2 (YOU write next): CLIP and DINOv2 embedding methods with the same interface.
"""
import imagehash
import numpy as np


class PHash:
    """Perceptual hash (pHash).

    How it works: shrink image to 32x32 greyscale -> 2-D DCT -> keep the
    top-left 8x8 low-frequency coefficients -> each bit = coefficient above
    the median or not. 64 bits total.

    Why it's robust to JPEG/resizing: those attacks mostly destroy HIGH
    frequencies, and pHash only looks at LOW frequencies.
    Why it should fail on crops/flips/caption bars: those change the global
    layout, which shifts every low-frequency coefficient.
    """
    name = "phash"

    def __init__(self, hash_size: int = 8):
        self.hash_size = hash_size

    def encode(self, images) -> np.ndarray:
        # imagehash returns a bool matrix [hash_size, hash_size]; flatten to bits.
        return np.stack([
            imagehash.phash(im, hash_size=self.hash_size).hash.flatten()
            for im in images
        ]).astype(np.uint8)

    def similarity(self, q: np.ndarray, g: np.ndarray) -> np.ndarray:
        # Hamming distance = number of differing bits. Computed for all pairs
        # with broadcasting: [Q,1,B] vs [1,G,B] -> [Q,G]. Negate so higher = closer.
        ham = (q[:, None, :] != g[None, :, :]).sum(axis=2)
        return -ham.astype(np.float32)


class _TorchEmbed:
    """Shared logic for deep-embedding methods.

    A pretrained network maps each image to a D-dimensional vector
    ("embedding"). Images with similar content land close together.
    We L2-normalise every vector to length 1, so the dot product of two
    vectors equals their cosine similarity (1 = same direction, 0 = unrelated).
    """

    def _forward(self, batch):          # subclasses: tensor [B,3,H,W] -> [B,D]
        raise NotImplementedError

    def encode(self, images, batch_size: int = 32) -> np.ndarray:
        import torch
        feats = []
        with torch.no_grad():           # inference only: no gradients = faster, less memory
            for i in range(0, len(images), batch_size):
                batch = torch.stack([self.preprocess(im) for im in images[i:i + batch_size]])
                f = self._forward(batch)
                f = f / f.norm(dim=-1, keepdim=True)   # L2-normalise each row
                feats.append(f.float().cpu())
        return torch.cat(feats).numpy()

    def similarity(self, q: np.ndarray, g: np.ndarray) -> np.ndarray:
        return q @ g.T                  # cosine similarity, [Q,D] x [D,G] -> [Q,G]


class CLIPEmbed(_TorchEmbed):
    """CLIP ViT-B/32 (OpenAI weights), trained on ~400M image-caption pairs
    to put an image near the TEXT that describes it. So it encodes
    *semantics* ("explosion at night, city skyline"), not exact pixels."""
    name = "clip"

    def __init__(self):
        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            # OpenAI's original weights were trained with the "QuickGELU"
            # activation, so we must use the matching architecture variant.
            "ViT-B-32-quickgelu", pretrained="openai")
        self.model.eval()               # inference mode (disables dropout etc.)

    def _forward(self, batch):
        return self.model.encode_image(batch)


class DINOv2Embed(_TorchEmbed):
    """DINOv2 ViT-S/14, self-supervised: trained with NO labels or text,
    only by making different augmented views of the same image agree.
    It tends to keep more visual/structural detail than CLIP."""
    name = "dinov2"

    def __init__(self):
        from torchvision import transforms as T
        from transformers import AutoModel
        # Same ViT-S/14 weights as Meta's torch.hub release, via Hugging Face.
        self.model = AutoModel.from_pretrained("facebook/dinov2-small")
        self.model.eval()
        # 224 = 16 x 14 (ViT-S/14 cuts the image into 14x14-pixel patches).
        # Mean/std are ImageNet statistics, which DINOv2 was trained with.
        self.preprocess = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _forward(self, batch):
        # pooler_output = the layer-normed CLS token: one 384-d vector that
        # summarises the whole image.
        return self.model(pixel_values=batch).pooler_output


METHODS = {"phash": PHash, "clip": CLIPEmbed, "dinov2": DINOv2Embed}
