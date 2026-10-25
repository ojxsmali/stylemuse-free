"""Natural hair recolouring for StyleMuse Studio (no paid AI API).

Google MediaPipe hair-segmentation TFLite model runs locally in the Streamlit
process. The first launch downloads Google's public model once, without keys.
"""
from __future__ import annotations

import io
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "image_segmenter/hair_segmenter/float32/1/hair_segmenter.tflite"
)
BACKUP_MODEL_URL = (
    "https://huggingface.co/spaces/nagauta/mediapipe-hair-segmentation/"
    "resolve/main/hair_segmenter.tflite"
)

PRESETS = {
    "Espresso": "#30231E",
    "Chocolate": "#654334",
    "Chestnut": "#865038",
    "Copper": "#B65F3C",
    "Auburn": "#8E4034",
    "Honey blonde": "#C5A16A",
    "Ash blonde": "#A99B85",
    "Platinum": "#D7CDB8",
    "Burgundy": "#7E364B",
    "Rose brown": "#AB737D",
    "Midnight blue": "#344B72",
    "Silver": "#B9B8C1",
}


def load_image(raw: bytes, max_side: int = 1100) -> Image.Image:
    """Handle phone rotation and constrain memory use for free hosting."""
    if not raw or len(raw) > 12 * 1024 * 1024:
        raise ValueError("Please choose a photo smaller than 12 MB.")
    with Image.open(io.BytesIO(raw)) as source:
        source.verify()
    with Image.open(io.BytesIO(raw)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    if image.width < 96 or image.height < 96:
        raise ValueError("Please upload a larger photo (at least 96 × 96).")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def get_model_path() -> str:
    """Install public model on cold start if not included in repo."""
    local = Path(__file__).resolve().parent / "hair_segmenter.tflite"
    if local.is_file() and local.stat().st_size > 300_000:
        return str(local)

    cached = Path(tempfile.gettempdir()) / "stylemuse_models" / "hair_segmenter.tflite"
    if cached.is_file() and cached.stat().st_size > 300_000:
        return str(cached)
    cached.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    for url in (MODEL_URL, BACKUP_MODEL_URL):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "StyleMuse/2.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                model_bytes = response.read(20_000_001)
            if not 300_000 < len(model_bytes) < 20_000_000:
                raise ValueError("The downloaded model did not have the expected size.")
            tmp = cached.with_suffix(".download")
            tmp.write_bytes(model_bytes)
            tmp.replace(cached)
            return str(cached)
        except (OSError, ValueError) as exc:
            failures.append(str(exc))
    raise RuntimeError(
        "Could not fetch Google's free hair-segmentation model. "
        "Try restarting the app, or add hair_segmenter.tflite to your repo. "
        "Errors: " + "; ".join(failures)
    )


def segment_hair(rgb_image: np.ndarray, model_path: str) -> np.ndarray:
    """Return a float32 H×W probability mask for hair, not a binary stencil."""
    import mediapipe as mp

    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Expected RGB input")
    rgb_image = np.ascontiguousarray(rgb_image.astype(np.uint8, copy=False))
    options = mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        output_category_mask=False,
        output_confidence_masks=True,
    )
    with mp.tasks.vision.ImageSegmenter.create_from_options(options) as segmenter:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = segmenter.segment(image)
        if result.confidence_masks and len(result.confidence_masks) > 1:
            mask = result.confidence_masks[1].numpy_view().copy()
        elif result.category_mask is not None:
            mask = (result.category_mask.numpy_view() == 1).astype(np.float32)
        else:
            raise RuntimeError("Hair segmentation did not return a hair mask.")
    if mask.shape != rgb_image.shape[:2]:
        resized = Image.fromarray(np.uint8(np.clip(mask, 0, 1) * 255)).resize(
            (rgb_image.shape[1], rgb_image.shape[0]), Image.Resampling.BILINEAR
        )
        mask = np.asarray(resized, dtype=np.float32) / 255.0
    return np.clip(mask.astype(np.float32), 0, 1)


def smooth_mask(mask: np.ndarray, threshold: float = 0.22, feather: float = 2.5) -> np.ndarray:
    """Smooth fuzzy edges and suppress weak detections."""
    if mask.ndim != 2:
        raise ValueError("Mask must be 2-D")
    mask = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    # Soft threshold, not a harsh binary line around individual strands.
    mask = np.clip((mask - threshold) / max(0.01, 1 - threshold), 0, 1)
    if feather > 0:
        mask_img = Image.fromarray(np.uint8(mask * 255), mode="L")
        mask = np.asarray(mask_img.filter(ImageFilter.GaussianBlur(feather)), dtype=np.float32) / 255.0
    return mask


def recolor_hair(
    image: Image.Image,
    mask: np.ndarray,
    hex_color: str,
    strength: float = 0.93,
    brightness: int = 0,
) -> Image.Image:
    """Recolor segmented hair, preserving photographed strand luminance detail.

    This replaces only chromatic appearance and mean luminance in the hair
    region; it does not synthesize hair strands or change the haircut shape.
    """
    if not 0 <= strength <= 1:
        raise ValueError("strength must be 0..1")
    if not -40 <= brightness <= 40:
        raise ValueError("brightness must be between -40 and 40")
    source = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    if mask.shape != source.shape[:2]:
        raise ValueError("The hair mask and photograph must have the same dimensions")
    code = hex_color.lstrip("#")
    if len(code) != 6:
        raise ValueError("Color must be a six-digit RGB hex code")
    shade = np.array([int(code[x:x + 2], 16) for x in (0, 2, 4)], dtype=np.float32) / 255.0
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    luma = source @ weights
    shade_luma = float(shade @ weights)

    # Preserve the original pixel-to-pixel shading (real hair texture), but
    # shift its average lightness toward the selected dye's target lightness.
    weights_mask = np.clip(mask, 0, 1)
    coverage = float(weights_mask.sum())
    if coverage < 1:
        return image.convert("RGB")
    original_mean = float((luma * weights_mask).sum() / coverage)
    shifted_luma = np.clip(luma + 0.82 * (shade_luma - original_mean) + brightness / 255.0, 0.02, 0.98)

    # Lightly desaturate the chosen dye for a more photographic hair finish.
    natural_chroma = shade * 0.83 + shade_luma * 0.17
    chosen_luma = float(natural_chroma @ weights)
    colored = np.clip(shifted_luma[..., None] * natural_chroma[None, None, :] / max(0.08, chosen_luma), 0, 1)
    alpha = np.clip(weights_mask * strength, 0, 1)[..., None]
    final = np.uint8(np.clip((source * (1 - alpha) + colored * alpha) * 255, 0, 255))
    return Image.fromarray(final, mode="RGB")


def make_mask_preview(image: Image.Image, mask: np.ndarray) -> Image.Image:
    rgba = np.asarray(image.convert("RGB"), dtype=np.float32)
    overlay = np.zeros_like(rgba)
    overlay[..., 0] = 255
    overlay[..., 1] = 88
    overlay[..., 2] = 136
    alpha = np.clip(mask[..., None] * 0.58, 0, 1)
    return Image.fromarray(np.uint8(rgba * (1 - alpha) + overlay * alpha))


def to_png_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()
