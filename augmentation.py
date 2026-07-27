#!/usr/bin/env python3

import math
import random
from io import BytesIO
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

try:
    import augraphy
    HAS_AUGRAPHY = True
except ImportError:
    HAS_AUGRAPHY = False

def scanner_bed(
    image: Image.Image,
    intensity: float = 0.5,
) -> Image.Image:
    """
    Simulate a document placed slightly askew on a flatbed scanner lid,
    showing the dark borders of the scanner background.
    """
    # Slight rotation
    angle = random.uniform(-1.5, 1.5) * intensity
    
    # Expand canvas to show scanner background
    pad_x = int(image.width * random.uniform(0.02, 0.08))
    pad_y = int(image.height * random.uniform(0.02, 0.08))
    
    new_w = image.width + pad_x * 2
    new_h = image.height + pad_y * 2
    
    # Scanner background color (dark gray/black)
    bg_color = (
        random.randint(10, 45), 
        random.randint(10, 45), 
        random.randint(10, 45)
    )
    bg = Image.new("RGB", (new_w, new_h), bg_color)
    
    # Convert receipt to RGBA to handle rotation transparency
    img_rgba = image.convert("RGBA")
    rotated = img_rgba.rotate(angle, expand=True)
    
    # Paste slightly off-center
    offset_x = (new_w - rotated.width) // 2 + random.randint(-15, 15)
    offset_y = (new_h - rotated.height) // 2 + random.randint(-15, 15)
    
    bg.paste(rotated, (offset_x, offset_y), rotated)
    
    # Optional: add a tiny bit of shadow at the edges of the receipt
    # (Simplified shadow by just returning the pasted image for speed)
    return bg.convert("RGB")


# ═══════════════════════════════════════════════════════════════════════
# STAGE: LIGHTING GRADIENT
# ═══════════════════════════════════════════════════════════════════════

def lighting_gradient(
    image: Image.Image,
    intensity: float = 0.4,
) -> Image.Image:
    """
    Simulate uneven lighting: desk lamp from one side, flash hotspot,
    or general ambient unevenness.
    """
    img_np = np.array(image).astype(np.float32)
    h, w = img_np.shape[:2]

    effect = random.choice(["linear_gradient", "radial_hotspot", "dual_gradient"])

    if effect == "linear_gradient":
        direction = random.choice(["horizontal", "vertical", "diagonal"])
        bright_end = 1.0 + intensity * 0.15
        dark_end = 1.0 - intensity * 0.12

        if direction == "horizontal":
            gradient = np.linspace(dark_end, bright_end, w, dtype=np.float32)
            if random.random() > 0.5:
                gradient = gradient[::-1]
            gradient = np.tile(gradient, (h, 1))
        elif direction == "vertical":
            gradient = np.linspace(dark_end, bright_end, h, dtype=np.float32)
            if random.random() > 0.5:
                gradient = gradient[::-1]
            gradient = np.tile(gradient.reshape(-1, 1), (1, w))
        else:
            Y, X = np.mgrid[:h, :w]
            gradient = (X / w + Y / h) / 2.0
            gradient = dark_end + gradient * (bright_end - dark_end)
            gradient = gradient.astype(np.float32)

    elif effect == "radial_hotspot":
        cx = random.uniform(0.2, 0.8) * w
        cy = random.uniform(0.2, 0.8) * h
        max_r = math.sqrt(w**2 + h**2) * 0.7

        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(np.float32)
        gradient = 1.0 + intensity * 0.15 * np.exp(-(dist**2) / (2 * (max_r * 0.3)**2))
        gradient = gradient.astype(np.float32)

    else:  # dual_gradient
        g1_x = random.uniform(0.1, 0.4) * w
        g1_y = random.uniform(0.1, 0.4) * h
        g2_x = random.uniform(0.6, 0.9) * w
        g2_y = random.uniform(0.6, 0.9) * h

        Y, X = np.ogrid[:h, :w]
        max_r = math.sqrt(w**2 + h**2) * 0.5

        d1 = np.sqrt((X - g1_x)**2 + (Y - g1_y)**2).astype(np.float32)
        d2 = np.sqrt((X - g2_x)**2 + (Y - g2_y)**2).astype(np.float32)

        g1 = np.exp(-(d1**2) / (2 * (max_r * 0.4)**2))
        g2 = np.exp(-(d2**2) / (2 * (max_r * 0.3)**2))

        gradient = 1.0 + intensity * 0.1 * (g1 * 0.6 + g2 * 0.4)
        gradient = gradient.astype(np.float32)

    if img_np.ndim == 3:
        gradient = gradient[:, :, np.newaxis]

    result = (img_np * gradient).clip(0, 255).astype(np.uint8)
    return Image.fromarray(result)


# ═══════════════════════════════════════════════════════════════════════
# STAGE: THERMAL FADE
# ═══════════════════════════════════════════════════════════════════════

def thermal_fade(
    image: Image.Image,
    intensity: float = 0.5,
) -> Image.Image:
    """
    Simulate thermal receipt paper degradation over time.
    - Yellows paper
    - Edge-biased vignette fading
    - Print head horizontal streaks
    - Dark friction scratches
    """
    img = image.copy()

    # 1. Yellow/warm tint (thermal paper aging)
    tint_strength = intensity * 0.10
    img_np = np.array(img).astype(np.float32)
    img_np[:, :, 0] = np.clip(img_np[:, :, 0] + tint_strength * 45, 0, 255)  # R
    img_np[:, :, 1] = np.clip(img_np[:, :, 1] + tint_strength * 28, 0, 255)  # G
    img_np[:, :, 2] = np.clip(img_np[:, :, 2] - tint_strength * 18, 0, 255)  # B
    img = Image.fromarray(img_np.astype(np.uint8))

    # 2. Reduce contrast
    contrast_factor = 1.0 - intensity * 0.35
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    # 3. Reduce brightness slightly
    brightness_factor = 1.0 - intensity * 0.08
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    # 4. Patchy & Edge-Biased Fading (Vignette)
    img_np = np.array(img).astype(np.float32)
    h, w = img_np.shape[:2]

    small_h, small_w = max(4, h // 40), max(4, w // 40)
    noise_map = np.random.uniform(
        1.0 - intensity * 0.22,
        1.0 + intensity * 0.05,
        (small_h, small_w)
    ).astype(np.float32)

    noise_map = cv2.resize(noise_map, (w, h), interpolation=cv2.INTER_LINEAR)
    noise_map = cv2.GaussianBlur(noise_map, (0, 0), sigmaX=w * 0.05)

    # Radial vignette (darker edges = more fading)
    Y, X = np.ogrid[:h, :w]
    center_y, center_x = h / 2, w / 2
    max_dist = math.sqrt(center_x**2 + center_y**2)
    dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    vignette = 1.0 - (dist / max_dist) * (intensity * 0.35)
    fade_map = noise_map * vignette

    img_np = img_np * fade_map[:, :, np.newaxis]
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_np)

    # 5. Horizontal streaks (print head wear)
    img_np = np.array(img).astype(np.float32)
    n_streaks = random.randint(3, 8)
    for _ in range(n_streaks):
        y = random.randint(0, h - 1)
        thickness = random.randint(1, 3)
        fade_factor = random.uniform(0.82, 0.94)
        y_start = max(0, y - thickness)
        y_end = min(h, y + thickness)
        img_np[y_start:y_end, :, :] *= fade_factor
    img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))

    # 6. Thermal Friction Scratches
    draw = ImageDraw.Draw(img)
    num_scratches = random.randint(0, int(4 * intensity) + 1)
    
    for _ in range(num_scratches):
        # Thermal scratches react black/dark-blue to heat/friction
        scratch_color = (
            random.randint(20, 50), 
            random.randint(20, 50), 
            random.randint(20, 60)
        )
        x1 = random.randint(0, w)
        y1 = random.randint(0, h)
        x2 = x1 + random.randint(-50, 50)
        y2 = y1 + random.randint(-50, 50)
        
        draw.line(
            [(x1, y1), (x2, y2)], 
            fill=scratch_color, 
            width=random.randint(1, 2)
        )

    return img


# ═══════════════════════════════════════════════════════════════════════
# STAGE: SCANNER ARTIFACTS (Augraphy)
# ═══════════════════════════════════════════════════════════════════════

def scanner_artifacts(
    image: Image.Image,
    intensity: float = 0.5,
) -> Image.Image:
    """
    Apply scanner/copier artifacts using Augraphy if available.
    Falls back to basic PIL operations if not.
    """
    if HAS_AUGRAPHY:
        return _augraphy_scanner(image, intensity)
    else:
        return _pil_scanner_fallback(image, intensity)


def _augraphy_scanner(image: Image.Image, intensity: float) -> Image.Image:
    from augraphy import (
        AugraphyPipeline, InkBleed, LowInkRandomLines, ColorPaper, 
        BrightnessTexturize, NoiseTexturize, DirtyRollers, Brightness, 
        SubtleNoise, Jpeg, Folding
    )

    img_np = np.array(image)
    min_dim = min(img_np.shape[0], img_np.shape[1])

    # Scale down for small images (POS receipts)
    if min_dim < 600:
        intensity = max(0.2, intensity * 0.7)

    ink = [
        InkBleed(
            intensity_range=(0.15 * intensity, 0.35 * intensity),
            kernel_size=(3, 5),
            severity=(0.3, 0.5),
            p=0.85,
        ),
        LowInkRandomLines(
            count_range=(2, max(3, int(8 * intensity))),
            p=0.7,
        ),
    ]


    paper = [
        ColorPaper(
            hue_range=(0, int(20 + 30 * intensity)),
            saturation_range=(10, int(15 + 25 * intensity)),
            p=0.9,
        ),
        BrightnessTexturize(
            texturize_range=(0.92 - 0.05 * intensity, 0.98),
            deviation=0.02 + 0.02 * intensity,
            p=0.85,
        ),
        NoiseTexturize(
            sigma_range=(1, max(2, int(4 * intensity))),
            turbulence_range=(2, max(3, int(6 * intensity))),
            p=0.75,
        ),
        DirtyRollers(
            line_width_range=(2, max(4, int(12 * intensity))),
            scanline_type=0,
            p=0.7,
        ),
    ]

    # Add folding structure to make the paper look previously handled
    structural = [
        Folding(
            fold_count=random.randint(1, max(2, int(4 * intensity))),
            fold_noise=0.1 * intensity,
            fold_angle_range=(-180, 180),
            gradient_width=(0.1, 0.2),
            p=0.85
        )
    ]

    post = [
        Brightness(
            brightness_range=(0.85, 1.02),
            p=0.8,
        ),
        SubtleNoise(
            subtle_range=max(3, int(8 * intensity)),
            p=0.85,
        ),
        Jpeg(
            quality_range=(max(45, int(85 - 40 * intensity)), 88),
            p=0.8,
        ),
    ]

    pipeline = AugraphyPipeline(ink_phase=ink, paper_phase=paper, post_phase=post)
    result = pipeline(img_np)
    return Image.fromarray(result)


def _pil_scanner_fallback(image: Image.Image, intensity: float) -> Image.Image:
    """Basic PIL fallback for scanner-like effects when augraphy isn't installed."""
    img = image.copy()

    tints = [(255, 252, 248), (248, 250, 255), (255, 255, 245),
             (252, 248, 245), (248, 252, 248)]
    tint = random.choice(tints)
    overlay = Image.new("RGB", img.size, tint)
    alpha = 0.04 + intensity * 0.06
    img = Image.blend(img, overlay, alpha=alpha)

    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.90, 1.04))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.88, 1.04))

    img_np = np.array(img).astype(np.float32)
    noise_std = 3 + intensity * 10
    noise = np.random.normal(0, noise_std, img_np.shape)
    img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(img_np)


# ═══════════════════════════════════════════════════════════════════════
# STAGE: CAMERA NOISE
# ═══════════════════════════════════════════════════════════════════════

def camera_noise(
    image: Image.Image,
    intensity: float = 0.3,
) -> Image.Image:
    """Light sensor noise + optional slight defocus."""
    img_np = np.array(image).astype(np.float32)
    noise_std = 1.5 + intensity * 5
    noise = np.random.normal(0, noise_std, img_np.shape)
    img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_np)

    if intensity > 0.4:
        radius = 0.2 + intensity * 0.5
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    return img


def jpeg_compress(
    image: Image.Image,
    intensity: float = 0.5,
) -> Image.Image:
    """Apply JPEG compression artifacts."""
    quality = int(92 - intensity * 50)
    quality = max(25, min(92, quality))

    buffer = BytesIO()
    img_rgb = image.convert("RGB")
    img_rgb.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")

ALL_STAGES = {
    "scanner_bed": scanner_bed,
    "lighting_gradient": lighting_gradient,
    "thermal_fade": thermal_fade,
    "scanner_artifacts": scanner_artifacts,
    "camera_noise": camera_noise,
    "jpeg_compress": jpeg_compress,
}

PRESETS: Dict[str, Dict[str, Any]] = {
    "clean": {
        "stages": [],
    },

    "bad_scan": {
        "stages": [
            ("scanner_bed", 1.0, 0.50),         # Adds the black borders and skews it
            ("lighting_gradient", 1.0, 0.45),   # uneven scanner light
            ("scanner_artifacts", 1.0, 0.65),   # ink bleed, dirty rollers, FOLDS, paper tint
            ("camera_noise", 1.0, 0.25),        # light sensor grain
            ("jpeg_compress", 1.0, 0.40),       # compression from saving
        ],
    },

    "faded": {
        "stages": [
            ("thermal_fade", 1.0, 0.60),        # yellowing + vignette + streaks + scratches
            ("lighting_gradient", 1.0, 0.30),   # slight uneven light
            ("camera_noise", 1.0, 0.20),        # light grain
            ("jpeg_compress", 1.0, 0.25),       # mild compression
        ],
    },
}


class DocumentAugmentor:

    def __init__(
        self,
        preset: Optional[str] = None,
        stages: Optional[List[str]] = None,
        intensity: float = 0.5,
        seed: Optional[int] = None,
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if preset:
            preset_config = PRESETS.get(preset, PRESETS["bad_scan"])
            self.stage_configs = preset_config["stages"]
        elif stages:
            self.stage_configs = [
                (s, 1.0, intensity) for s in stages
            ]
        else:
            self.stage_configs = PRESETS["bad_scan"]["stages"]

        self.preset_name = preset or "custom"

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply the augmentation pipeline to an image."""
        img = image.copy()

        for stage_name, probability, stage_intensity in self.stage_configs:
            if random.random() > probability:
                continue

            fn = ALL_STAGES.get(stage_name)
            if fn is None:
                continue

            actual_intensity = stage_intensity * random.uniform(0.95, 1.05)
            actual_intensity = max(0.05, min(1.0, actual_intensity))

            try:
                img = fn(img, intensity=actual_intensity)
            except Exception as e:
                print(f"  [augmentation warning: {stage_name} failed: {e}]")

        return img

    @staticmethod
    def multi_degrade(
        image: Image.Image,
        presets: List[str] = None,
    ) -> Dict[str, Image.Image]:
        """
        Render the same image at multiple degradation levels.
        Returns a dict of {preset_name: degraded_image}.
        """
        if presets is None:
            presets = ["clean", "bad_scan", "faded"]

        results = {}
        for preset_name in presets:
            if preset_name == "clean":
                results[preset_name] = image.copy()
            else:
                aug = DocumentAugmentor(preset=preset_name)
                results[preset_name] = aug(image)

        return results

    @staticmethod
    def list_presets() -> List[str]:
        return list(PRESETS.keys())

    @staticmethod
    def list_stages() -> List[str]:
        return list(ALL_STAGES.keys())


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Test document augmentation")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("--output-dir", "-o", default="./aug_test")
    parser.add_argument("--preset", "-p", default=None, choices=list(PRESETS.keys()))
    parser.add_argument("--stage", "-s", action="append")
    parser.add_argument("--intensity", type=float, default=0.5)
    parser.add_argument("--multi", action="store_true", help="Generate all preset variants")
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    img = Image.open(args.input).convert("RGB")

    if args.multi:
        print("Multi-degradation mode")
        results = DocumentAugmentor.multi_degrade(img)
        for name, result in results.items():
            out_path = os.path.join(args.output_dir, f"{name}.png")
            result.save(out_path)
            print(f"  {name}: {out_path} ({result.size})")
    elif args.stage:
        aug = DocumentAugmentor(stages=args.stage, intensity=args.intensity,
                                seed=args.seed)
        result = aug(img)
        out_path = os.path.join(args.output_dir, f"{'_'.join(args.stage)}.png")
        result.save(out_path)
        print(f"Saved: {out_path}")
    else:
        preset = args.preset or "bad_scan"
        aug = DocumentAugmentor(preset=preset, seed=args.seed)
        result = aug(img)
        out_path = os.path.join(args.output_dir, f"{preset}.png")
        result.save(out_path)
        print(f"Saved: {out_path}")