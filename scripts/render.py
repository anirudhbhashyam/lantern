"""Render the sampler's raw grayscale frames as a labeled, looping GIF."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SIDE, CLASSES = 28, 10
FRAME_SIZE = SIDE * SIDE * CLASSES
WIDTH, HEIGHT = 888, 494
BG, PANEL, EDGE = "#101722", "#151f2d", "#28374b"
WHITE, MUTED, ACCENT = "#f3f7ff", "#a4b5ca", "#69e0ca"


def font(size, *, mono=False, weight="Regular"):
    name = "SFNSMono.ttf" if mono else "SFNS.ttf"
    try:
        face = ImageFont.truetype("/System/Library/Fonts/" + name, size)
    except OSError:
        return ImageFont.load_default(size=size)
    try:
        face.set_variation_by_name(weight)
    except (OSError, ValueError):
        # Some macOS versions ship static system fonts.
        pass
    return face


def render(raw: bytes, output: Path, scale: int = 1):
    if not raw or len(raw) % FRAME_SIZE:
        raise ValueError("Input must contain complete frames of ten 28×28 images.")
    count = len(raw) // FRAME_SIZE
    if count < 2:
        raise ValueError("Expected a noise frame and a generated frame.")

    if scale not in (1, 2):
        raise ValueError("Scale must be 1 or 2.")

    def px(*coords):
        return tuple(coord * scale for coord in coords)

    header = font(32 * scale, weight="Bold")
    subtitle = font(14 * scale)
    label = font(17 * scale, mono=True, weight="Semibold")
    caption = font(13 * scale, mono=True)
    indexed = []
    palette = None
    for frame in range(count):
        image = Image.new("RGB", px(WIDTH, HEIGHT), BG)
        d = ImageDraw.Draw(image)
        d.text(px(27, 14), "MNIST  /  REVERSE DIFFUSION", font=header, fill=WHITE)
        d.text(px(28, 57), "Ten digit classes generated in parallel from Gaussian noise", font=subtitle, fill=MUTED)
        d.line(px(28, 83, 860, 83), fill=EDGE, width=scale)

        for digit in range(CLASSES):
            x, y = 28 + digit % 5 * 169, 96 + digit // 5 * 165
            d.rounded_rectangle(px(x, y, x + 155, y + 151), radius=10 * scale, fill=PANEL, outline=EDGE, width=scale)
            start = frame * FRAME_SIZE + digit * SIDE * SIDE
            tile = Image.frombytes("L", (SIDE, SIDE), raw[start:start + SIDE * SIDE])
            tile = tile.resize(px(112, 112), Image.Resampling.NEAREST)
            tile = ImageOps.colorize(tile, "#080d15", "#effaff")
            image.paste(tile, px(x + 22, y + 9))
            d.text(px(x + 78, y + 136), str(digit), anchor="mm", font=label, fill=ACCENT)

        track_y = 452
        d.rounded_rectangle(px(28, track_y, 860, track_y + 5), radius=2 * scale, fill=EDGE)
        progress = round(832 * frame / (count - 1))
        if progress > 0:
            d.rounded_rectangle(px(28, track_y, 28 + progress, track_y + 5), radius=2 * scale, fill=ACCENT)
        d.text(px(28, 467), f"DENOISING  {frame:02d} / {count - 1:02d}", font=caption, fill=MUTED)
        status = "GAUSSIAN NOISE" if frame == 0 else ("GENERATED DIGITS" if frame == count - 1 else "DDIM SAMPLING")
        d.text(px(860, 466), status, anchor="rt", font=caption, fill=ACCENT)
        # A fixed palette prevents colors from shifting between frames.
        if palette is None:
            palette = image.quantize(colors=256)
        indexed.append(image.quantize(palette=palette, dither=Image.Dither.NONE))

    durations = [55] * count
    durations[0] = 450
    durations[-1] = 1800
    indexed[0].save(output, save_all=True, append_images=indexed[1:], duration=durations, loop=0, optimize=False)
    image.save(output.with_suffix(".png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", type=Path, nargs="?", default=ROOT / "artifacts/frames.gray")
    parser.add_argument("gif", type=Path, nargs="?", default=ROOT / "artifacts/mnist-diffusion.gif")
    parser.add_argument("--scale", type=int, choices=(1, 2), default=1, help="output scale (default: 1)")
    args = parser.parse_args()
    render(args.frames.read_bytes(), args.gif, args.scale)
