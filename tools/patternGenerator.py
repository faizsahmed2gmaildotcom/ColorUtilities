import os
import random
import math
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
TARGET_SIZE = 800  # Final output dimension (800x800)
CANVAS_SIZE = 1600  # Oversized canvas to support translation/zoom cropping

# Color Palettes (RGB) based on textile research
PALETTES = {
    "awning": [
        [(200, 35, 45), (250, 250, 245)],  # Cabana Red & Off-White
        [(20, 35, 75), (250, 250, 245)],  # Navy Blue & Off-White
        [(30, 80, 50), (245, 240, 225)],  # Forest Green & Cream
        [(235, 175, 30), (250, 250, 245)],  # Sunburst Yellow & Off-White
        [(35, 35, 40), (250, 250, 245)],  # Charcoal Black & Off-White
    ],
    "bengal": [
        [(70, 130, 180), (248, 249, 250)],  # Classic Shirting Blue & White
        [(25, 45, 85), (248, 249, 250)],  # Navy Shirting & White
        [(220, 120, 140), (248, 249, 250)],  # Rose Pink & White
        [(160, 35, 50), (248, 249, 250)],  # Crimson & White
        [(110, 120, 135), (248, 249, 250)],  # Slate Grey & White
    ],
    "candy": [
        [(230, 50, 60), (255, 255, 255)],  # Candy Red & White
        [(240, 140, 175), (255, 255, 255)],  # Bubblegum Pink & White
        [(100, 180, 240), (255, 255, 255)],  # Sky Blue & White
        [(60, 180, 120), (255, 255, 255)],  # Peppermint Green & White
        [(250, 200, 40), (255, 255, 255)],  # Bright Yellow & White
    ],
    "pencil": [
        [(20, 30, 55), (210, 215, 220)],  # Navy Suiting & Light Grey Stripe
        [(45, 48, 55), (240, 240, 240)],  # Charcoal Suiting & White Stripe
        [(25, 25, 28), (200, 205, 210)],  # Black Suiting & Silver Stripe
        [(235, 242, 250), (20, 40, 80)],  # Light Blue Shirting & Navy Stripe
    ],
    "hairline": [
        [(180, 210, 235), (20, 60, 120)],  # Soft Light Blue & Navy Fine Stripe
        [(245, 245, 245), (40, 45, 50)],  # Light Grey & Charcoal Fine Stripe
        [(230, 225, 240), (80, 60, 110)],  # Soft Lavender & Deep Purple Stripe
        [(240, 242, 245), (15, 30, 65)],  # Off-White & Deep Navy Stripe
    ],
    "buffalo_check": [
        # Base, Secondary, Blend (Intersection)
        [(200, 30, 40), (20, 20, 20), (100, 20, 25)],  # Red & Black
        [(240, 240, 240), (20, 20, 20), (120, 120, 125)],  # White & Black
        [(30, 50, 110), (15, 15, 20), (20, 30, 60)],  # Navy & Black
        [(35, 85, 45), (15, 20, 15), (25, 50, 30)],  # Hunter Green & Black
        [(220, 160, 35), (20, 20, 20), (110, 85, 25)],  # Mustard & Black
    ],
    "gingham": [
        # Base/Light, Midtone, Solid Dark
        [(255, 255, 255), (155, 185, 225), (45, 95, 170)],  # Sky Blue Gingham
        [(255, 255, 255), (235, 145, 155), (195, 35, 50)],  # Classic Red Gingham
        [(255, 255, 255), (145, 150, 155), (35, 35, 40)],  # Charcoal Black Gingham
        [(255, 255, 255), (160, 200, 170), (45, 115, 65)],  # Sage Green Gingham
        [(255, 255, 255), (235, 195, 130), (210, 135, 25)],  # Amber Yellow Gingham
    ],
    "houndstooth": [
        # Light Thread, Dark Thread
        [(245, 245, 240), (20, 20, 25)],  # Classic Black & White
        [(240, 235, 220), (60, 40, 25)],  # Camel Brown & Cream
        [(245, 248, 250), (15, 30, 70)],  # Navy & Off-White
        [(220, 225, 230), (45, 50, 55)],  # Charcoal & Light Grey
        [(240, 235, 235), (85, 25, 35)],  # Burgundy & Off-White
    ],
    "shepherd_check": [
        # Light Thread, Dark Thread
        [(245, 245, 240), (25, 25, 30)],  # Classic Black & White
        [(245, 240, 230), (50, 35, 25)],  # Cream & Espresso Brown
        [(240, 242, 245), (20, 35, 75)],  # Soft Off-White & Deep Navy
        [(235, 235, 230), (55, 75, 50)],  # Warm Off-White & Hunter Green
        [(240, 235, 235), (95, 35, 45)],  # Off-White & Deep Burgundy
    ],
}


def getPalette(palette_name: str, jitter: float = 0.2):
    if (jitter > 1) or (jitter < 0):
        raise ValueError("Param 'jitter' must be in the range [0, 1]")

    jitter_rgb = lambda rgb: tuple(map(lambda c: min(int(c * random.uniform(1 - jitter, 1 + jitter)), 255), rgb))
    return list(map(jitter_rgb, random.choice(PALETTES[palette_name])))


# ==========================================
# PATTERN DRAWING FUNCTIONS
# ==========================================

def draw_awning_stripes(size):
    """1/2 inch width uniform stripes (100px per stripe -> 8 stripes per 800px)."""
    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)
    stripe_w = 100
    c1, c2 = getPalette("awning")

    for x in range(0, size, stripe_w * 2):
        draw.rectangle([x, 0, x + stripe_w, size], fill=c1)
        draw.rectangle([x + stripe_w, 0, x + 2 * stripe_w, size], fill=c2)
    return img


def draw_bengal_stripes(size):
    """1/4 inch width uniform stripes (50px per stripe)."""
    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)
    stripe_w = 50
    c1, c2 = getPalette("bengal")

    for x in range(0, size, stripe_w * 2):
        draw.rectangle([x, 0, x + stripe_w, size], fill=c1)
        draw.rectangle([x + stripe_w, 0, x + 2 * stripe_w, size], fill=c2)
    return img


def draw_candy_stripes(size):
    """1/8 inch width uniform stripes (25px per stripe)."""
    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)
    stripe_w = 25
    c1, c2 = getPalette("candy")

    for x in range(0, size, stripe_w * 2):
        draw.rectangle([x, 0, x + stripe_w, size], fill=c1)
        draw.rectangle([x + stripe_w, 0, x + 2 * stripe_w, size], fill=c2)
    return img


def draw_pencil_stripes(size):
    """1/16 inch width stripes (12.5px) with gaps much larger than stripe width."""
    bg_color, stripe_color = getPalette("pencil")
    img = Image.new("RGB", (size, size), color=bg_color)
    draw = ImageDraw.Draw(img)

    stripe_w = 12
    gap_w = random.randint(70, 105)  # Gaps much larger than 12px
    period = stripe_w + gap_w

    for x in range(0, size, period):
        draw.rectangle([x, 0, x + stripe_w, size], fill=stripe_color)
    return img


def draw_hairline_stripes(size):
    """
    Generates hairline stripes (<2px) with straight lines, micro-zigzags,
    or square waves, finished with a subtle blur to remove digital sharpness.
    """
    bg_color, stripe_color = getPalette("hairline")
    img = Image.new("RGB", (size, size), color=bg_color)
    draw = ImageDraw.Draw(img)

    stripe_w = random.randint(1, 2)
    gap_w = random.randint(10, 18)
    period = stripe_w + gap_w

    # Randomly select pattern type
    pattern_type = random.choice(["zigzag", "square_wave", "straight"])
    step_y = random.randint(3, 5)
    amp = random.choice([1, 2])

    for x_start in range(0, size, period):
        points = []

        if pattern_type == "zigzag":
            # Micro-zigzag pattern
            curr_x = x_start
            for y in range(0, size + step_y, step_y):
                points.append((curr_x, min(y, size)))
                curr_x = x_start + amp if curr_x == x_start else x_start

        elif pattern_type == "square_wave":
            # Square wave pattern (vertical segments connected by horizontal steps)
            curr_x = x_start
            for y in range(0, size, step_y):
                next_y = min(y + step_y, size)
                points.append((curr_x, y))
                points.append((curr_x, next_y))
                curr_x = x_start + amp if curr_x == x_start else x_start

        else:
            # Straight line
            points = [(x_start, 0), (x_start, size)]

        if len(points) > 1:
            draw.line(points, fill=stripe_color, width=stripe_w)

    # Apply a soft Gaussian blur to smooth out pixelation and harsh edges
    blur_radius = random.uniform(0.35, 0.55)
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return img


def draw_buffalo_check(size):
    """Buffalo check grid (approx. 24 squares visible per 800x800 area -> ~160px cell size)."""
    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)

    sq_size = 160  # 800px / 160px = 5x5 grid = 25 squares (~24)
    c_base, c_sec, c_blend = getPalette("buffalo_check")

    cols = math.ceil(size / sq_size)
    rows = math.ceil(size / sq_size)

    for r in range(rows):
        for c in range(cols):
            x1 = c * sq_size
            y1 = r * sq_size
            x2 = x1 + sq_size
            y2 = y1 + sq_size

            # Classic 2x2 weave block logic
            if r % 2 == 0 and c % 2 == 0:
                color = c_base
            elif r % 2 == 1 and c % 2 == 1:
                color = c_sec
            else:
                color = c_blend

            draw.rectangle([x1, y1, x2, y2], fill=color)

    return img


def draw_gingham_check(size):
    """
    Generates a Gingham check pattern with 3 distinct tones
    (Base Light, Midtone overlap, and Solid Dark cross).
    """
    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)

    # Square sizes typically range between 25px and 45px for gingham scale
    sq_size = random.randint(25, 45)
    c_light, c_mid, c_dark = getPalette("gingham")

    cols = math.ceil(size / sq_size)
    rows = math.ceil(size / sq_size)

    for r in range(rows):
        for c in range(cols):
            x1 = c * sq_size
            y1 = r * sq_size
            x2 = x1 + sq_size
            y2 = y1 + sq_size

            # Gingham 3-tone grid logic
            if r % 2 == 0 and c % 2 == 0:
                color = c_light
            elif r % 2 == 1 and c % 2 == 1:
                color = c_dark
            else:
                color = c_mid

            draw.rectangle([x1, y1, x2, y2], fill=color)

    return img


def draw_houndstooth(size):
    """
    Generates a Houndstooth Check using exact 2/2 twill weave geometry
    with alternating 4-thread color bands.
    """
    c_light, c_dark = getPalette("houndstooth")

    # Thread thickness in pixels (3 to 5 px per thread)
    thread_px = random.randint(3, 5)

    num_threads = math.ceil(size / thread_px)

    cols = np.arange(num_threads)
    rows = np.arange(num_threads)
    col_grid, row_grid = np.meshgrid(cols, rows)

    # 4-and-4 thread band assignments (True = Dark, False = Light)
    warp_is_dark = (col_grid % 8) < 4
    weft_is_dark = (row_grid % 8) < 4

    # 2/2 twill weave rule: True = Warp thread over Weft thread
    warp_on_top = ((col_grid + row_grid) % 4) < 2

    # Resolve surface thread color at each warp/weft intersection
    is_dark = np.where(warp_on_top, warp_is_dark, weft_is_dark)

    # Construct RGB matrix from boolean thread map
    grid_rgb = np.where(is_dark[:, :, None], np.array(c_dark), np.array(c_light)).astype(np.uint8)

    # Upscale pixel matrix to match target canvas size
    img_thread = Image.fromarray(grid_rgb)
    img = img_thread.resize((num_threads * thread_px, num_threads * thread_px), Image.Resampling.NEAREST)
    img = img.crop((0, 0, size, size))

    return img


def draw_shepherd_check(size):
    """
    Generates a Shepherd's Check pattern: a gingham-style grid featuring solid
    light squares, solid dark squares, and intermediate squares filled with crisp
    diagonal twill stripes.
    """
    c_light, c_dark = getPalette("shepherd_check")

    # Check square dimension (35px to 65px per grid cell)
    sq_size = random.randint(35, 65)

    # Width of individual diagonal stripes inside mixed squares (4px to 8px)
    stripe_w = random.randint(4, 8)
    stripe_period = stripe_w * 2

    # Coordinate grids across canvas
    y, x = np.ogrid[:size, :size]

    r_grid = y // sq_size
    c_grid = x // sq_size

    # Grid block classifications
    is_solid_light = (r_grid % 2 == 0) & (c_grid % 2 == 0)
    is_solid_dark = (r_grid % 2 == 1) & (c_grid % 2 == 1)

    # 45-degree diagonal stripe mask for intermediate blocks
    stripe_mask = ((x - y) % stripe_period) < stripe_w

    # Composite thread assignment: Solid Light -> False, Solid Dark -> True, Mixed -> Stripe Mask
    is_dark = np.where(
        is_solid_light,
        False,
        np.where(is_solid_dark, True, stripe_mask)
    )

    # Convert boolean mask into RGB image array
    rgb = np.where(
        is_dark[:, :, None],
        np.array(c_dark, dtype=np.uint8),
        np.array(c_light, dtype=np.uint8)
    )

    return Image.fromarray(rgb)


# ==========================================
# TRANSFORMATIONS & AUGMENTATIONS
# ==========================================

def apply_transforms(oversized_img):
    """
    Applies random ~5% zoom and ~25% translation,
    cropping back to final 800x800 image size.
    """
    # Random Zoom (~5% -> scale 0.95 to 1.05)
    zoom_factor = random.uniform(0.9, 1.1)
    crop_size = int(TARGET_SIZE / zoom_factor)

    # Base center in 1600x1600 canvas
    center_x = CANVAS_SIZE // 2
    center_y = CANVAS_SIZE // 2

    # Random Translation (~25% of TARGET_SIZE -> +/- 200px)
    max_trans = int(TARGET_SIZE * 0.25)
    trans_x = random.randint(-max_trans, max_trans)
    trans_y = random.randint(-max_trans, max_trans)

    # Calculate crop coordinates
    crop_center_x = center_x + trans_x
    crop_center_y = center_y + trans_y

    left = crop_center_x - crop_size // 2
    top = crop_center_y - crop_size // 2
    right = left + crop_size
    bottom = top + crop_size

    # Crop and scale back to 800x800
    cropped = oversized_img.crop((left, top, right, bottom))
    final_img = cropped.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.LANCZOS)

    return final_img


def apply_noise(final_image):
    """
    Simulates microscopic thread imperfections (fiber grain and weave slub)
    along with trace dust particles across the final 800x800 image.
    """
    # Convert PIL Image to float array for precise noise math
    img_arr = np.array(final_image, dtype=np.float32)
    height, width, _ = img_arr.shape

    # 1. Microscopic Thread Grain (Fine-grained Gaussian noise)
    # Simulates micro-variations in individual cotton/poly fibers and camera noise
    grain_std = np.random.uniform(2.5, 5.0)
    grain = np.random.normal(0, grain_std, (height, width, 1))
    img_arr += grain

    # 2. Fabric Weave "Slub" / Thread Density Variations
    # Adds subtle horizontal and vertical luminance bands (non-uniform yarn thickness)
    h_slub = np.random.normal(1.0, 0.012, (height, 1, 1))
    v_slub = np.random.normal(1.0, 0.012, (1, width, 1))
    img_arr *= (h_slub * v_slub)

    # Clamp values back to valid RGB [0, 255] range
    img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
    noisy_img = Image.fromarray(img_arr)

    # # 3. Trace Dust Specks & Micro-Fibers
    # # Draws a small, random count of microscopic dark or light specks
    # draw = ImageDraw.Draw(noisy_img)
    # num_dust = np.random.randint(10, 30)  # Trace count across an 800x800 image
    #
    # for _ in range(num_dust):
    #     x = np.random.randint(0, width)
    #     y = np.random.randint(0, height)
    #     radius = np.random.uniform(0.5, 1.5)  # Tiny 1-3px specks
    #
    #     # Randomly pick dark environmental dust or light lint/fiber specks
    #     dust_val = np.random.choice([np.random.randint(20, 70), np.random.randint(200, 245)])
    #     dust_color = (dust_val, dust_val, dust_val)
    #
    #     draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=dust_color)

    return noisy_img


def generate_pattern_image(pattern_name):
    """Generates a fully augmented final 800x800 image for a given pattern."""
    if pattern_name not in PATTERN_GENERATORS:
        raise ValueError(f"Unknown pattern: {pattern_name}")

    # Generate on oversized canvas to prevent cropping edge artifacts
    raw_canvas = PATTERN_GENERATORS[pattern_name](CANVAS_SIZE)

    # Apply zoom and translation transforms, and noise
    final_image = apply_transforms(raw_canvas)
    final_image = apply_noise(final_image)
    return final_image


# Map patterns to generator functions
PATTERN_GENERATORS = {
    "awning_stripe": draw_awning_stripes,
    "bengal_stripe": draw_bengal_stripes,
    "candy_stripe": draw_candy_stripes,
    "pencil_stripe": draw_pencil_stripes,
    "hairline_stripe": draw_hairline_stripes,
    "buffalo_check": draw_buffalo_check,
    "gingham": draw_gingham_check,
    # "houndstooth": draw_houndstooth,
    "shepherd_check": draw_shepherd_check,
}

# ==========================================
# DEMO / DATASET GENERATION SCRIPT
# ==========================================

if __name__ == "__main__":
    output_dir = "dataset_preview"

    print("Generating pattern samples...")
    for pattern_type in PATTERN_GENERATORS.keys():
        os.makedirs(os.path.join(output_dir, pattern_type), exist_ok=True)
        # Generate N samples per pattern
        for i in range(200):
            img = generate_pattern_image(pattern_type)
            filename = os.path.join(output_dir, pattern_type, f"generated_{i + 1}.png")
            img.save(filename)
            print(f"Saved: {filename}")

    print("\nDataset generation complete!")
