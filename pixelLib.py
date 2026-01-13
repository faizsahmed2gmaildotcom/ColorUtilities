from config import *
import numpy as np
from math import dist
from collections import Counter
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color

DEFAULT_LIGHTNESS = 0


def normalizeRGB(rgb_code: tuple[float, float, float], factor: float = 1 / 255) -> tuple[float, ...]:
    return tuple(rgb * factor for rgb in rgb_code)


def rgbToLab(rgb_code: tuple[float, float, float], ignore_lightness=False) -> LabColor:
    lab_color = convert_color(sRGBColor(*normalizeRGB(rgb_code)), LabColor)
    if ignore_lightness: lab_color.lab_l = DEFAULT_LIGHTNESS
    return lab_color


ALL_COLOR_NAMES = {"fancy-colors": [rgb for rgb in config["fancy-colors"]], "primary-colors": [rgb for rgb in config["primary-colors"]]}
ALL_COLOR_CODES = {"RGB": {"fancy-colors": [c for c in config["fancy-colors"].values()], "primary-colors": [rgb for rgb in config["primary-colors"].values()]},
                   "CIE2000": {"fancy-colors": [rgbToLab(rgb) for rgb in config["fancy-colors"].values()],
                               "primary-colors": [rgbToLab(rgb, True) for rgb in config["primary-colors"].values()]}}

LUM_709 = {'r': 0.2126, 'g': 0.7152, 'b': 0.0722}  # From the Rec. 709 brightness formula
salient_selection_method = ["top_percent", "zscore"][1]


def insertAndIncrement(dict_: dict[Any, int], key: Any):
    if key not in dict_:
        dict_.update({key: 1})
    else:
        dict_[key] += 1


def flattenArray(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(-1, arr.shape[2])


def removeDims(list_: list, dim_to_remove: int):
    """
    Removes a dimension from a 2D list
    :param list_: a 2D list
    :param dim_to_remove: dimension to remove from each iterable in arr
    """
    for i in range(len(list_)):
        list_[i] = [list_[i][v] for v in range(len(list_[i])) if v != dim_to_remove]


def getFrequency(pixels: np.ndarray[Any]) -> list[list[Any]]:
    pixels = np.asarray(pixels)

    data, occurrences = np.unique(pixels, axis=0, return_counts=True)
    data = data.tolist()
    occurrences = occurrences.tolist()
    return [[d, o] for d, o in zip(data, occurrences)]


def removeWhitePixels(pixels: np.ndarray[Any]) -> np.ndarray[Any]:
    color_end_row = 0
    color_end_col = 0
    middle_row = len(pixels) // 2
    middle_col = len(pixels[0]) // 2
    white = 255 * 3

    while pixels[color_end_row][middle_col].sum() == white:
        color_end_row += 1
    while pixels[middle_row][color_end_col].sum() == white:
        color_end_col += 1

    return cropPixels(pixels, color_end_row, len(pixels) - 1 - color_end_row, color_end_col, len(pixels[0]) - 1 - color_end_col)


def cropPixels(pixels: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:
    vertical_mask = np.zeros(len(pixels), dtype=bool)
    vertical_mask[top:bottom + 1] = True
    pixels = pixels[vertical_mask]

    horizontal_mask = np.zeros(len(pixels[0]), dtype=bool)
    horizontal_mask[left:right + 1] = True
    pixels = pixels[:, horizontal_mask]

    return pixels


def delta_e_cie2000_patched(color1: LabColor, color2: LabColor, Kl=1, Kc=1, Kh=1):
    """
    Patched from delta_e_cie2000 in colormath.color_diff_matrix
    """
    from colormath.color_diff_matrix import delta_e_cie2000
    color1_vector = np.array([color1.lab_l, color1.lab_a, color1.lab_b])
    color2_matrix = np.array([(color2.lab_l, color2.lab_a, color2.lab_b)])
    delta_e = delta_e_cie2000(color1_vector, color2_matrix, Kl=Kl, Kc=Kc, Kh=Kh)[0]
    return delta_e.item()  # Updated numpy.asscalar to nparray.item


def colorDist(color_1: tuple[float, float, float] | LabColor, color_2: tuple[float, float, float] | LabColor) -> float:
    if isinstance(color_1, LabColor):
        return delta_e_cie2000_patched(color_1, color_2)
    else:
        return dist(color_1, color_2)


def getNearestColorName(rgb_color: tuple[int, int, int], color_type: Literal["fancy-colors", "primary-colors"], mode: Literal["RGB", "CIE2000"] = "CIE2000") -> str:
    min_dist = float("inf")
    min_idx = -1
    if mode == "CIE2000":
        rgb_color = rgbToLab(rgb_color)
        if color_type == "primary-colors":
            rgb_color.lab_l = DEFAULT_LIGHTNESS

    for i, color_code in enumerate(ALL_COLOR_CODES[mode][color_type]):
        current_dist = colorDist(color_code, rgb_color)
        if current_dist < min_dist:
            min_dist = current_dist
            min_idx = i
    print(f"{color_type} min_dist: {min_dist}")

    return ALL_COLOR_NAMES[color_type][min_idx]


def blend(pixels: np.ndarray, block_size_x: int, block_size_y: int) -> np.ndarray:
    blended_pixels: list[list[int]] = []
    for j in range(len(pixels) // block_size_y):
        blended_pixels.append([])
        for i in range(len(pixels[0]) // block_size_x):
            block = cropPixels(pixels, j * block_size_y, (j + 1) * block_size_y, i * block_size_x,
                               (i + 1) * block_size_x)
            sum_colors = np.array([0, 0, 0])
            for row in block:
                for pixel in row:
                    sum_colors += pixel
            sum_colors //= (block_size_x * block_size_y)
            blended_pixels[-1].append(sum_colors.tolist())

    return np.asarray(blended_pixels)


def medianFilter(pixels: np.ndarray, window_size: int) -> np.ndarray:
    print("Applying median filter...")
    if window_size == 1:
        return pixels
    if (window_size % 2) == 0:
        raise ValueError("window_size must be odd")

    processed_pixels = np.zeros(pixels.shape, dtype=int)
    for j in range(pixels.shape[0]):
        for i in range(pixels.shape[1]):
            processed_pixels[j][i] = np.median(getSurroundingPixels(pixels, window_size, j, i), (0, 1))

    return processed_pixels


def getSurroundingPixels(pixels: np.ndarray, window_size: int, row: int, col: int) -> np.ndarray:
    apothem = window_size // 2
    top = row - apothem if row - apothem > 0 else 0
    bottom = row + apothem if row + apothem < pixels.shape[0] else pixels.shape[0] - 1
    left = col - apothem if col - apothem > 0 else 0
    right = col + apothem if col + apothem < pixels.shape[1] else pixels.shape[1] - 1

    return pixels[top:bottom + 1, left:right + 1]


def spreadSalientPixels(
        pixels: np.ndarray,
        radius: int = 1,
        selection_method: str = salient_selection_method,
        percent: float = 95.0,
        zscore_k: float = 1.0,
        weight_lum: float = 0.7,
        weight_sat: float = 0.3,
) -> Union[list, np.ndarray]:
    """
    Spread brighter/more noticeable pixels to surrounding pixels.

    Parameters:
    - pixels: 2D list or numpy array with shape (H, W, 3), values 0..255
    - radius: integer radius of spreading (Euclidean)
    - selection_method: "top_percent" or "zscore"
    - percent: percentile for "top_percent" selection (0-100)
    - zscore_k: multiplier for standard deviation when using "zscore"
    - weight_lum / weight_sat: relative weights for luminance and saturation

    Returns:
    - same type as input (list if input was list, otherwise numpy array)
    """
    print("Spreading salient pixels...")
    arr_in = np.asarray(pixels, dtype=np.float32)
    if arr_in.ndim != 3 or arr_in.shape[2] < 3:
        raise ValueError("pixels must be a 2D array with RGB codes")

    H, W = arr_in.shape[:2]
    R, G, B = arr_in[..., 0], arr_in[..., 1], arr_in[..., 2]
    lum = (LUM_709['r'] * R + LUM_709['g'] * G + LUM_709['b'] * B)

    # Approximate saturation: (max - min) / (max + eps)
    mx = arr_in.max(axis=2)
    mn = arr_in.min(axis=2)
    sat = (mx - mn) / (mx + 1e-6)  # normalized saturation

    # Normalize luminance to 0..1
    lum_norm = (lum - lum.min()) / (np.ptp(lum) + 1e-6)

    # Score combining luminance and saturation
    score = weight_lum * lum_norm + weight_sat * sat
    score = np.clip(score, 0.0, 1.0)

    # Select pixels to spread
    if selection_method == "top_percent":
        thresh = np.percentile(score, percent)
        selected = score >= thresh
    elif selection_method == "zscore":
        mean = score.mean()
        std = score.std()
        selected = score >= (mean + zscore_k * std)
    else:
        raise ValueError("selection_method must be 'top_percent' or 'zscore'")

    # Prepare output and assignment strength map
    out = arr_in.copy().astype(np.uint8)
    assigned_strength = score.copy()  # current assigned strength per pixel

    # Precompute neighbor offsets within radius (Euclidean)
    offsets = []
    r = int(max(0, radius))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            d = np.sqrt(dx * dx + dy * dy)
            if d <= radius:
                offsets.append((dy, dx, d))
    # Sort offsets by increasing distance so center overwrites first, optional
    offsets.sort(key=lambda x: x[2])

    # Iterate selected pixels and spread
    sel_indices = np.argwhere(selected)
    for (y, x) in sel_indices:
        src_color = arr_in[y, x].astype(np.uint8)
        src_strength = float(score[y, x])
        if src_strength <= 0:
            continue
        for dy, dx, d in offsets:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= H or nx < 0 or nx >= W:
                continue
            # distance penalty: linear falloff
            dist_penalty = (d / (radius + 1e-6))
            assign_strength = src_strength * max(0.0, 1.0 - dist_penalty)
            if assign_strength > assigned_strength[ny, nx]:
                out[ny, nx] = src_color
                assigned_strength[ny, nx] = assign_strength

    # Return same type as input
    if isinstance(pixels, np.ndarray):
        return out
    else:
        return out.tolist()


def spreadDominantPixels(
        pixels: np.ndarray,
        radius: int = 1,
        top_n_colors: int = 1,
        min_area_percent: float = 0.5,
        distance_weight: bool = True,
) -> Union[list, np.ndarray]:
    """
    Spread the most dominant (most frequent) pixel colors to surrounding pixels.

    Parameters:
    - pixels: 2D list or numpy array with shape (H, W, 3), values 0..255
    - radius: integer radius of spreading (Euclidean)
    - top_n_colors: number of most frequent colors to consider as dominant
    - min_area_percent: only consider colors that occupy at least this % of the image
                         (helps avoid spreading very rare but frequent colors)
    - distance_weight: whether to apply linear falloff based on distance

    Returns:
    - same type as input (list if input was list, otherwise numpy array)
    """
    print("Spreading dominant pixels...")
    arr_in = np.asarray(pixels, dtype=np.float32)
    if arr_in.ndim != 3 or arr_in.shape[2] < 3:
        raise ValueError("pixels must be a 2D array with RGB codes")

    H, W = arr_in.shape[:2]
    total_pixels = H * W

    # Flatten to list of tuples for counting
    pixels_flat = arr_in.reshape(-1, 3).astype(np.uint8)
    color_counts = Counter(tuple(color) for color in pixels_flat)

    # Get the most common colors, sorted by frequency descending
    sorted_colors = color_counts.most_common()

    # Select dominant colors: top N and at least min_area_percent
    min_count = int(total_pixels * min_area_percent / 100.0)
    dominant_colors = []
    count = 0
    for color, freq in sorted_colors:
        if freq >= min_count:
            dominant_colors.append((color, freq))
            count += 1
            if count >= top_n_colors:
                break
        else:
            # Since list is sorted, we can stop early
            break

    if not dominant_colors:
        print("Warning: No colors meet the minimum area threshold. Returning original.")
        return pixels if isinstance(pixels, np.ndarray) else pixels.tolist()

    # Prepare output and assignment strength map
    # We use frequency as initial strength (higher frequency → stronger claim)
    out = arr_in.copy().astype(np.uint8)
    assigned_strength = np.zeros((H, W), dtype=np.float32)

    # Precompute neighbor offsets within radius (Euclidean)
    offsets = []
    r = int(max(0, radius))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            d = np.sqrt(dx * dx + dy * dy)
            if d <= radius:
                offsets.append((dy, dx, d))
    # Sort by increasing distance (closer pixels overwrite first)
    offsets.sort(key=lambda p: p[2])

    # Create a mask for pixels belonging to dominant colors
    dominant_set = {color for color, _ in dominant_colors}
    is_dominant = np.array([tuple(p) in dominant_set for p in pixels_flat]).reshape(H, W)

    # Assign initial strength based on frequency
    max_freq = max(freq for _, freq in dominant_colors) if dominant_colors else 1
    for (color, freq) in dominant_colors:
        color_mask = np.all(arr_in == np.array(color), axis=2)
        strength = freq / max_freq  # normalize to 0..1
        assigned_strength[color_mask] = strength

    # Spread dominant colors
    sel_indices = np.argwhere(is_dominant)
    for (y, x) in sel_indices:
        src_color = arr_in[y, x].astype(np.uint8)
        src_strength = assigned_strength[y, x]
        if src_strength <= 0:
            continue
        for dy, dx, d in offsets:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= H or nx < 0 or nx >= W:
                continue
            if distance_weight:
                dist_penalty = d / (radius + 1e-6)
                assign_strength = src_strength * max(0.0, 1.0 - dist_penalty)
            else:
                assign_strength = src_strength

            if assign_strength > assigned_strength[ny, nx]:
                out[ny, nx] = src_color
                assigned_strength[ny, nx] = assign_strength

    # Return in the same format as input
    if isinstance(pixels, np.ndarray):
        return out
    else:
        return out.tolist()


def enhanceWhitePoint(pixels: np.ndarray) -> np.ndarray:
    """
    Enhances the white point of pixels by increasing brightness of lighter colors.
    Applies a curve adjustment that amplifies values closer to white (255).
    :param pixels: 2D array of RGB pixel values
    :return: 2D array with enhanced white point
    """
    print("Enhancing white point...")
    pixels_float = pixels.astype(float)

    luminance = LUM_709['r'] * pixels_float[:, :, 0] + LUM_709['g'] * pixels_float[:, :, 1] + LUM_709['b'] * \
                pixels_float[:, :, 2]

    # Bias to enhance bright pixels only
    lum_norm = luminance / 255.0
    enhancement_curve = np.where(lum_norm > 5, np.power(lum_norm, 0.5) * 255.0, luminance)

    max_luminance = np.maximum(luminance, 1)
    scale_factor = enhancement_curve / max_luminance

    enhanced_pixels = pixels_float * scale_factor[:, :, np.newaxis]
    enhanced_pixels = np.clip(enhanced_pixels, 0, 255).astype(np.uint8)

    return enhanced_pixels


if __name__ == "__main__":
    print(ALL_COLOR_NAMES, '\n', ALL_COLOR_CODES["CIE2000"])
