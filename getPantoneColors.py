# Source: https://github.com/pedroduke/pantone-tcx/blob/main/src/pantoneColors.js
import re
from typing import Dict, Tuple


def hex_to_rgb(hex_str: str) -> Tuple:
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))


def extract_pantone_colors() -> Dict[str, Tuple]:
    with open('pantoneColors.js', 'r') as f:
        content = f.read()

    # Extract all objects using regex
    pattern = r'\{\s*name:\s*"([^"]+)",\s*hex:\s*"([^"]+)",\s*tcx:\s*"([^"]+)",\s*\}'
    matches = re.findall(pattern, content)

    color_dict: Dict[str, Tuple] = {}
    for name, hex_code, _ in matches:
        rgb = hex_to_rgb(hex_code)
        color_dict[name] = rgb

    return color_dict

for name, RGB in extract_pantone_colors().items():
    print(f'"{name}" = [{str(RGB).strip('()')}]')
