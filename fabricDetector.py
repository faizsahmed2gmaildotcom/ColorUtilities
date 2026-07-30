from typing import Literal
import pixelLib as pL
from kmeans import kmeans as kmeans_orig
from PIL import Image
import os
from patternDetectorPyTorch import predictFull


def kmeans(points, k, centers=None, tolerance=1, max_iterations=0) -> list:
    kmeans_result = kmeans_orig(points, k, centers, tolerance, max_iterations)
    while [0, 0, 0] in kmeans_result: kmeans_result.remove([0, 0, 0])  # Remove anomalous results
    return kmeans_result


test_folder = "test-data"
sec_kmeans_centers = 20
max_sec_colors = 2
min_sec_color_makeup = 0.0


def predictImage(_img_path, mode: Literal['shirting', 'suiting']):
    pixels = pL.preprocessPixels(_img_path)
    flattened_pixels = pL.flattenTupleArray(pixels)

    fancy_kmeans = kmeans(pL.getFreqList(flattened_pixels), 1)
    try:
        secondary_kmeans = kmeans(pL.getFreqList(flattened_pixels), sec_kmeans_centers)
    except ValueError:
        secondary_kmeans = kmeans(pL.getFreqList(flattened_pixels), 5)

    # Color detection
    _fancy_color = pL.getNearestColorName(fancy_kmeans[0], "fancy-colors")
    _secondary_colors = list(pL.getColorFreq(secondary_kmeans, "primary-colors", min_sec_color_makeup).keys())
    _primary_color = _secondary_colors.pop(0)
    if len(_secondary_colors) > max_sec_colors: _secondary_colors = _secondary_colors[:2]

    # Pattern detection
    processed_img = Image.new('RGB', (len(pixels[0]), len(pixels)))
    processed_img.putdata(list(map(tuple, flattened_pixels.tolist())))
    _pattern = predictFull(processed_img, "models/" + mode)

    return _fancy_color, _primary_color, _secondary_colors, _pattern


if __name__ == "__main__":
    for img_file_name in sorted(os.listdir(test_folder)):
        img_name = os.path.splitext(img_file_name)[0]
        print('\n' + img_name)
        # img_name_new = img_name.removesuffix(config["general"]["main_image_suffix"])
        # if (not pL.debug) and (img_name == img_name_new):
        #     print("Skipping: not main image...")
        #     continue
        # img_name = img_name_new

        img_path = os.path.join(test_folder, img_file_name)
        fancy_color, primary_color, secondary_colors_list, pattern_dat = predictImage(img_path, 'shirting')
        secondary_colors = ', '.join(secondary_colors_list)

        # new_row = sL.Row()
        # new_row.update(sku=img_name, Product_Name=fancy_color, color_filter_primary=primary_color,
        #                color_filter_secondary=secondary_colors, pattern=pattern)
        # sL.insertRow(new_row)
        # sL.save()

        print("---Name---\n"
              f"{fancy_color}")
        print("---Primary Color---\n"
              f"{primary_color}")
        print("---Secondary Colors---\n"
              f"{secondary_colors}")
        print("---Pattern---")
        for i, (tier_classes, tier_confs) in enumerate(zip(pattern_dat[0], pattern_dat[1])):
            formatted_confs = list(map(lambda p: f"{p:.2f}%", tier_confs))
            print(f"\tModel Tier {i + 1}:")
            print(f"\t\tClasses:     {tier_classes}")
            print(f"\t\tConfidences: {formatted_confs}")
        print(f"\nFinal pattern: {' + '.join(pattern_dat[0][-1][i] for i, conf in enumerate(pattern_dat[1][-1]) if conf > 20.0)}")
        print()
