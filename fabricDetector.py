from config import config
import pixelLib as pL
import spreadsheetLib as sL
from kmeans import kmeans as kmeans_orig
from PIL import Image
import os
from patternDetectorPyTorch import predictPattern, predictWeave

def kmeans(points, k, centers=None, tolerance=1, max_iterations=0) -> list:
    kmeans_result = kmeans_orig(points, k, centers, tolerance, max_iterations)
    # remove anomalous results
    while [0, 0, 0] in kmeans_result: kmeans_result.remove([0, 0, 0])
    return kmeans_result

test_folder = "test-images"
vertical_offset = 10
horizontal_offset = 10
median_filter_size = 5
salient_pixel_bias = 10
img_scale_factor = 0.25
sec_kmeans_centers = 20
min_sec_color_ratio = 0.0


def processImage(img_path):
    pixels = pL.getPixelList(img_path, img_scale_factor)
    pixels = pL.removeWhiteBackground(pixels)
    pixels = pL.cropPixels(pixels, vertical_offset, len(pixels) - 1 - vertical_offset, horizontal_offset,
                           len(pixels[0]) - 1 - horizontal_offset)
    pixels = pL.medianFilter(pixels, median_filter_size)
    pixels = pL.spreadSalientPixels(pixels, salient_pixel_bias)
    flattened_pixels = pL.flattenArrayOfTuples(pixels)
    if pL.debug:
        cf_img = Image.new("RGB", (len(pixels[0]), len(pixels)))
        cf_img.putdata(list(map(tuple, flattened_pixels.tolist())))
        cf_img.save(os.path.join("processed-images", str(os.path.basename(img_path))), format="jpeg")

    fancy_kmeans = kmeans(pL.getFreqList(flattened_pixels), 1)
    try:
        secondary_kmeans = kmeans(pL.getFreqList(flattened_pixels), sec_kmeans_centers)
    except ValueError:
        secondary_kmeans = kmeans(pL.getFreqList(flattened_pixels), 5)
    print(f"{secondary_kmeans = }")

    fancy_color = pL.getNearestColorName(fancy_kmeans[0], "fancy-colors")
    secondary_colors = list(pL.getFreqColorDict(secondary_kmeans, "primary-colors", min_sec_color_ratio).keys())
    primary_color = secondary_colors.pop(0)
    if len(secondary_colors) > 2: secondary_colors = secondary_colors[:2]
    pattern = predictPattern(img_path)
    weave = predictWeave(img_path)

    return fancy_color, primary_color, secondary_colors, pattern, weave


if __name__ == "__main__":
    for img_file_name in sorted(os.listdir(test_folder)):
        img_name = os.path.splitext(img_file_name)[0]
        print('\n' + img_name)
        img_name_new = img_name.removesuffix(config["general"]["main_image_suffix"])
        if (not pL.debug) and (img_name == img_name_new):
            print("Skipping: not main image...")
            continue
        img_name = img_name_new

        img_path = os.path.join(test_folder, img_file_name)
        fancy_color, primary_color, secondary_colors_list, pattern, weave = processImage(img_path)
        secondary_color = ', '.join(secondary_colors_list)

        new_row = sL.Row()
        new_row.update(sku=img_name, Product_Name=fancy_color, color_filter_primary=primary_color,
                       color_filter_secondary=secondary_color, pattern=pattern, weave=weave)
        sL.insertRow(new_row)
        sL.save()

        print("---Name---\n"
              f"{fancy_color}")
        print("---Primary Color---\n"
              f"{primary_color}")
        print("---Secondary Colors---\n"
              f"{secondary_color}")
        print("---Pattern---\n"
              f"{pattern}")
        print("---Weave---\n"
              f"{weave}")
