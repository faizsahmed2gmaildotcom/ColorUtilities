import pixelLib as pL
import spreadsheetLib as sL
from kmeans import kmeans
import numpy as np
from PIL import Image
import os

image_folder_name = "test-images"
vertical_offset = 80
horizontal_offset = 80
min_ratio_in_radius = 0.0001
num_kmeans_centers = 1
kmeans_alternative_radius = 30
median_filter_size = 5
salient_pixel_bias = 10
images_scale_factor = 0.25


def getPixelList(path: str, scale: float = images_scale_factor) -> np.ndarray[tuple[int]]:
    """
    Returns the list of pixels from an image
    :param scale: Scale the image by this factor to decrease processing times
    :param path: Relative path to image
    :return: List of pixels from the image at path
    """
    img = Image.open(path).convert("RGB")
    if scale != 1:
        img.thumbnail((img.width * scale, img.height * scale), Image.Resampling.LANCZOS)
    pixel_data = np.asarray(img)
    img.close()

    return pixel_data


if __name__ == "__main__":
    color_codes = set()  # Debug

    for img_file_name in sorted(os.listdir(image_folder_name)):
        img_name = os.path.splitext(img_file_name)[0]
        print('\n' + img_name)
        img_name_new = img_name.removesuffix("_1f")
        if img_name == img_name_new:
            print("Skipping: not main image...")
            continue
        img_name = img_name_new

        img_path = os.path.join(image_folder_name, img_file_name)
        pixels = getPixelList(img_path)
        pixels = pL.removeWhitePixels(pixels)
        pixels = pL.cropPixels(pixels, vertical_offset, len(pixels) - 1 - vertical_offset, horizontal_offset, len(pixels[0]) - 1 - horizontal_offset)
        pixels = pL.medianFilter(pixels, median_filter_size)
        # pixels = uF.spreadSalientPixels(pixels, salient_pixel_bias)
        # pixels = uF.spreadDominantPixels(pixels, salient_pixel_bias, 3)
        # pixels = uF.enhanceWhitePoint(pixels)

        cf_img = Image.new("RGB", (len(pixels[0]), len(pixels)))
        flattened_pixels = pL.flattenArray(pixels)
        cf_img.putdata(list(map(tuple, flattened_pixels.tolist())))
        cf_img.save(os.path.join("processing-images", img_file_name), format="jpeg")  # Debug

        kmeans_results = kmeans(pL.getFrequency(flattened_pixels), num_kmeans_centers)

        new_row = sL.Row()
        fancy_colors = ", ".join(set([pL.getNearestColorName(c, "fancy-colors") for c in kmeans_results]))
        primary_colors = ", ".join(set([pL.getNearestColorName(c, "primary-colors") for c in kmeans_results]))
        new_row.update(sku=img_name, Product_Name=fancy_colors, color_filter_primary=primary_colors)
        sL.insertRow(new_row)
        sL.save()

        print("---Name---\n"
              f"{fancy_colors}")
        print("---Primary Color---\n"
              f"{primary_colors}")

        color_codes.update({str(list(map(lambda c: (c // 10) * 10, kmeans_results[0])))})  # Debug
    print(color_codes)  # Debug
