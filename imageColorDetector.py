from config import config
import pixelLib as pL
import spreadsheetLib as sL
from kmeans import kmeans
from PIL import Image
import os

image_folder_name = "test-images"
vertical_offset = 10
horizontal_offset = 10
num_kmeans_centers = 1
median_filter_size = 5
salient_pixel_bias = 10
img_scale_factor = 0.25

if __name__ == "__main__":
    for img_file_name in sorted(os.listdir(image_folder_name)):
        img_name = os.path.splitext(img_file_name)[0]
        print('\n' + img_name)
        img_name_new = img_name.removesuffix(config["general"]["main_image_suffix"])
        if img_name == img_name_new:
            print("Skipping: not main image...")
            continue
        img_name = img_name_new

        img_path = os.path.join(image_folder_name, img_file_name)
        pixels = pL.getPixelList(img_path, img_scale_factor)
        pixels = pL.removeWhiteBackground(pixels)
        pixels = pL.cropPixels(pixels, vertical_offset, len(pixels) - 1 - vertical_offset, horizontal_offset, len(pixels[0]) - 1 - horizontal_offset)
        pixels = pL.medianFilter(pixels, median_filter_size)
        pixels = pL.spreadSalientPixels(pixels, salient_pixel_bias)

        cf_img = Image.new("RGB", (len(pixels[0]), len(pixels)))  # Debug
        flattened_pixels = pL.flattenArrayOfTuples(pixels)
        cf_img.putdata(list(map(tuple, flattened_pixels.tolist())))  # Debug
        cf_img.save(os.path.join("processed-images", img_file_name), format="jpeg")  # Debug

        kmeans_result = kmeans(pL.getFrequency(flattened_pixels), num_kmeans_centers)
        secondary_kmeans_results = kmeans(pL.getFrequency(flattened_pixels), 2)

        new_row = sL.Row()
        fancy_color = pL.getNearestColorName(kmeans_result[0], "fancy-colors")
        primary_color = pL.getNearestColorName(kmeans_result[0], "primary-colors")
        secondary_color = set([pL.getNearestColorName(c, "primary-colors") for c in secondary_kmeans_results])
        secondary_color = secondary_color.difference({primary_color}).pop() if (len(secondary_color) == 2) and (primary_color in secondary_color) else None
        if (secondary_color is not None) and (not pL.isSignificantlyDifferent(secondary_kmeans_results.pop(), secondary_kmeans_results.pop())): secondary_color = None
        new_row.update(sku=img_name, Product_Name=fancy_color, color_filter_primary=primary_color, color_filter_secondary=secondary_color)
        sL.insertRow(new_row)
        sL.save()

        print("---Name---\n"
              f"{fancy_color}")
        print("---Primary Color---\n"
              f"{primary_color}")
        print("---Secondary Color---\n"
              f"{secondary_color}")
