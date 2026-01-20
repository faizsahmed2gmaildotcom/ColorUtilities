from os import listdir, path, mkdir
from config import config
from shutil import rmtree
from pixelLib import removeWhiteBackground, flattenArrayOfTuples
from PIL import Image
from pixelLib import getPixelList

DIVIJ_DATASET = True

def prepare():
    if not path.exists("processed-images"):
        mkdir("processed-images")

    if path.exists(path.join("processed-images", "training-data")):
        rmtree(path.join("processed-images", "training-data"))
    mkdir(path.join("processed-images", "training-data"))


    for dirname in listdir("training-data"):
        new_dir_path = path.join("processed-images", "training-data", dirname)
        if not path.exists(new_dir_path):
            mkdir(new_dir_path)

        old_dir_path = path.join("training-data", dirname)
        for f_name in listdir(old_dir_path):
            if DIVIJ_DATASET and (not f_name.split('.')[0].endswith(config["general"]["main_image_suffix"])): continue

            pixels = getPixelList(path.join(old_dir_path, f_name), 1)
            pixels = removeWhiteBackground(pixels)

            cf_img = Image.new("RGB", (len(pixels[0]), len(pixels)))
            flattened_pixels = flattenArrayOfTuples(pixels)
            if flattened_pixels.shape[0] == 0: continue
            cf_img.putdata(list(map(tuple, flattened_pixels.tolist())))
            cf_img.save(path.join(new_dir_path, f_name), format="jpeg")

if __name__ == '__main__':
    prepare()
