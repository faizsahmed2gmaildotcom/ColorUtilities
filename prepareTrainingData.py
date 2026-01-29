import os
from time import time
from config import config
from pixelLib import preprocessImage
import pandas as pd

DIVIJ_DATASET = False
main_img_ext = (config["general"]["main_image_suffix"] if DIVIJ_DATASET else "") + ".jpeg"

# Format: {ext: {filePath: fileName}}
file_names = {
    main_img_ext: {},
    ".jpg": {},
    ".png": {},
    ".xlsx": {}
}

PATTERN_MAPS = {"checks": "check", "windowpane checks": "windowpane check", "striped": "stripes", "glen checks": "glen check", "poppytooth checks": "houndstooth",
                "puppytooth checks": "houndstooth", "puppytooth check": "houndstooth", "houndstooth checks": "houndstooth", "gingham": "gingham checks",
                "grid checks": "grid check",
                "gun club checks": "gun club check", "solids": "solid"}
VALID_COLS = [["sku"], ["pattern", "pattern_primary"]]  # Order is important! Can use a list of column names to select any one of them.
TRAINING_DATA_PATH = "training-data"


def getAllFpInDir(dir_path: str):
    if not os.path.exists(dir_path):
        raise FileNotFoundError(dir_path + " directory does not exist")
    if os.path.isdir(dir_path):
        for fp in os.listdir(dir_path):
            getAllFpInDir(os.path.join(dir_path, fp))
    else:
        for ext in file_names:
            if dir_path.endswith(ext):
                file_names[ext].update({os.path.basename(dir_path).removesuffix(ext): dir_path})


def prepareDivij():
    getAllFpInDir("Divij.com")
    for sheet_path in file_names[".xlsx"].values():
        try:
            spreadsheet = pd.DataFrame(pd.read_excel(sheet_path))
        except (ValueError, TypeError):
            print(f"Skipping {sheet_path}: cannot open!")
            continue

        found_valid_cols = []
        for VC in VALID_COLS:
            for c in VC:
                if c in set(spreadsheet.columns):
                    found_valid_cols.append(c)
                    continue  # Uses the first column with the same meaning that it comes across

        if (len(found_valid_cols) == len(VALID_COLS)) and (not spreadsheet.empty):
            print(f"{sheet_path}: valid format, processing...")
            imgs_processed = 0
            total_time = 0.0

            for _, row in spreadsheet[found_valid_cols].iterrows():
                sku = row[found_valid_cols[0]]
                pattern = row[found_valid_cols[1]]

                if (sku in file_names[main_img_ext]) and isinstance(pattern, str):
                    pattern = pattern.lower().strip()
                    if pattern in PATTERN_MAPS: pattern = PATTERN_MAPS[pattern]
                    # print(f"Found {sku} at {file_names[img_ext][sku]}")
                    class_dir = os.path.join("training-data", pattern)
                    if not os.path.exists(class_dir):
                        os.mkdir(class_dir)

                    result_path = os.path.join(class_dir, sku + ".jpeg")
                    if not os.path.exists(result_path):
                        start = time()
                        preprocessImage(file_names[main_img_ext][sku], result_path)

                        total_time += time() - start
                        imgs_processed += 1
                    print(f"\rImages processed: {imgs_processed}; {total_time / imgs_processed if imgs_processed > 0 else 0:.3f}s/image", end="", flush=True)
            print()

        else:
            print(f"Skipping {sheet_path}: invalid format!")


def prepare():
    getAllFpInDir("training-data")
    for ext in file_names:
        for img_path in sorted(file_names[ext].values()):
            print(f"Checking {img_path}...")
            preprocessImage(img_path)


if __name__ == '__main__':
    prepare()
    # getAllFpInDir("temp")
    # for ext in file_names:
    #     for img_path in sorted(file_names[ext].values()):
    #         cropImage(img_path, (41, None, None, None))
