import os, shutil
from time import time
from pixelLib import preprocessImage
import pandas as pd
import re
import ast

# Format: {ext: {fileName: filePath}}
DEFAULT_FILENAME_DICT: dict[str, dict[str, str]] = {
    '.jpeg': {},
    '.jpg': {},
    '.png': {},
    '.xlsx': {}
}

out_dir = os.path.join("processed-images", "training-data")
if not os.path.exists(out_dir): os.mkdir(out_dir)


class FileDict(dict):
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.clear()
        self.update(DEFAULT_FILENAME_DICT)


file_names = FileDict()

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


def prepareDivijExcel():
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
                sku: str = row[found_valid_cols[0]]
                pattern: str = row[found_valid_cols[1]]

                if (sku in file_names['.jpeg']) and isinstance(pattern, str):
                    pattern = pattern.lower().strip()
                    # print(f"Found {sku} at {file_names[img_ext][sku]}")
                    class_dir = os.path.join("training-data", pattern)
                    if not os.path.exists(class_dir):
                        os.mkdir(class_dir)

                    result_path = os.path.join(class_dir, sku + ".jpeg")
                    if not os.path.exists(result_path):
                        start = time()
                        preprocessImage(file_names['.jpeg'][sku], result_path)

                        total_time += time() - start
                        imgs_processed += 1
                    print(f"\rImages processed: {imgs_processed}; {total_time / imgs_processed if imgs_processed > 0 else 0:.3f}s/image", end="", flush=True)
            print()

        else:
            print(f"Skipping {sheet_path}: invalid format!")


def prepareDivijSQL(preprocess):
    getAllFpInDir("product_images")
    if preprocess:
        for ext in file_names:
            for img_name, img_path in sorted(file_names[ext].items(), key=lambda item: item[1]):
                # print(f"Checking {img_path}...")
                if not img_name.endswith("_1f"):
                    os.remove(img_path)
                    continue
                preprocessImage(img_path)

        file_names.reset()
        getAllFpInDir("product_images")

    with open("product_data.txt") as f:
        contents = f.readlines()
        f.close()

    masters = {}
    idx = {
        'sku': 1,
        'url_key': 37,
        'pattern_id': 18,
        'weave_id': 19,
        'composition_id': 21
    }

    for l in contents:
        raw_data = re.search(r"INSERT INTO (`.*_master`) VALUES (.*)", l)
        if raw_data:
            master_name = raw_data.group(1).strip('`')
            master_contents: dict[int, str] = dict(map(lambda k_v: (int(k_v.split(',')[0]), k_v.split(',')[1].strip("'")), raw_data.group(2).strip('();').split('),(')))
            masters.update({master_name.removesuffix("_master") + "_id": master_contents})

    for pattern in masters['pattern_id'].values():
        pattern_dir = os.path.join(out_dir, pattern)
        if not os.path.exists(pattern_dir): os.mkdir(pattern_dir)

    def idToVal(id_type: str, _id: int) -> str | int:
        if id_type not in masters: return _id
        master_table = masters[id_type]
        if _id not in master_table: return _id
        return master_table[_id]

    img_skus = sorted(file_names['.jpeg'].keys())
    for l in contents:
        raw_data = re.search(r"INSERT INTO `products` VALUES (.*)", l)
        if raw_data:
            raw_data = raw_data.group(1).strip('();').split('),(')
            for entry in raw_data:
                parsed_entry = ast.literal_eval(f"[{entry.replace('NULL', 'None').replace("_binary", '')}]")
                img_data = {key: idToVal(key, parsed_entry[val]) for key, val in idx.items()}  # Convert IDs to their values
                img_data['sku'] = '-'.join(img_data['sku'].lower().split('-')[:2])
                if img_data['pattern_id'] is None: continue

                for sku in img_skus:
                    if img_data['sku'] in sku.lower():
                        copy_path = os.path.join(out_dir, img_data['pattern_id'], img_data['sku'] + '.jpeg')
                        try:
                            shutil.copy2(file_names['.jpeg'][sku], copy_path)
                        except FileNotFoundError:
                            print(f"WARNING: {img_data} MISSING")
                        break


def prepare():
    getAllFpInDir("test-images")
    for ext in file_names:
        for img_path in sorted(file_names[ext].values()):
            print(f"Checking {img_path}...")
            preprocessImage(img_path)


if __name__ == '__main__':
    prepareDivijSQL(False)
    # getAllFpInDir("temp")
    # for ext in file_names:
    #     for img_path in sorted(file_names[ext].values()):
    #         cropImage(img_path, (41, None, None, None))
