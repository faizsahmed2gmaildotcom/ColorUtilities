import os, random

def randomlyDelete(folder_path: str, max_size: int):
    to_delete = random.sample(os.listdir(folder_path), max(0, len(os.listdir(folder_path)) - max_size))
    for fn in to_delete:
        fp = os.path.join(folder_path, fn)
        print(f"Deleting {fp}... ", end='', flush=True)
        os.remove(fp)
        print("Deleted!")

if __name__ == '__main__':
    randomlyDelete("training-data/shirting/main/solid", 500)
