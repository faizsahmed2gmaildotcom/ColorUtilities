import os, random, shutil
from typing import Literal

# WARNING: DELETES AND RE-GENERATES STRIPES & CHECKS FOLDERS IN MAIN
# Gets check & stripes dataset in for shirting/main
main_path = '../training-data/shirting'


def getDistribution(mode: Literal['check', 'stripes'], num_imgs: int):
    mode_fp = f'{main_path}/{mode}/main'
    num_classes = len(os.listdir(mode_fp))
    imgs_per_class = num_imgs // num_classes
    classes = sorted(mode_fp + '/' + fn for fn in os.listdir(mode_fp))
    sizes = [len(os.listdir(fp)) for fp in classes]
    class_imgs = [min(imgs_per_class, s) for s in sizes]

    i = -1
    can_add = False
    while sum(class_imgs) < num_imgs:
        i = (i + 1) % num_classes
        if i == 0: can_add = False
        if class_imgs[i] < sizes[i]:
            class_imgs[i] += 1
            can_add = True
        if (not can_add) and (i == (num_classes - 1)):
            print("Unable to fully populate!")
            break

    return zip(classes, class_imgs)


def populate(mode: Literal['check', 'stripes'], distrib: zip):
    result_path = main_path + '/main/' + mode
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.mkdir(result_path)

    for fp, n in distrib:
        for fn in random.sample(os.listdir(fp), n):
            shutil.copy(fp + '/' + fn, result_path + '/' + fn)


if __name__ == '__main__':
    populate('check', getDistribution('check', 500))
    populate('stripes', getDistribution('stripes', 500))
