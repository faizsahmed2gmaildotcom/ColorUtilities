import os
from MLMTrainerPyTorchConvnext import train_transform_pattern, visualize_transform_samples, containsDir, getFirstFP


def visualizeAll(_training_dirname="", _depth=0):
    training_dirpath = os.path.join('training-data', _training_dirname)
    if not containsDir(getFirstFP(training_dirpath)):
        return True

    if not os.path.exists(os.path.join('models', _training_dirname)):
        os.mkdir(os.path.join('models', _training_dirname))

    if (_depth > 0) and not os.path.exists(os.path.join(training_dirpath, 'main')):
        raise FileNotFoundError(f"{training_dirpath}/main does not exist!")

    for dirname in os.listdir(training_dirpath):
        cur_training_dir = os.path.join(_training_dirname, dirname)
        if visualizeAll(cur_training_dir, _depth + 1):
            visualize_transform_samples(os.path.join('training-data', cur_training_dir), train_transform_pattern, 6)

    return False

if __name__ == '__main__':
    for _ in range(2):
        visualizeAll("shirting/check")
