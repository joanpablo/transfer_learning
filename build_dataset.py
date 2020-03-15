# This script is for creating the dataset directory from the original images directory
# Usage: python build_dataset.py

from sklearn.model_selection import train_test_split
from deeplearning import config
from tqdm import tqdm
import random
import shutil
import os

# loop over each class directory and extract the amount of images files
# specified in 'config.AMOUNT_PER_CLASS'
image_paths = []
for label in config.CLASSES:
    label_dir = os.path.sep.join([config.ORIGIN_IMAGES, label])
    images = os.listdir(label_dir)
    random.seed(42)
    random.shuffle(images)
    images = images[:config.AMOUNT_PER_CLASS]
    images = list(map(lambda image_path: os.path.sep.join([label_dir, image_path]), images))

    image_paths.extend(images)

# shuffle image path collection
random.seed(42)
random.shuffle(image_paths)

# split data in training, validation and test samples
train_paths, val_paths = train_test_split(image_paths, test_size=0.40, random_state=42)
val_paths, test_paths = train_test_split(val_paths, test_size=0.50, random_state=42)

# check if dataset directory is not created
if not os.path.exists(config.DATASET_PATH):
    os.mkdir(config.DATASET_PATH)


def build_dataset_split(file_paths, split_name):
    split_directory = os.path.sep.join([config.DATASET_PATH, split_name])
    if not os.path.exists(split_directory):
        os.mkdir(split_directory)

    progress = tqdm(total=len(file_paths))
    for path in file_paths:
        label = path.split(os.path.sep)[-2]
        filename = path.split(os.path.sep)[-1]
        dest_directory = os.path.sep.join([split_directory, label])
        if not os.path.exists(dest_directory):
            os.mkdir(dest_directory)

        dest_file = os.path.sep.join([dest_directory, filename])
        shutil.copy2(path, dest_file)
        progress.update(1)

    progress.close()


# copy train, validation and evaluation images to dataset directory
print('[INFO] copying training images...')
build_dataset_split(train_paths, config.TRAIN_PATH)
print('[INFO] copying validation images...')
build_dataset_split(val_paths, config.VAL_PATH)
print('[INFO] copying evaluation images...')
build_dataset_split(test_paths, config.TEST_PATH)
