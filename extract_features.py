# This script is responsible of extract the feature vectors from the dataset
# and store then for later classification training.
# Usage: python extract_features.py
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications import VGG16
from sklearn.preprocessing import LabelEncoder
from deeplearning import config
from imutils import paths
from tqdm import tqdm
import numpy as np
import random
import os

# load VGG16 model previously trained with ImageNet dataset
model = VGG16(weights="imagenet", include_top=False)

le = None

# create output directory is it doesn't exists
if not os.path.exists(config.CSV_PATH):
    os.makedirs(config.CSV_PATH)

# loop over dataset splits for extracting features of all images inside each one
for split in (config.TRAIN_PATH, config.VAL_PATH, config.TEST_PATH):
    print('[INFO] extracting features from {} images'.format(split))
    # read image paths with in split directory
    split_dir = os.path.sep.join([config.DATASET_PATH, split])
    image_paths = list(paths.list_images(split_dir))

    # random shuffle image paths and extract class labels from paths
    random.seed(42)
    random.shuffle(image_paths)
    labels = [p.split(os.path.sep)[-2] for p in image_paths]

    # if the label encoder is None, create it
    if le is None:
        le = LabelEncoder()
        le.fit(labels)

    # create the cvs file to store features for the split
    csv_path = os.path.sep.join([config.CSV_PATH, '{}.csv'.format(split)])
    csv = open(csv_path, 'w')

    # loop over image paths in form of batches for extracting features
    progress = tqdm(total=len(image_paths))
    for i in range(0, len(image_paths), config.BATCH_SIZE):
        paths_batch = image_paths[i: i + config.BATCH_SIZE]
        labels_batch = labels[i: i + config.BATCH_SIZE]
        labels_batch = le.transform(labels_batch)
        images_batch = []
        # loop over images path in current paths batch
        for path in paths_batch:
            # load image using Keras utility functions while ensuring the size of images are 128x128
            image = load_img(path, target_size=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
            image = img_to_array(image)

            # subtracting RGB pixel intensity mean from ImageNet dataset to the image
            image = np.expand_dims(image, axis=0)
            image = imagenet_utils.preprocess_input(image)[0]

            # add processed image to batch of images
            images_batch.append(image)

        # pass the images through the network for extracting features
        # and flatter the resulting features
        images_batch = np.array(images_batch)
        features = model.predict(images_batch)
        features = features.reshape((features.shape[0], features.shape[1] * features.shape[2] * features.shape[3]))

        # writing features to csv file in a format of <label>,<features> per line
        for label, vector in zip(labels_batch, features):
            vec = ','.join([str(v) for v in vector])
            csv_row = '{}, {}\n'.format(label, vec)
            csv.write(csv_row)

        # updating the progress bar
        progress.update(config.BATCH_SIZE)

    # closing and freeing resources
    progress.close()
    csv.close()
