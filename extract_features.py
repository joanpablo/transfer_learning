# This script is responsible of extract the feature vectors from the dataset
# and store then for later classification training.
# Usage: python extract_features.py

from tensorflow.keras.applications import VGG16

# load VGG16 model previously trained with ImageNet dataset
model = VGG16(weights="imagenet", include_top=False)

