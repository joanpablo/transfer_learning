import os

# initialize the path to the original directory of *images*
ORIGIN_IMAGES = 'images'

# set the dataset base path
DATASET_PATH = './dataset'

# define name for train, validation and test directories
TRAIN_PATH = 'training'
VAL_PATH = 'validation'
TEST_PATH = 'evaluation'

# set the output directory
OUTPUT = 'output'

# initialize the list of classes labels
CLASSES = ['backpackbag', 'bottles', 'cup', 'keyboard']

# set the batch size
BATCH_SIZE = 32

# set the images size
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

# set the amount of samples per class
AMOUNT_PER_CLASS = 500

# initialize the label enconder file to store labels after encoding
LE_PATH = os.path.sep.join([OUTPUT, 'le.cpickle'])

# initialize the output directory where to store extracted features in CSV format
CSV_PATH = os.path.sep.join([OUTPUT, 'features'])

# set the path to the serialized model after training
MODEL_PATH = os.path.sep.join([OUTPUT, 'model.h5'])
