# This script is responsible for loading extracting features and train the model.
# Usage: python train.py

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from deeplearning import config
import numpy as np
import pickle
import os


def load_split_data(split_data_filename):
    # open split csv file
    split_data_file = open(split_data_filename)

    # initialize labels and features arrays
    split_labels = []
    split_features = []

    # loop over lines in data split
    for line in split_data_file.readlines():
        # extract the label and the features from the line
        columns = line.split(',')
        label = int(columns[0])
        features = np.array(columns[1:], dtype='float')

        # append label and features to collections
        split_labels.append(label)
        split_features.append(features)

    # close csv file
    split_data_file.close()

    # convert labels and features arrays to numpy array
    split_labels = np.array(split_labels)
    split_features = np.array(split_features)

    return split_features, split_labels


# load label encoder from filesystem
print('[INFO] loading label encoder...')
le = pickle.load(open(config.LE_PATH, 'rb'))

# create model for training
print('[INFO] creating model for training...')
model = LogisticRegression(verbose=1)

# load train data
print('[INFO] loading train features from CSV file...')
train_data_file = os.path.sep.join([config.CSV_PATH, '{}.csv'.format(config.TRAIN_PATH)])
train_data, train_label = load_split_data(train_data_file)

# load test data
print('[INFO] loading test features from CSV file...')
test_data_file = os.path.sep.join([config.CSV_PATH, '{}.csv'.format(config.TEST_PATH)])
test_data, test_label = load_split_data(train_data_file)

# train the model
print('[INFO] training the model')
model.fit(train_data, train_label)

# evaluate the model
print('[INFO] evaluating the model')
predictions = model.predict(test_data)

# show evaluation reports
print(classification_report(test_label, predictions, target_names=le.classes_))

# save the model to filesystem
model_file = open(config.MODEL_PATH, 'wb')
model_file.write(pickle.dumps(model))
model_file.close()
