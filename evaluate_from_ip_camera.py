from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications import VGG16
from deeplearning import config
import numpy as np
import imutils
import pickle
import cv2
import sys

# load the network model and label binarizer
print("[INFO] loading network and label binarizer...")
network = VGG16(weights="imagenet", include_top=False)
lb = pickle.loads(open(config.LE_PATH, "rb").read())

# create the logistic regression model for evaluation
print('[INFO] creating model for training...')
model = pickle.loads(open(config.MODEL_PATH, "rb").read())

# connecting to IP camera
print('[INFO] connecting to IP camera...')
video = cv2.VideoCapture(config.IP_CAMERA_URL)
if not video.isOpened():
    sys.exit('Failed to connect to IP camera...')

# loop while receiving frames from camera
print('[INFO] successfully connected to camera...')
while True:
    # read frame from IP camera
    _, frame = video.read()

    # resize frame to fit the configured height
    output = imutils.resize(frame, height=600)

    # apply sharpening convolution and resize image
    frame = cv2.resize(frame, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))

    # convert image from BGR to RGB because model was training that way
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # subtracting RGB pixel intensity mean from ImageNet dataset to the image
    frame = np.expand_dims(frame, axis=0)
    frame = imagenet_utils.preprocess_input(frame)

    # extract features from frame image
    features = network.predict(frame)
    features = features.reshape((features.shape[0], features.shape[1] * features.shape[2] * features.shape[3]))

    # evaluate the model
    predictions = model.predict(features)

    # find the class label index with the largest corresponding probability
    label = lb.classes_[predictions[0]]

    # draw the class label + probability on the output image
    text = "{}".format(label)
    cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow('image', output)

    # if 'q' key is pressed then exit program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
video.release()
cv2.destroyAllWindows()
