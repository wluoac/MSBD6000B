import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.applications.xception import Xception

def tfInit():
    config = tf.ConfigProto()
    set_session(tf.Session(config=config))


def train(epochs):
    image_size = (299,299)
    # variables to hold features and labels
    features = []
    labels   = []
    
    # default setting in keras models
    class_count = 1000
    X_test = []
    name_test = []

    trainData = np.loadtxt("./train.txt", dtype="str", delimiter=' ')
    for k in range(len(trainData)):
        aLine = trainData[k]
        image_path = aLine[0]
        label = int(aLine[1])
        ground_truth = np.zeros(class_count, dtype=np.float32)
        ground_truth[label] = 1

        img = image.load_img(image_path, target_size=image_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        labels.append(ground_truth)
        features.append(x[0])

    trainData = np.loadtxt("./val.txt", dtype="str", delimiter=' ')
    for k in range(len(trainData)):
        aLine = trainData[k]
        image_path = aLine[0]
        label = int(aLine[1])
        ground_truth = np.zeros(class_count, dtype=np.float32)
        ground_truth[label] = 1

        img = image.load_img(image_path, target_size=image_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        labels.append(ground_truth)
        features.append(x[0])

    testData = np.loadtxt("./test.txt", dtype="str", delimiter=' ')
    for k in range(len(testData)):
        aLine = testData[k]
        image_path = aLine
        img = image.load_img(image_path, target_size=image_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        X_test.append(x[0])
        name_test.append(image_path)

    X_train = features
    y_train = labels

    X_train = np.array(X_train)
    Y_train = np.array(y_train)
    
    # test image
    X_test = np.array(X_test)

    # Use Xception
    model = Xception(include_top=True, weights='imagenet', classes=class_count)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
    model.fit(X_train, Y_train, epochs=epochs, verbose=1, validation_split=0.3)

    Y_pred = model.predict(X_test)
    
    f = open('project2_08573584.txt', 'w')
    for i in range(len(name_test)):
        predict = Y_pred[i].argmax(axis=0)
        f.write(str(predict) + '\n')
    f.close()


tfInit()
train(epochs=30)
