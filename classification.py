"""
@ author: S.J.Huang
"""

from keras.models import Sequential
from keras.preprocessing.image import img_to_array, array_to_img
from keras.utils import np_utils
from keras.models import Sequential  
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,BatchNormalization

import os
import pandas as pd
import numpy as np
import csv
import utils.mnist_reader as mnist_reader
import argparse

def CNN_model():
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                    activation ='relu', input_shape = (28,28,1)))
    model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same',
                    activation ='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same',
                    activation ='relu'))
    model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same',
                    activation ='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',
                    activation ='relu'))
    model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',
                    activation ='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(512, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = "softmax"))

    return model

def parse_args() -> argparse:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',                default='0', type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # -------- Read data ---------#
    train_x, train_t = mnist_reader.load_mnist('data/fashion', kind='train')
    test_x, test_t = mnist_reader.load_mnist('data/fashion', kind='t10k')

    # ------ Preprocess data -----#
    x_train = train_x.reshape(train_x.shape[0], 28, 28, 1).astype('float32')
    x_train_norm = x_train / 255.0
    t_train_onehot = np_utils.to_categorical(train_t)

    x_test = test_x.reshape(test_x.shape[0], 28, 28, 1).astype('float32')
    x_test_norm = x_test / 255.0

    # ------- Model training----- #
    model = CNN_model()
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    train_history = model.fit(x=x_train_norm, y=t_train_onehot,
                            epochs=100, batch_size=300, verbose=1)
    # train_history = model.fit(x=x_train_norm, y=t_train_onehot, epochs=20, batch_size=600, verbose=1)
    # y_test = model.predict_classes(x_test_norm)
    y_test = model.predict(x_test_norm)

    arg_ytest = np.argmax(y_test, axis=1) # we can use this code for one-hot encoding
    acc = np.sum(arg_ytest == test_t) / len(arg_ytest)  # count correct prediction
    print("accuracy is {}".format(acc))

    # with open('answer.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['id', 'label'])
    #     for i in range(len(y_test)):
    #         writer.writerow((i, y_test[i]))
    #     print("Answer is saved in csv.")

