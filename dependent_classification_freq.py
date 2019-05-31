"""
@ author: S.J.Huang
"""

from keras.models import Sequential
from keras.preprocessing.image import img_to_array, array_to_img
from keras.utils import np_utils
from keras.models import Sequential  
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,BatchNormalization
from keras.optimizers import Adam
# import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import csv
import utils.mnist_reader as mnist_reader

def dependency_check():
    file_path = './euijin_code/one-hot-testresult.txt'
    # file_path = '/home/soopil/Desktop/github/fashion-mnist/euijin_code/one-hot-testresult.txt'
    fd = open(file_path)
    lines = fd.readlines()
    contents = []
    for i, line in enumerate(lines):
        if i != 0 and line != '\n':
            contents.append(line.replace('\n', '').replace(':', ''))
    depend_array = []
    for i in range(len(contents)):
        # print(contents[i])
        assert len(contents[i]) != 0
        if i % 3 == 0:
            new_line = [contents[i],
                        contents[i + 1].split(',')[0].split('=')[1], contents[i + 1].split(',')[1].split('=')[1],
                        contents[i + 2].split(',')[0].split('=')[1], contents[i + 2].split(',')[1].split('=')[1]]
            print(new_line)
            depend_array.append(new_line)
    return depend_array

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
    model.add(Dense(8, activation = "sigmoid"))  #change softmax to sigmoid for dependancy labeling

    return model

def check_freq(index_list):    # to check how much each class is counted
      total=np.zeros(10)
      for i in range(len(index_list)):
          if index_list[i]==0:
              total[0]+=1
          elif index_list[i]==1:
              total[1]+=1
          elif index_list[i]==2:
              total[2]+=1
          elif index_list[i]==3:
              total[3]+=1
          elif index_list[i]==4:
              total[4]+=1
          elif index_list[i]==5:
              total[5]+=1
          elif index_list[i]==6:
              total[6]+=1
          elif index_list[i]==7:
              total[7]+=1
          elif index_list[i]==8:
              total[8]+=1
          elif index_list[i]==9:
              total[9]+=1

      num = np.argmax(total)
      percent = total[num]/np.sum(total)
      return print("number ={}, Frequency={}".format(num,percent))

def encode_dependancy(onehot):
    depend_list = np.array(dependency_check())
    freq = np.delete(depend_list, 3, 1)
    freq = np.delete(freq, 1, 1)
    freq = np.delete(freq, 0, 1)
    freq = freq.astype(np.float32)
    # for f in freq:
    #     print(f)

    # assert False
    depend_list = np.delete(depend_list, -3, 1)
    depend_list = np.delete(depend_list, -1, 1)
    depend_list = depend_list.astype(np.int32)

    for i in range(10):
        for j in range(2):
            if freq[i][j] < 0.4:
                freq[i][j] = -1
                depend_list[i][j+1] = -1
    depend_list = np.delete(depend_list, 7, 0)
    depend_list = np.delete(depend_list, 5, 0)
    depend_list = np.delete(depend_list, 4, 0)

    for i, e in enumerate(depend_list):
        print(i, e)

    depend_array = depend_list
    label_list = [[0 for _ in range(8)] for k in range(10)]

    for i in range(8):
        for j in range(3):
            if j == 0:
                label_list[int(depend_array[i][j])][i] = 1
            else:
                label_list[int(depend_array[i][j])][i] = float(freq[int(depend_array[i][j])][j-1])

    print(label_list)
    assert False
    # assert False
    t_depend_encode= np.zeros((len(onehot),8))
    for i in range(len(onehot)):
        if list(onehot[i])==[1,0,0,0,0,0,0,0,0,0]:#0
            t_depend_encode[i]=label_list[0]

        elif list(onehot[i])==[0,1,0,0,0,0,0,0,0,0]:#1
            t_depend_encode[i]=label_list[1]

        elif list(onehot[i])==[0,0,1,0,0,0,0,0,0,0]:#2
            t_depend_encode[i]=label_list[2]

        elif list(onehot[i])==[0,0,0,1,0,0,0,0,0,0]:#3
            t_depend_encode[i]=label_list[3]

        elif list(onehot[i])==[0,0,0,0,1,0,0,0,0,0]:#4
            t_depend_encode[i]=label_list[4]

        elif list(onehot[i])==[0,0,0,0,0,1,0,0,0,0]:#5
            t_depend_encode[i]=label_list[5]

        elif list(onehot[i])==[0,0,0,0,0,0,1,0,0,0]:#6
            t_depend_encode[i]=label_list[6]

        elif list(onehot[i])==[0,0,0,0,0,0,0,1,0,0]:#7
            t_depend_encode[i]=label_list[7]

        elif list(onehot[i])==[0,0,0,0,0,0,0,0,1,0]:#8
            t_depend_encode[i]=label_list[8]

        elif list(onehot[i])==[0,0,0,0,0,0,0,0,0,1]:#9
            t_depend_encode[i]=label_list[9]
    return t_depend_encode

def measure_distance(prediction,label):
    distance = np.sum(np.abs(prediction - label))
    return distance

def shortest_distance(prediction_array):
    depend_list = np.array(dependency_check())
    depend_list = np.delete(depend_list, -4, 0)
    depend_list = np.delete(depend_list, -1, 0)
    depend_array = depend_list[:, :2]
    label_list = [[0 for _ in range(8)] for k in range(10)]
    for i in range(8):
        for j in range(2):
            label_list[int(depend_array[i][j])][i] = 1
    t_depend_encode = label_list

    distance_arr=np.zeros((len(prediction_array),10))
    final_class=list()
    for i in range(len(prediction_array)):
        for j in range(10):
            distance_arr[i, j] = measure_distance(prediction_array[i], t_depend_encode[j])
        #
        # distance_arr[i,0]=measure_distance(prediction_array[i],np.array([1,0,1,1,0,0]))
        # distance_arr[i,1]=measure_distance(prediction_array[i],np.array([0,1,1,0,1,0]))
        # distance_arr[i,2]=measure_distance(prediction_array[i],np.array([0,1,1,1,0,0]))
        # distance_arr[i,3]=measure_distance(prediction_array[i],np.array([0,1,1,0,0,0]))
        # distance_arr[i,4]=measure_distance(prediction_array[i],np.array([1,1,1,0,0,0]))
        # distance_arr[i,5]=measure_distance(prediction_array[i],np.array([0,0,0,0,1,0]))
        # distance_arr[i,6]=measure_distance(prediction_array[i],np.array([1,1,1,1,0,0]))
        # distance_arr[i,7]=measure_distance(prediction_array[i],np.array([0,0,0,0,0,1]))
        # distance_arr[i,8]=measure_distance(prediction_array[i],np.array([0,0,1,0,1,1]))
        # distance_arr[i,9]=measure_distance(prediction_array[i],np.array([0,0,0,0,1,1]))

    for i in range(len(distance_arr)):
        final_class.append(np.argmin(distance_arr[i]))

    return np.array(final_class)

if __name__ == "__main__":
    epoch=600
    LR=0.001
    # -------- Read data ---------#
    train_x, train_t = mnist_reader.load_mnist('data/fashion', kind='train')
    test_x, test_t = mnist_reader.load_mnist('data/fashion', kind='t10k')

    # ------ Preprocess data -----#
    x_train = train_x.reshape(train_x.shape[0], 28, 28, 1).astype('float32')
    x_train_norm = x_train / 255.0
    t_train_onehot = np_utils.to_categorical(train_t)

    t_depend_encode = encode_dependancy(t_train_onehot) #change one-hot to dependancy labeling

    x_test = test_x.reshape(test_x.shape[0], 28, 28, 1).astype('float32')
    x_test_norm = x_test / 255.0

    # ------- Model training----- #
    model = CNN_model()
    adam = Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, decay=LR*(1/epoch), amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
    history = model.fit(x=x_train_norm, y=t_depend_encode, epochs=epoch, batch_size=6000, verbose=1)

    # plt.plot(history.history['loss'])
    # plt.title('Model accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['loss'], loc='upper left')
    # plt.show()

    y_test = model.predict(x_test_norm)
#    arg_ytest = np.argmax(y_test, axis=1) # we can use this code for one-hot encoding
    arg_ytest = shortest_distance(y_test) # distance measure and return shortest argument for dependancy result
    acc = np.sum(arg_ytest==test_t)/len(arg_ytest) #count correct prediction
    print("accuracy is {}".format(acc))