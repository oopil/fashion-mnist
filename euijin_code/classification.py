"""
@ author: S.J.Huang
"""

from keras.models import Sequential
from keras.preprocessing.image import img_to_array, array_to_img
from keras.utils import np_utils
from keras.models import Sequential  
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,BatchNormalization
from keras.optimizers import Adam 
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import csv
import utils.mnist_reader as mnist_reader

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
    model.add(Dense(6, activation = "sigmoid"))  #change softmax to sigmoid for dependancy labeling

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
  
def encode_dependancy(onehot):  # below encoding is considered about relation of each image dataset
                                #  (example. label 5,7,9 is highly related eachother)
        t_depend_encode= np.zeros((len(onehot),6))
        for i in range(len(onehot)):
            if list(onehot[i])==[1,0,0,0,0,0,0,0,0,0]:#0
                t_depend_encode[i]=np.array([1,0,1,1,0,0])
                
            elif list(onehot[i])==[0,1,0,0,0,0,0,0,0,0]:#1
                t_depend_encode[i]=np.array([0,1,1,0,1,0])
                
            elif list(onehot[i])==[0,0,1,0,0,0,0,0,0,0]:#2
                t_depend_encode[i]=np.array([0,1,1,1,0,0])
                
            elif list(onehot[i])==[0,0,0,1,0,0,0,0,0,0]:#3
                t_depend_encode[i]=np.array([0,1,1,0,0,0])
                
            elif list(onehot[i])==[0,0,0,0,1,0,0,0,0,0]:#4
                t_depend_encode[i]=np.array([1,1,1,0,0,0])
                
            elif list(onehot[i])==[0,0,0,0,0,1,0,0,0,0]:#5
                t_depend_encode[i]=np.array([0,0,0,0,1,0])
                
            elif list(onehot[i])==[0,0,0,0,0,0,1,0,0,0]:#6
                t_depend_encode[i]=np.array([1,1,1,1,0,0])
                
            elif list(onehot[i])==[0,0,0,0,0,0,0,1,0,0]:#7
                t_depend_encode[i]=np.array([0,0,0,0,0,1])
                
            elif list(onehot[i])==[0,0,0,0,0,0,0,0,1,0]:#8
                t_depend_encode[i]=np.array([0,0,1,0,1,1])
                
            elif list(onehot[i])==[0,0,0,0,0,0,0,0,0,1]:#9
                t_depend_encode[i]=np.array([0,0,0,0,1,1])
        return t_depend_encode
def measure_distance(prediction,label):
    distance = np.sum(np.abs(prediction - label))
    
    return distance

def shortest_distance(prediction_array): 
    distance_arr=np.zeros((len(prediction_array),10))
    final_class=list()
    for i in range(len(prediction_array)):
        distance_arr[i,0]=measure_distance(prediction_array[i],np.array([1,0,1,1,0,0]))
        distance_arr[i,1]=measure_distance(prediction_array[i],np.array([0,1,1,0,1,0]))
        distance_arr[i,2]=measure_distance(prediction_array[i],np.array([0,1,1,1,0,0]))
        distance_arr[i,3]=measure_distance(prediction_array[i],np.array([0,1,1,0,0,0]))
        distance_arr[i,4]=measure_distance(prediction_array[i],np.array([1,1,1,0,0,0]))
        distance_arr[i,5]=measure_distance(prediction_array[i],np.array([0,0,0,0,1,0]))
        distance_arr[i,6]=measure_distance(prediction_array[i],np.array([1,1,1,1,0,0]))
        distance_arr[i,7]=measure_distance(prediction_array[i],np.array([0,0,0,0,0,1]))
        distance_arr[i,8]=measure_distance(prediction_array[i],np.array([0,0,1,0,1,1]))
        distance_arr[i,9]=measure_distance(prediction_array[i],np.array([0,0,0,0,1,1]))
        
    for i in range(len(distance_arr)):       
        final_class.append(np.argmin(distance_arr[i]))
        
    return np.array(final_class)

if __name__ == "__main__":
    epoch=500
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

    plt.plot(history.history['loss'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.legend(['loss'], loc='upper left')
    plt.show()
    
    y_test = model.predict(x_test_norm)
#    arg_ytest = np.argmax(y_test, axis=1) # we can use this code for one-hot encoding
    arg_ytest = shortest_distance(y_test) # distance measure and return shortest argument for dependancy result
    acc = np.sum(arg_ytest==test_t)/len(arg_ytest) #count correct prediction
    print("accuracy is {}".format(acc))

#    below code is for checking dependancy when we train with one-hot encoding
#    second and third probable results are shown. result are in one-hot-testresult.txt
    
#    sorted_y_test = np.argsort(y_test,axis=1)
#    second_0=list()
#    third_0=list()
#    
#    second_1=list()
#    third_1=list()
#    
#    second_2=list()
#    third_2=list()
#    
#    second_3=list()
#    third_3=list()
#    
#    second_4=list()
#    third_4=list()
#    
#    second_5=list()
#    third_5=list()
#    
#    second_6=list()
#    third_6=list()
#    
#    second_7=list()
#    third_7=list()
#    
#    second_8=list()
#    third_8=list()
#    
#    second_9=list()
#    third_9=list()
#    
#    
#    for i in range(len(sorted_y_test)):    
#        if sorted_y_test[i,9]==0:
#            second_0.append(sorted_y_test[i,8])
#            third_0.append(sorted_y_test[i,7])
#            
#        elif sorted_y_test[i,9]==1:
#            second_1.append(sorted_y_test[i,8])
#            third_1.append(sorted_y_test[i,7])
#            
#        elif sorted_y_test[i,9]==2:
#            second_2.append(sorted_y_test[i,8])
#            third_2.append(sorted_y_test[i,7])
#            
#        elif sorted_y_test[i,9]==3:
#            second_3.append(sorted_y_test[i,8])
#            third_3.append(sorted_y_test[i,7])
#            
#        elif sorted_y_test[i,9]==4:
#            second_4.append(sorted_y_test[i,8])
#            third_4.append(sorted_y_test[i,7])
#            
#        elif sorted_y_test[i,9]==5:
#            second_5.append(sorted_y_test[i,8])
#            third_5.append(sorted_y_test[i,7])
#            
#        elif sorted_y_test[i,9]==6:
#            second_6.append(sorted_y_test[i,8])
#            third_6.append(sorted_y_test[i,7])
#            
#        elif sorted_y_test[i,9]==7:
#            second_7.append(sorted_y_test[i,8])
#            third_7.append(sorted_y_test[i,7])
#        
#        elif sorted_y_test[i,9]==8:
#            second_8.append(sorted_y_test[i,8])
#            third_8.append(sorted_y_test[i,7])
#            
#        elif sorted_y_test[i,9]==9:
#            second_9.append(sorted_y_test[i,8])
#            third_9.append(sorted_y_test[i,7])
#        
#        
#        
#      
#    check_freq(second_0)
#    check_freq(third_0)  
#    
#    check_freq(second_1)
#    check_freq(third_1)
#    
#    check_freq(second_2)
#    check_freq(third_2)
#    
#    check_freq(second_3)
#    check_freq(third_3)
#    
#    check_freq(second_4)
#    check_freq(third_4)
#    
#    check_freq(second_5)
#    check_freq(third_5)
#    
#    check_freq(second_6)
#    check_freq(third_6)
#    
#    check_freq(second_7)
#    check_freq(third_7)
#    
#    check_freq(second_8)
#    check_freq(third_8)
#    
#    check_freq(second_9)
#    check_freq(third_9)
        
        
        
        
        
        
