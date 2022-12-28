import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.layers import Activation, Add, AveragePooling1D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform


def VanLSTM(input_shape, num_class):
    # Input to models
    x1 = sx.reshape((sx.shape[0], int(sx.shape[1]/ n_features), n_features)) # inputs of CNN, LSTM

    #Train data
    #All data input
    x_trainDL1 = x1

    y_train = sy

    y_train_one_hot = to_categorical(y_train)


    # input layer
    visible1 = Input(shape=(x_trainDL1.shape[1], x_trainDL1.shape[2]))
    lstm1 = LSTM(200)(visible1)  
    dense1 = Dense(200, activation='relu')(lstm1)  
    dense2 = Dense(200, activation='relu')(dense1)  
    output1 = Dense(class_num, activation='softmax')(dense2)
    return Model(inputs=[visible1], outputs=output1)