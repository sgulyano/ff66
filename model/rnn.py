import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model


def VanLSTM(input_shape, num_class):
    # input layer
    visible1 = Input(shape=(input_shape[1], input_shape[2]))
    lstm1 = LSTM(200)(visible1)  
    dense1 = Dense(200, activation='relu')(lstm1)  
    dense2 = Dense(200, activation='relu')(dense1)  
    output1 = Dense(num_class, activation='sigmoid')(dense2)
    return Model(inputs=[visible1], outputs=output1)