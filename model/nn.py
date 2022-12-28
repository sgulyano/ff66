import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.layers import Activation, Add, AveragePooling1D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform

def ANN(input_shape, num_class):
    # input layer
    inp = Input(shape=input_shape)
    # Model
    x = Dense(200, activation='tanh')(inp) 
    x = Dense(200, activation='tanh')(x) 
    output = Dense(num_class, activation='sigmoid')(x)
    return Model(inputs=inp, outputs=output)


def AlexNet(input_shape, num_class):
    # input layer
    inp = Input(shape=(input_shape[0], 1))

    #Model
    # 1st layer (conv + pool + batchnorm)
    conv1 = Conv1D(16, kernel_size=7, activation='relu', padding='valid')(inp)
    batch1 = BatchNormalization()(conv1)  
    pool1 = MaxPooling1D(pool_size=2, strides=2, padding='valid')(batch1)
    # 2nd layer (conv + pool + batchnorm)
    conv2 = Conv1D(32, kernel_size=5, activation='relu', padding='valid')(pool1)
    batch2 = BatchNormalization()(conv2)   
    pool2 = MaxPooling1D(pool_size=2, strides=2, padding='valid')(batch2)
    # layer 3 (conv + batchnorm)      <--- note that the authors did not add a POOL layer here
    conv3 = Conv1D(64, kernel_size=3, activation='relu', padding='valid')(pool2)
    batch3 = BatchNormalization()(conv3)
    pool3 = MaxPooling1D(pool_size=2, strides=2, padding='valid')(batch3)
    # layer 4 (conv + batchnorm)      <--- similar to layer 3
    conv4 = Conv1D(128, kernel_size=3, activation='relu', padding='valid')(pool3)
    batch4 = BatchNormalization()(conv4)
    # pool4 = MaxPooling1D(pool_size=2, strides=2, padding='valid')(batch4)
    # layer 5 (conv + batchnorm)  
    conv5 = Conv1D(128, kernel_size=3, activation='relu', padding='valid')(batch4)
    batch5 = BatchNormalization()(conv5) 
    pool5 = MaxPooling1D(pool_size=2, strides=2, padding='valid')(batch5)
    # Flatten the CNN output to feed it with fully connected layers
    flat = Flatten()(pool5) 
    dense1 = Dense(200, activation='relu')(flat)
    drop1 = Dropout(0.5)(dense1)
    dense2 = Dense(200, activation='relu')(drop1)  
    drop2 = Dropout(0.5)(dense2)
    output = Dense(num_class, activation='sigmoid')(drop2)
    return Model(inputs=inp, outputs=output)


def VGG16(input_shape, num_class):
    # input layer
    visible1 = Input(shape=(input_shape[0], 1))

    #Model
    # Block 1
    x = Conv1D(64, 3, activation='relu', padding='same')(visible1)
    x = BatchNormalization()(x)
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2, strides=2)(x)
    # Block 2
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)  
    x = MaxPooling1D(2, strides=2)(x)

    # Block 3
    x = Conv1D(256, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)  
    x = Conv1D(256, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)  
    x = Conv1D(256, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)  
    x = MaxPooling1D(2, strides=2)(x)

    # Block 4
    x = Conv1D(512, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)  
    x = Conv1D(512, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)  
    x = Conv1D(512, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)  
    x = MaxPooling1D(2, strides=2)(x)

    # Block 5
    x = Conv1D(512, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)  
    x = Conv1D(512, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)  
    x = Conv1D(512, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)  
    x = MaxPooling1D(2, strides=2)(x)

    #x = GlobalAveragePooling1D()(x)
    #x = GlobalMaxPooling1D()(x)
    #------VGG16 Feature Extraction -------

    flat = Flatten()(x)  
    dense1 = Dense(200, activation='relu')(flat)  
    dense2 = Dense(200, activation='relu')(dense1)  
    output1 = Dense(num_class, activation='sigmoid')(dense2)
    return Model(inputs=visible1, outputs=output1)



def bottleneck_residual_block(X, f, filters, stage, block, reduce=False, s=2):
    """    
    Arguments:
    X -- input tensor of shape (m, height, width, channels)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    reduce -- boolean, True = identifies the reduction layer at the beginning of each learning stage
    s -- integer, strides
    
    Returns:
    X -- output of the identity block, tensor of shape (H, W, C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    if reduce:
        # if we are to reduce the spatial size, apply a 1x1 CONV layer to the shortcut path
        # to do that, we need both CONV layers to have similar strides 
        X = Conv1D(filters = F1, kernel_size = 1, strides = s, padding = 'valid', name = conv_name_base + '2a', 
                   kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(name = bn_name_base + '2a')(X)
        X = Activation('relu')(X)
        
        X_shortcut = Conv1D(filters = F3, kernel_size = 1, strides = s, padding = 'valid', name = conv_name_base + '1',
                        kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(name = bn_name_base + '1')(X_shortcut) 
    else: 
        # First component of main path
        X = Conv1D(filters = F1, kernel_size = 1, strides = 1, padding = 'valid', name = conv_name_base + '2a', 
                   kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(name = bn_name_base + '2a')(X)
        X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv1D(filters = F2, kernel_size = f, strides = 1, padding = 'same', name = conv_name_base + '2b', 
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv1D(filters = F3, kernel_size = 1, strides = 1, padding = 'valid', name = conv_name_base + '2c', 
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation 
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X
    

def ResNet50(input_shape, num_class):
    # input layer
    visible1 = Input(shape=(input_shape[0], 1))

    #Model
    # Stage 1
    X = Conv1D(64, 7, strides=2, name='conv1', kernel_initializer=glorot_uniform(seed=0))(visible1)
    X = BatchNormalization(name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling1D(3, strides=2)(X)

    # Stage 2
    X = bottleneck_residual_block(X, 3, [64, 64, 256], stage=2, block='a', reduce=True, s=1)
    X = bottleneck_residual_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = bottleneck_residual_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3 
    X = bottleneck_residual_block(X, 3, [128, 128, 512], stage=3, block='a', reduce=True, s=2)
    X = bottleneck_residual_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = bottleneck_residual_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = bottleneck_residual_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4 
    X = bottleneck_residual_block(X, 3, [256, 256, 1024], stage=4, block='a', reduce=True, s=2)
    X = bottleneck_residual_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = bottleneck_residual_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = bottleneck_residual_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = bottleneck_residual_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = bottleneck_residual_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 
    X = bottleneck_residual_block(X, 3, [512, 512, 2048], stage=5, block='a', reduce=True, s=2)
    X = bottleneck_residual_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = bottleneck_residual_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL 
    X = AveragePooling1D(1, name="avg_pool")(X)

    # output layer
    flat = Flatten()(X)
    dense1 = Dense(200, activation='relu')(flat)  
    dense2 = Dense(200, activation='relu')(dense1)  
    output1 = Dense(num_class, activation='softmax')(dense2)
    return Model(inputs=[visible1], outputs=output1)


l = 12 #L
num_filter = 36 #k
compression = 0.5
dropout_rate = 0.2

def add_denseblock(input, num_filter = 12, dropout_rate = 0.2):
    global compression
    temp = input
    for _ in range(l):
        BatchNorm = BatchNormalization()(temp)
        relu = Activation('relu')(BatchNorm)
        Conv2D_3_3 = Conv1D(int(num_filter*compression), 3, use_bias=False ,padding='same')(relu)
        if dropout_rate>0:
            Conv2D_3_3 = Dropout(dropout_rate)(Conv2D_3_3)
        concat = Concatenate(axis=-1)([temp,Conv2D_3_3])
        temp = concat
    return temp
   
def add_transition(input, num_filter = 12, dropout_rate = 0.2):
    global compression
    BatchNorm = BatchNormalization()(input)
    relu = Activation('relu')(BatchNorm)
    Conv2D_BottleNeck = Conv1D(int(num_filter*compression), 1, use_bias=False ,padding='same')(relu)
    if dropout_rate>0:
        Conv2D_BottleNeck = Dropout(dropout_rate)(Conv2D_BottleNeck)
    avg = AveragePooling1D(pool_size=2)(Conv2D_BottleNeck)
    return avg

def output_layer(input, num_class):
    global compression
    BatchNorm = BatchNormalization()(input)
    relu = Activation('relu')(BatchNorm)
    AvgPooling = AveragePooling1D(pool_size=2)(relu)
    flat = Flatten()(AvgPooling)
    dense1 = Dense(200, activation='relu', name='linear')(flat)
    dense2 = Dense(200, activation='relu')(dense1)  
    output = Dense(num_class, activation='softmax')(dense2)
    return output

def DenseNet(input_shape, num_class):
    # input layer
    visible1 = Input(shape=(input_shape[0], 1))

    #Model
    First_Conv2D = Conv1D(num_filter, 3, use_bias=False ,padding='same')(visible1)
    # Back_Prop_First_Conv2D
    First_Block = add_denseblock(First_Conv2D, num_filter, dropout_rate)
    First_Transition = add_transition(First_Block, num_filter, dropout_rate)

    Second_Block = add_denseblock(First_Transition, num_filter, dropout_rate)
    Second_Transition = add_transition(Second_Block, num_filter, dropout_rate)

    Third_Block = add_denseblock(Second_Transition, num_filter, dropout_rate)
    Third_Transition = add_transition(Third_Block, num_filter, dropout_rate)

    Last_Block = add_denseblock(Third_Transition,  num_filter, dropout_rate)
    output1 = output_layer(Last_Block, num_class)  

    return Model(inputs=[visible1], outputs=output1)
    
    
# def recall(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall_keras = true_positives / (possible_positives + K.epsilon())
#     return recall_keras