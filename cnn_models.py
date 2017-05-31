__author__ = 'JCajandi'

from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Flatten
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Average
from keras.layers import SimpleRNN
from keras.optimizers import Adam, SGD

def model_m3(n_filter, input_filter_length, input_stride, hidden_filter_length, hidden_stride):
    model = Sequential()
    #first layer
    model.add(Conv1D(filters=n_filter,
                     kernel_size=input_filter_length,
                     strides=input_stride,
                     input_shape=(20000,1)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    # model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=4, strides=None))

    #second layer
    model.add(Conv1D(filters= n_filter*2,
                     kernel_size= hidden_filter_length,
                     strides= hidden_stride))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    # model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=4, strides=None))

    #Flattening
    model.add(GlobalAveragePooling1D())

    #third layer is FC instead of general average pooling
    model.add(Dense(1))
    # model.add(Average())
    model.add(Activation("sigmoid"))
    return model

#M5 from paper
def model_m5(n_filter, input_filter_length, input_stride, hidden_filter_length, hidden_stride):
    model = Sequential()
    #first layer
    model.add(Conv1D(filters=n_filter,
                     kernel_size=input_filter_length,
                     strides=input_stride,
                     input_shape=(20000,1)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    # model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=4, strides=None))

    #second layer
    model.add(Conv1D(filters= n_filter*2,
                     kernel_size= hidden_filter_length,
                     strides= hidden_stride))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    # model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=4, strides=None))

    #third layer
    model.add(Conv1D(filters=n_filter*4,
                     kernel_size=hidden_filter_length,
                     strides=hidden_stride))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    # model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=4, strides=None))

    #fourth layer
    model.add(Conv1D(filters=n_filter*8,
                     kernel_size=hidden_filter_length,
                     strides=hidden_stride))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    # model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=4, strides=None))

    #Flattening
    model.add(GlobalAveragePooling1D())

    #fifth layer is FC instead of general average pooling
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    return model

def model_m7(n_filter, input_filter_length, input_stride, hidden_filter_length, hidden_stride):
    model = Sequential()

    #first layer
    model.add(Conv1D(filters=n_filter,
                     kernel_size=input_filter_length,
                     strides=input_stride,
                     input_shape=(20000,1)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    # model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=4, strides=None))

    #2nd and 3rd layer
    for i in range(2):
        model.add(Conv1D(filters= n_filter*2,
                         kernel_size= hidden_filter_length,
                         strides= hidden_stride))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        # model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=4, strides=None))

    #4th and 5th layer
    for i in range(2):
        model.add(Conv1D(filters= n_filter*4,
                         kernel_size= hidden_filter_length,
                         strides= hidden_stride))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        # model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=4, strides=None))

    #6th layer
    model.add(Conv1D(filters= n_filter*8,
                     kernel_size= hidden_filter_length,
                     strides= hidden_stride))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    # model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=4, strides=None))

    #7th layer is FC instead of general average pooling
    model.add(GlobalAveragePooling1D())
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    return model

def model_m9(n_filter, input_filter_length, input_stride, hidden_filter_length, hidden_stride):
    model = Sequential()

    #first layer
    model.add(Conv1D(filters=n_filter,
                     kernel_size=input_filter_length,
                     strides=input_stride,
                     input_shape=(20000,1)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    # model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=4, strides=None))

    #2nd and 3rd layer
    for i in range(2):
        model.add(Conv1D(filters= n_filter*2,
                         kernel_size= hidden_filter_length,
                         strides= hidden_stride))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        # model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=4, strides=None))

    #4th and 5th layer
    for i in range(2):
        model.add(Conv1D(filters= n_filter*4,
                         kernel_size= hidden_filter_length,
                         strides= hidden_stride))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        # model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=4, strides=None))

    #6th to 8th layer
    for i in range(3):
        model.add(Conv1D(filters= n_filter*8,
                         kernel_size= hidden_filter_length,
                         strides= hidden_stride))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        # model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=4, strides=None))

    #9th layer is FC instead of general average pooling
    model.add(GlobalAveragePooling1D())
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    return model