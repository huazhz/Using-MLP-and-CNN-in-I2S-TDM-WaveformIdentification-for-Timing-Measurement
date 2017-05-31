__author__ = 'jcajandi_0809'

import numpy as np
import pickle
import time

from keras.optimizers import Adam, SGD
from cnn_models import *

testfile = r'data\test.pkl'
trainfile = r'data\train.pkl'

start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"

#load data
with open(testfile, 'rb') as fp:
    test_label, test_data = pickle.load(fp)
    test_data = test_data.astype('float32')
    fp.close()

with open(trainfile, 'rb') as fp:
    train_label, train_data = pickle.load(fp)
    fp.close()

n_train, n_input = np.shape(train_data)
n_test, _ = np.shape(test_data)

#need to reshape training and input data to a 3d tensor with shape (number of data, number of features, 1)
train_data = np.reshape(train_data,(n_train,n_input,1))
test_data = np.reshape(test_data,(n_test,n_input,1))


input_filter_length = 400 #corresponds to a 4ns of waveform, 1 division/5
input_stride = 10 #1ns stride
hidden_filter_length = 4
hidden_stride = 1
n_filter = 4
learning_rate = 0.0001
batch_size = 64

accuracy = []
false_positive = []
false_negative = []

models = ['m3', 'm5', 'm7', 'm9']

for model_version in models:
    if model_version == 'm3':
        model = model_m3(n_filter,input_filter_length,input_stride,hidden_filter_length,hidden_stride)
    elif model_version == 'm5':
        model = model_m5(n_filter,input_filter_length,input_stride,hidden_filter_length,hidden_stride)
    elif model_version == 'm7':
        model = model_m7(n_filter,input_filter_length,input_stride,hidden_filter_length,hidden_stride)
    elif model_version == 'm9':
        model = model_m9(n_filter,input_filter_length,input_stride,hidden_filter_length,hidden_stride)

    adam = Adam(lr=learning_rate)
    sgd = SGD(lr=learning_rate)

    model.compile(optimizer= sgd, loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())

    model.fit(train_data, train_label,
              epochs=100,
              batch_size=batch_size, shuffle=True)

    model_prediction = model.predict(test_data)
    model_prediction [model_prediction < 0.5] = 0.
    model_prediction [model_prediction != 0.] = 1.

    false_pos = 0
    false_neg = 0
    correct = 0

    for x in range(250):
        if test_label[x] == model_prediction[x]:
            correct += 1
        elif test_label[x] > model_prediction[x]:
            false_neg += 1
        elif test_label[x] < model_prediction[x]:
            false_pos += 1

    accuracy.append((correct/250)*100)
    false_negative.append((false_neg/250)*100)
    false_positive.append((false_pos/250)*100)


print(accuracy)
print(false_negative)
print(false_positive)

with open(r'data\samples_results_cnn_bn.pkl','wb') as fp:
    pickle.dump([accuracy,false_negative,false_positive],fp,-1)
    fp.close()
