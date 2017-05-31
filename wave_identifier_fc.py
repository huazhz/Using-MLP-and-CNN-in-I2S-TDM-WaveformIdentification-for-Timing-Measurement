__author__ = 'jcajandi_0809'

import numpy as np
import pickle
import time

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

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

samples = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
end = 10000
offset = 0
accuracy = []
false_positive = []
false_negative = []

for sample in samples:
    train_data_sampled = train_data[:,offset::sample].copy()
    test_data_sampled = test_data[:,offset::sample].copy()

    n_train, n_input = np.shape(train_data_sampled)
    n_hidden = int(n_input * 1)
    learning_rate = 0.0001
    dropout = 0.8
    batch_size = 32

    print('Sample Size:', n_input, n_train)

    print('Building model...')

    #model is a 2 layer ANN
    model = Sequential()
    model.add(Dense(n_hidden, input_dim= n_input))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(n_hidden))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    #optimizer
    sgd = SGD(lr=learning_rate)

    #compile model
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    print('Model built.', "Elapsed: " , elapsed(time.time() - start_time))

    start_time = time.time()
    print('Training...')
    model.fit(train_data_sampled, train_label, nb_epoch=100, batch_size=batch_size, shuffle=True)
    print('Training finished.', "Elapsed: " , elapsed(time.time() - start_time))

    model_prediction = model.predict(test_data_sampled)
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
        # Accuracy: 86.0%
    accuracy.append((correct/250)*100)
    false_negative.append((false_neg/250)*100)
    false_positive.append((false_pos/250)*100)
        # print("\nTest accuracy: %.1f%%" % score[1])

print(accuracy)
print(false_negative)
print(false_positive)
with open(r'data\samples_false_pos_full.pkl','wb') as fp:
    pickle.dump([accuracy,false_negative,false_positive],fp,-1)
    fp.close()