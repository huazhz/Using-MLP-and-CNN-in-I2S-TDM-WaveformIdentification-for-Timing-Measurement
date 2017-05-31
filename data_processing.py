__author__ = 'jcajandi_0809'

import pandas as pd
import numpy as np
import glob
import pickle
import os

from sklearn.utils import shuffle
from sklearn import preprocessing

#get labels should be nominal, ff, ss
labels = []

data1 = pd.read_csv(r'data\raw\nominal_labels.csv', header= None)
data2 = pd.read_csv(r'data\raw\ff_labels.csv', header= None)
data3 = pd.read_csv(r'data\raw\ss_labels.csv', header= None)

print(data1.shape)
print(data2.shape)
print(data3.shape)

labels.append(data1)
labels.append(data2)
labels.append(data3)

data_labels = pd.concat(labels, ignore_index= True)
print(data_labels.shape)

#get data: should be nominal, ff, ss
folder_path = r'data\raw'
skews = [r'\nominal',r'\ss',r'\ss']
# skews = [r'\nominal']

waves = []
for skew in skews:
    filenames = glob.glob(folder_path + skew + '/*.txt')
    for filename in filenames:
        data = pd.read_csv(filename, names=['wave'])
        waves.append(data['wave'].values)
data_waves = np.array(waves)
print(data_waves.shape)

data_pd = pd.DataFrame(data_waves, index= None)
data_pd['labels'] = data_labels
data_pd.to_csv(r'data\data.csv', index= False)
print(data_pd['labels'].shape)

data_file = r'data\data.pkl'
with open(data_file,'wb') as fp:
    pickle.dump([data_pd],fp,-1)
    fp.close()

testfile = r'data\test.pkl'
trainfile = r'data\train.pkl'
datafile = r'data\data_processed.pkl'

# preprocess labels to convert strings to numbers
le = preprocessing.LabelEncoder()

if not os.path.isfile(datafile):
    # read data
    data_raw = pd.read_csv(r'data\data.csv')

    # shuffle data
    data_raw  = shuffle(data_raw)

    train_raw = pd.DataFrame()
    train_raw = data_raw[:100]
    test_raw = pd.DataFrame()
    test_raw = data_raw[100:]

    print(data_raw['labels'].shape, train_raw.shape, test_raw.shape)

    wave_label_raw = data_raw['labels'].values
    wave_data = data_raw.drop(['labels'] , axis=1).values
    test_label_raw = test_raw['labels'].values
    test_data = test_raw.drop(['labels'] , axis=1).values
    train_label_raw = train_raw['labels'].values
    train_data = train_raw.drop(['labels'] , axis=1).values

    le.fit(wave_label_raw)
    wave_label = le.transform(wave_label_raw)
    le.fit(test_label_raw)
    test_label = le.transform(test_label_raw)
    le.fit(train_label_raw)
    train_label = le.transform(train_label_raw)

    with open(datafile,'wb') as fp:
        pickle.dump([wave_label,wave_data],fp,-1)
        fp.close()

    with open(testfile,'wb') as fp:
        pickle.dump([test_label,test_data],fp,-1)
        fp.close()

    with open(trainfile,'wb') as fp:
        pickle.dump([train_label,train_data],fp,-1)
        fp.close()
