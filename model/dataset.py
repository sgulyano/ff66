import re
import os
import glob
import numpy as np
import pandas as pd

import tensorflow as tf

from scipy import interpolate

import matplotlib.pyplot as plt

class SpectraDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
        # read ref shift
        self.df_ref = pd.read_csv(os.path.join(data_dir, 'ref.csv'))
        self.df_reduce_ref = self._reduce_res(self.df_ref.values)
        
        # read training data
        self.train_data, self.target, self.class_name = self._read_data('train')
        print(f'Read {self.train_data.shape[0]} spectra for training.')
        print('No. of data points/shifts per spectra (input shape) :', self.get_input_shape())
        print('No. of classes :', self.get_num_class())
        
        # read test data
        self.X_test, y_target = self._read_data('test')
        print(f'Read {self.X_test.shape[0]} spectra for testing.')
        self.y_test = [(np.array(y_target) == label).astype('int8') for label in self.class_name]
        self.y_test = np.vstack(self.y_test).T
        
#         fl = glob.glob(os.path.join(data_dir, 'train', '*.csv'))
#         self.target = [re.search(r'/SD_(.*)_', f).group(1) for f in fl]
        
#         u, c = np.unique(self.target, return_counts=True)
#         self.class_name = u
#         self.num_class = len(u)
#         print(f'No. of materials : {len(u)}')
#         print(np.vstack([u, c]).T)

#         # preprocess training data
#         self.df_reduce_ref = self._reduce_res(self.df_ref.values)
#         self.train_data = pd.DataFrame(columns=self.df_reduce_ref,
#                                        index=range(len(fl)))
#         for i, f in enumerate(fl):
#             df = pd.read_csv(f, header=None, names=['shift', 'intensity'])
# #             print(df)
#             signal = self._align_data(df)
# #             print(signal)
# #             plt.figure(figsize=(20,5))
# #             plt.plot(self.df_ref, signal)
#             signal = self._reduce_res(signal)
# #             print(signal)
# #             plt.plot(self.df_reduce_ref, signal, '--')
# #             plt.show()
#             self.train_data.iloc[i] = self._normalize(signal)
            
#         print(self.train_data.head())
#         print(f'Read {len(fl)} spectra.')
        
        
    def _read_data(self, folder):
        # read data from folder, either train or test
        print(f'Reading {folder} data')
        fl = glob.glob(os.path.join(self.data_dir, folder, '*.csv'))
        target = [re.search(r'/SD_(.*)_', f).group(1) for f in fl]
        
        if folder == 'train':
            u, c = np.unique(target, return_counts=True)
            class_name = u
            num_class = len(u)
            print(f'No. of materials : {len(u)}')
            print(np.vstack([u, c]).T)

        # preprocess training data
        data = pd.DataFrame(columns=self.df_reduce_ref,
                            index=range(len(fl)))
        for i, f in enumerate(fl):
            df = pd.read_csv(f, header=None, names=['shift', 'intensity'])
            signal = self._align_data(df)
            signal = self._reduce_res(signal)
            data.iloc[i] = self._normalize(signal)
        
        data = data.astype(np.float32)
#         print(data.head())
        if folder == 'train':
            return data, target, class_name
        elif folder == 'test':
            return data, target
            
    
    def _align_data(self, df_signal):
        f = interpolate.interp1d(df_signal['shift'], 
                                 df_signal['intensity'], 
                                 kind='nearest')
        return f(self.df_ref['shift'])
    
    def _reduce_res(self, signal, block_size=8):
        """ Reduce spectrum resolution by taking average intensity 
            within the non-overlapping blocks, where block size can 
            be 2, 4, 8, or 16"""
        return signal.reshape(-1,block_size).mean(axis=1)
    
    def _normalize(self, signal):
        return (signal - signal.min()) / (signal.max() - signal.min())

    def get_training_data(self):
        return self.train_data.values
    
    def get_binary_target(self, label):
        return (np.array(self.target) == label).astype('int8')
    
    def get_binary_targets(self):
        return np.vstack([self.get_binary_target(label) for label in self.class_name]).T
    
    def get_shift_data(self):
        return self.df_reduce_ref
    
    def get_input_shape(self):
        return (len(self.df_reduce_ref),)
    
    def get_num_class(self):
        return len(self.class_name)
    
    def get_test_data(self):
        # # read training data
        # X_test, y_target = self._read_data('test')
        # print(f'Read {X_test.shape[0]} spectra for testing.')
        # y_test = [(np.array(y_target) == label).astype('int8') for label in self.class_name]
        # return X_test.values, np.vstack(y_test).T
        return self.X_test.values, self.y_test



class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for CNN/ANN models as 1D arrays'
    def __init__(self, dataset, batch_size=32, shuffle=True, sigma=1):
        'Initialization'
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_shape = dataset.train_data.shape
        self.sigma = sigma
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.data_shape[0] / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.data_shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = self.dataset.get_training_data()[indexes]
        y = self.dataset.get_binary_targets()[indexes]
        
        # add white gaussian noise
        noise_G = 0.01*np.random.rand()*np.random.normal(0, self.sigma, X.shape)
        X_noisy = X + noise_G
        
        return self.dataset._normalize(X_noisy), y



class DataGeneratorRNN(tf.keras.utils.Sequence):
    'Generates data for RNN models as 2D arrays (multiple chunks of 1D array)'
    def __init__(self, dataset, batch_size=32, shuffle=True, n_features = 24, sigma=1):
        'Initialization'
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.X = split2feat(dataset.train_data, n_features)
        # self.X = np.zeros((dataset.train_data.shape[0], int(np.ceil(dataset.train_data.shape[1]/n_features)*n_features)), dtype=np.int32)
        # # print(zeros.shape)
        # self.X[:,:dataset.train_data.shape[1]] = self.dataset.get_training_data()
        # self.X = self.X.reshape((self.X.shape[0], int(self.X.shape[1]/n_features), n_features))
        self.data_shape = self.X.shape
        
        self.sigma = sigma
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.data_shape[0] / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.data_shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = self.X[indexes]
        y = self.dataset.get_binary_targets()[indexes]
        
        # add white gaussian noise
        noise_G = 0.01*np.random.rand()*np.random.normal(0, self.sigma, X.shape)
        X_noisy = X + noise_G
        
        return self.dataset._normalize(X_noisy), y

    
def split2feat(x, n_features):
    x_new = np.zeros((x.shape[0], int(np.ceil(x.shape[1]/n_features)*n_features)), dtype=np.int32)
    x_new[:,:x.shape[1]] = x
    x_new = x_new.reshape((x_new.shape[0], int(x_new.shape[1]/n_features), n_features))
    return x_new
