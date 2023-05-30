import pandas as pd
import numpy as np

import os

import tensorflow.keras as tk
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, BatchNormalization, Dropout

import warnings
warnings.simplefilter('ignore')

class Models:
    '''
    
    fit and save as .h5 five convolutional NN models, one for every dataset.
    
    '''
    def __init__(self):
        '''
        
        define the folder path that contain all the tensors.
        
        '''
        self.path = 'tensors/'

    def arrays(self):
        '''
        
        load X, y of every dataset.
        
        '''
        for file in os.listdir(self.path):
            if file.startswith('en'):
                if file[3] == 'X':
                    self.en_X = np.load(self.path + file)
                if file[3] == 'y':
                    self.en_y = np.load(self.path + file)

            if file.startswith('fr'):
                if file[3] == 'X':
                    self.fr_X = np.load(self.path + file)
                if file[3] == 'y':
                    self.fr_y = np.load(self.path + file)

            if file.startswith('ge'):
                if file[3] == 'X':
                    self.ge_X = np.load(self.path + file)
                if file[3] == 'y':
                    self.ge_y = np.load(self.path + file)

            if file.startswith('it'):
                if file[3] == 'X':
                    self.it_X = np.load(self.path + file)
                if file[3] == 'y':
                    self.it_y = np.load(self.path + file)

            if file.startswith('lan'):
                if file[4] == 'X':
                    self.lan_X = np.load(self.path + file)
                if file[4] == 'y':
                    self.lan_y = np.load(self.path + file)

    def lan_arrays(self):
        '''
        
        function to return X, y of language dataset.
        
        '''
        self.arrays()

        return self.lan_X, self.lan_y

    def en_arrays(self):
        '''
        
        function to return X, y of english dataset.
        
        '''
        self.arrays()

        return self.en_X, self.en_y

    def ge_arrays(self):
        '''
        
        function to return X, y of german dataset.
        
        '''
        self.arrays()

        return self.ge_X, self.ge_y

    def it_arrays(self):
        '''
        
        function to return X, y of italian dataset.
        
        '''
        self.arrays()

        return self.it_X, self.it_y

    def fr_arrays(self):
        '''
        
        function to return X, y of french dataset.
        
        '''
        self.arrays()

        return self.fr_X.shape, self.fr_y
    
    def structure(self, units):
        '''
        
        define convolutional NN structure, 
        units = number classes for classification.
        
        '''
        K.clear_session()

        self.model = Sequential([

            Conv2D(filters = 64, 
                   kernel_size = (3, 3), 
                   strides = (3, 3), 
                   padding = 'valid', 
                   activation = tk.activations.relu, 
                   input_shape = (376, 376, 1), 
                   kernel_initializer = tk.initializers.GlorotNormal(seed = 34)), 

            BatchNormalization(), 

            MaxPooling2D(pool_size = (3, 3), 
                         strides = (3, 3), 
                         padding = 'valid'),

            Dropout(0.4), 

            Conv2D(filters = 128, 
                   kernel_size = (3, 3), 
                   strides = (3, 3), 
                   padding = 'valid', 
                   activation = tk.activations.relu), 

            BatchNormalization(), 

            MaxPooling2D(pool_size = (3, 3), 
                         strides = (3, 3), 
                         padding = 'valid'),

            Dropout(0.4),  

            Flatten(), 

            Dense(units = 100, 
                  activation = tk.activations.relu, 
                  kernel_regularizer=regularizers.l2(0.001)), 

            BatchNormalization(), 

            Dense(units = 50, 
                  activation = tk.activations.relu, 
                  kernel_regularizer=regularizers.l2(0.001)), 

            BatchNormalization(), 

            Dense(units = 20, 
                  activation = tk.activations.relu, 
                  kernel_regularizer=regularizers.l2(0.001)), 

            BatchNormalization(), 

            Dense(units = units, 
                  activation = tk.activations.softmax)

        ])

        self.summary = self.model.summary()
        self.model.compile(optimizer = 'adam', 
                           loss = tk.losses.categorical_crossentropy, 
                           metrics = ['categorical_accuracy'])
          
    def lan_fit_save(self):
        '''
        
        fit the model on language datas and save it as .h5.
        
        '''
        for file in os.listdir(self.path):
            if file.startswith('lan'):
                if file[4] == 'X':
                    self.lan_X = np.load(self.path + file)
                if file[4] == 'y':
                    self.lan_y = np.load(self.path + file)

        self.lan_classes = list(np.unique(self.lan_y))
        self.structure(len(self.lan_classes))
        self.ser_lan_y = pd.Series(self.lan_y).map({self.lan_classes[0]: 0, self.lan_classes[1]: 1, self.lan_classes[2]: 2, self.lan_classes[3]: 3})
        self.cat_lan_y = to_categorical(self.ser_lan_y)

        self.lan_model = self.model.fit(self.lan_X, 
                                        self.cat_lan_y, 
                                        epochs = 20, 
                                        batch_size = 64, 
                                        validation_split = 0.2)
        
        # self.model.save('models/lan_model.h5')
       
    def en_fit_save(self):
        '''
        
        fit the model on english datas and save it as .h5.
        
        '''
        for file in os.listdir(self.path):
            if file.startswith('en'):
                if file[3] == 'X':
                    self.en_X = np.load(self.path + file)
                if file[3] == 'y':
                    self.en_y = np.load(self.path + file)

        self.en_classes = list(np.unique(self.en_y))
        self.structure(len(self.en_classes))
        self.ser_en_y = pd.Series(self.en_y).map({self.en_classes[0]: 0, self.en_classes[1]: 1, self.en_classes[2]: 2, self.en_classes[3]: 3, self.en_classes[4]: 4,self.en_classes[5]: 5})
        self.cat_en_y = to_categorical(self.ser_en_y)

        self.en_model = self.model.fit(self.en_X, 
                                        self.cat_en_y, 
                                        epochs = 20, 
                                        batch_size = 64, 
                                        validation_split = 0.2)
        
        # self.model.save('models/en_model.h5')
        
    def ge_fit_save(self):
        '''
        
        fit the model on german datas and save it as .h5.
        
        '''
        for file in os.listdir(self.path):
            if file.startswith('ge'):
                if file[3] == 'X':
                    self.ge_X = np.load(self.path + file)
                if file[3] == 'y':
                    self.ge_y = np.load(self.path + file)

        self.ge_classes = list(np.unique(self.ge_y))
        self.structure(len(self.ge_classes))
        self.ser_ge_y = pd.Series(self.ge_y).map({self.ge_classes[0]: 0, self.ge_classes[1]: 1, self.ge_classes[2]: 2, self.ge_classes[3]: 3, self.ge_classes[4]: 4, self.ge_classes[5]: 5, self.ge_classes[6]: 6})
        self.cat_ge_y = to_categorical(self.ser_ge_y)

        self.ge_model = self.model.fit(self.ge_X, 
                                        self.cat_ge_y, 
                                        epochs = 20, 
                                        batch_size = 64, 
                                        validation_split = 0.2)
        
        # self.model.save('models/ge_model.h5')
        
    def it_fit_save(self):
        '''
        
        fit the model on italian datas and save it as .h5.
        
        '''
        for file in os.listdir(self.path):
            if file.startswith('it'):
                if file[3] == 'X':
                    self.it_X = np.load(self.path + file)
                if file[3] == 'y':
                    self.it_y = np.load(self.path + file)

        self.it_classes = list(np.unique(self.it_y))
        self.structure(len(self.it_classes))
        self.ser_it_y = pd.Series(self.it_y).map({self.it_classes[0]: 0, self.it_classes[1]: 1, self.it_classes[2]: 2, self.it_classes[3]: 3, self.it_classes[4]: 4, self.it_classes[5]: 5, self.it_classes[6]: 6})
        self.cat_it_y = to_categorical(self.ser_it_y)

        self.it_model = self.model.fit(self.it_X, 
                                        self.cat_it_y, 
                                        epochs = 20, 
                                        batch_size = 64, 
                                        validation_split = 0.2)
        
        # self.model.save('models/it_model.h5')

        
    def fr_fit_save(self):
        '''
        
        fit the model on french datas and save it as .h5.
        
        '''
        for file in os.listdir(self.path):
            if file.startswith('fr'):
                if file[3] == 'X':
                    self.fr_X = np.load(self.path + file)
                if file[3] == 'y':
                    self.fr_y = np.load(self.path + file)

        self.fr_classes = list(np.unique(self.fr_y))
        self.structure(len(self.fr_classes))
        self.ser_fr_y = pd.Series(self.fr_y).map({self.fr_classes[0]: 0, self.fr_classes[1]: 1, self.fr_classes[2]: 2, self.fr_classes[3]: 3})
        self.cat_fr_y = to_categorical(self.ser_fr_y)

        self.fr_model = self.model.fit(self.fr_X, 
                                        self.cat_fr_y, 
                                        epochs = 10, 
                                        batch_size = 64, 
                                        validation_split = 0.2)
        
        # self.model.save('models/fr_model.h5')

# Models().lan_fit_save()
# Models().en_fit_save()
# Models().ge_fit_save()
# Models().it_fit_save()
# Models().fr_fit_save()