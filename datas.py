import pandas as pd
import numpy as np

from numpy import expand_dims

import os
import random

import torchaudio
import torch
from torchaudio import transforms

import warnings
warnings.simplefilter('ignore')

from tensorflow.keras.preprocessing.image import ImageDataGenerator

class Datasets:
    '''
    
    class to create dataframes with file names and label for every file of every dataset.
    
    '''
    def __init__(self):
        '''
        
        folder paths of languages dataframe, and for dataframes of sentiments for every language.
        
        '''
        self.languages = 'data_languages/'
        self.eng = 'eng_speech/wav/'
        self.fr = 'fr_speech/wav/'
        self.ger = 'ger_speech/wav/'
        self.ita = 'ita_speech/wav/'

    def languages_datas(self):
        '''
        
        extract filename, folder and label of 10000 files for every folder in self.languages and append it in a dataframe, 
        column 'files' with folder/filename, 
        column 'classes' with label.

        return the dataframe.
        
        '''
        self.lang_df = pd.DataFrame(columns=['files', 'classes'])
        self.files = []
        self.classes = []

        for self.folder in os.listdir(self.languages):
            if not self.folder.endswith('Store'):
                self.count = 0
                for self.file in os.listdir(self.languages + self.folder):
                    while self.count < 10000:
                        self.files.append(f'{self.folder}/{self.file}')
                        self.classes.append(self.folder)
                        self.count += 1

        self.lang_df['files'] = self.files
        self.lang_df['classes'] = self.classes

        return self.lang_df

    def eng_datas(self):
        '''
        
        extract filename and label of every file in self.eng and append it in a dataframe, 
        column 'files' with filename, 
        column 'classes' with label.
        
        return the dataframe.
        
        '''
        self.eng_df = pd.DataFrame(columns = ['files', 'classes'])
        self.files = []
        self.classes = []

        for self.file in os.listdir(self.eng):
            if self.file.endswith('wav'):
                self.files.append(self.file)
                if self.file[9:-7] == 'ANG':
                    self.classes.append('anger')
                if self.file[9:-7] == 'NEU':
                    self.classes.append('neutral')
                if self.file[9:-7] == 'DIS':
                    self.classes.append('disgust')
                if self.file[9:-7] == 'SAD':
                    self.classes.append('sad')
                if self.file[9:-7] == 'FEA':
                    self.classes.append('fear')
                if self.file[9:-7] == 'HAP':
                    self.classes.append('happy')
                if self.file[9:-7] == 'SA':
                    self.classes.append('sad')

        self.eng_df['files'] = self.files
        self.eng_df['classes'] = self.classes

        return self.eng_df
    
    def fr_datas(self):
        '''
        
        extract filename and label of every file in self.fr and append it in a dataframe, 
        column 'files' with filename, 
        column 'classes' with label.
        
        return the dataframe.
        
        '''
        self.fr_df = pd.DataFrame(columns = ['files', 'classes'])
        self.files = []
        self.classes = []

        for self.file in os.listdir(self.fr):
            if self.file.endswith('wav'):
                self.files.append(self.file)
                if self.file[4:6] == 'a1':
                    self.classes.append('friendly')
                if self.file[4:6] == 'a2':
                    self.classes.append('seductive')
                if self.file[4:6] == 'a3':
                    self.classes.append('dominant')
                if self.file[4:6] == 'a4':
                    self.classes.append('distant')

        self.fr_df['files'] = self.files
        self.fr_df['classes'] = self.classes

        return self.fr_df

    def ger_datas(self):
        '''
        
        extract filename and label of every file in self.ger and append it in a dataframe, 
        column 'files' with filename, 
        column 'classes' with label.
        
        return the dataframe.
        
        '''
        self.ger_df = pd.DataFrame(columns = ['files', 'classes'])
        self.files = []
        self.classes = []

        for self.file in os.listdir(self.ger):
            if self.file.endswith('wav'):
                self.files.append(self.file)
                if self.file[5] == 'W':
                    self.classes.append('anger')
                if self.file[5] == 'L':
                    self.classes.append('boredom')
                if self.file[5] == 'E':
                    self.classes.append('disgust')
                if self.file[5] == 'A':
                    self.classes.append('fear')
                if self.file[5] == 'F':
                    self.classes.append('happy')
                if self.file[5] == 'T':
                    self.classes.append('sad')
                if self.file[5] == 'N':
                    self.classes.append('neutral')

        self.ger_df['files'] = self.files
        self.ger_df['classes'] = self.classes

        return self.ger_df

    def ita_datas(self):
        '''
        
        extract filename and label of every file in self.ita and append it in a dataframe, 
        column 'files' with filename, 
        column 'classes' with label.
        
        return the dataframe.
        
        '''
        self.ita_df = pd.DataFrame(columns = ['files', 'classes'])
        self.files = []
        self.classes = []

        for self.file in os.listdir(self.ita):
            if self.file.endswith('wav'):
                self.files.append(self.file)
                if self.file[:3] == 'dis':
                    self.classes.append('disgust')
                if self.file[:3] == 'gio':
                    self.classes.append('joy')
                if self.file[:3] == 'pau':
                    self.classes.append('fear')
                if self.file[:3] == 'rab':
                    self.classes.append('anger')
                if self.file[:3] == 'sor':
                    self.classes.append('surprise')
                if self.file[:3] == 'tri':
                    self.classes.append('sad')
                if self.file[:3] == 'neu':
                    self.classes.append('neutral')

        self.ita_df['files'] = self.files
        self.ita_df['classes'] = self.classes

        return self.ita_df
    
class AudioUtil:
    '''
    
    list of function to audio and images processing.
    
    '''
    @staticmethod
    def open(audiofile):
        '''
        
        load an audio file and return the signal as a tensor and his sample rate
        
        '''
        sig, sr = torchaudio.load(audiofile)

        return sig, sr
    
    @staticmethod
    def rechannel(aud, new_channel):
        '''
        
        convert audio into desired number of channel
        
        '''
        sig, sr = aud

        if (sig.shape[0] == new_channel):
            return aud
        
        if (new_channel == 1):
            resig = sig[:1, :]

        else:
            resig = torch.cat([sig, sig])

        return ((resig, sr))

    @staticmethod
    def resample(aud, newsr):
        '''
        
        resample all audios to a given samplerate
        
        '''
        sig, sr = aud
        if (sr == newsr):
            return aud

        num_channels = sig.shape[0]
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :])

        if (num_channels > 1):
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:, :])
            resig = torch.cat([resig, retwo])
        
        return ((resig, newsr))

    @staticmethod
    def pad_trunc(aud, max_ms):
        '''
        
        truncate audiofile to a fixed lenght in milliseconds.
        
        '''
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr // 1000 * max_ms

        if (sig_len > max_len):
            sig = sig[:,:max_len]

        elif (sig_len < max_len):
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len -pad_begin_len

            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)

        return (sig, sr)
    
    @staticmethod
    def time_shift(aud, shift_limit):
        '''
        
        shift the signal to left or right.
        
        '''
        sig, sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)

        return (sig.roll(shift_amt), sr)
    
    @staticmethod
    def spectro_gram(aud, n_mels = 64, n_fft = 1024, hop_len = None):
        '''
        
        generate a spectrogram.
        
        '''
        sig, sr = aud
        top_db = 80

        spec = transforms.MelSpectrogram(sr, n_fft = n_fft, hop_length = hop_len, n_mels = n_mels)(sig)
        spec = transforms.AmplitudeToDB(top_db = top_db)(spec)

        return (spec)
    
    @staticmethod
    def spectro_augment(spec, max_mask_pct = 0.1, n_freq_masks = 1, n_time_masks = 1):
        '''
        
        mask some sections of the spectrogram both in frequency and time dimension, 
        helps the model to avoid overfitting and generalised better.
        
        '''
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec
        freq_mask_param = max_mask_pct * n_mels

        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps

        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec
    
class SoundDS():
    '''
    
    transformations pipeline.
    
    '''
    def __init__(self, df, data_path):
        '''
        
        dataframe and folder path as input, 
        duration 6 seconds, 
        samplerate 16000, 
        mono, 
        0,4 shift.
        
        '''
        self.df = df
        self.data_path = str(data_path)
        self.duration = 6000
        self.sr = 16000
        self.channel = 1
        self.shift_pct = 0.4

    def __len__(self):
        '''
        
        return the lenght of the dataframe.
        
        '''
        return len(self.df)
    
    def __getitem__(self, idx):
        '''
        
        for every index import and trasform the corresponding file, 
        return tensor and label.
        
        '''
        audio_file = self.data_path + self.df.loc[idx, 'files']
        class_Id = self.df.loc[idx, 'classes']

        aud = AudioUtil.open(audio_file)
        re_aud = AudioUtil.resample(aud, self.sr)
        re_chan = AudioUtil.rechannel(re_aud, self.channel)
        dur_aud = AudioUtil.pad_trunc(re_chan, self.duration)
        shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
        sgram = AudioUtil.spectro_gram(shift_aud, n_mels = 376, n_fft = 512, hop_len = None)
        aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct = 0.1, n_freq_masks = 2, n_time_masks = 2)

        return aug_sgram, class_Id

class Arrays:
    '''
    
    apply SoundDS through all the datas and export them and relative labels.
    
    '''
    def __init__(self):
        '''
        
        initialize Datasets class as self.datasets.
        
        '''
        self.datasets = Datasets()

    def lan_X_y(self):
        '''
        
        apply SoundDS on languages dataset.
        
        '''
        lang_ds = SoundDS(self.datasets.languages_datas(), 
                self.datasets.languages)
        len_lang = lang_ds.__len__()

        self.lan_X = []
        self.lan_y = []

        for i in range(len_lang):
            X, y = lang_ds.__getitem__(i)
            X = X.reshape((376, 376, 1))
            X = np.array(X)

            self.lan_X.append(X)
            self.lan_y.append(y)


    def lan_save(self):
        '''
        
        export tensors of all the language dataset as X and relatives labels as y.
        
        '''
        self.lan_X_y()
        self.lan_X = np.array(self.lan_X)
        np.save('tensors/lan_X', self.lan_X)

        self.lan_y = np.array(self.lan_y)
        np.save('tensors/lan_y', self.lan_y)

    
    def en_X_y(self):
        '''
        
        apply SoundDS on english dataset.
        
        '''
        en_ds = SoundDS(self.datasets.eng_datas(), 
                self.datasets.eng)
        len_en = en_ds.__len__()

        self.en_X = []
        self.en_y = []

        for i in range(len_en):
            X, y = en_ds.__getitem__(i)
            X = X.reshape((376, 376, 1))
            X = np.array(X)

            self.en_X.append(X)
            self.en_y.append(y)
        

            self.samples = expand_dims(X, 0)
            self.datagen = ImageDataGenerator(rotation_range=10, 
                                              width_shift_range=0.1, 
                                              height_shift_range=0.1, 
                                              zoom_range=0.1, 
                                              fill_mode='wrap')
            self.iter = self.datagen.flow(self.samples, 
                                          batch_size = 1)
            for i in range(7):
                self.batch = self.iter.next()
                self.image = self.batch[0].astype('uint8')
                self.en_X.append(self.image)
                self.en_y.append(y)
    
    def en_save(self):
        '''
        
        export tensors of all the english dataset as X and relatives labels as y.
        
        '''
        self.en_X_y()
        self.en_X = np.array(self.en_X)
        np.save('tensors/en_X', self.en_X)

        self.en_y = np.array(self.en_y)
        np.save('tensors/en_y', self.en_y)

    def ge_X_y(self):
        '''
        
        apply SoundDS on german dataset.
        
        '''
        ge_ds = SoundDS(self.datasets.ger_datas(), 
                self.datasets.ger)
        len_ge = ge_ds.__len__()

        self.ge_X = []
        self.ge_y = []

        for i in range(len_ge):
            X, y = ge_ds.__getitem__(i)
            X = X.reshape((376, 376, 1))
            X = np.array(X)

            self.ge_X.append(X)
            self.ge_y.append(y)


            self.samples = expand_dims(X, 0)
            self.datagen = ImageDataGenerator(rotation_range=10, 
                                              width_shift_range=0.1, 
                                              height_shift_range=0.1, 
                                              zoom_range=0.1, 
                                              fill_mode='wrap')
            self.iter = self.datagen.flow(self.samples, 
                                          batch_size = 1)
            for i in range(40):
                self.batch = self.iter.next()
                self.image = self.batch[0].astype('uint8')
                self.ge_X.append(self.image)
                self.ge_y.append(y)

    def ge_save(self):
        '''
        
        export tensors of all the german dataset as X and relatives labels as y.
        
        '''
        self.ge_X_y()
        self.ge_X = np.array(self.ge_X)
        np.save('tensors/ge_X', self.ge_X)

        self.ge_y = np.array(self.ge_y)
        np.save('tensors/ge_y', self.ge_y)
    
    def fr_X_y(self):
        '''
        
        apply SoundDS on french dataset.
        
        '''
        fr_ds = SoundDS(self.datasets.fr_datas(), 
                self.datasets.fr)
        len_fr = fr_ds.__len__()

        self.fr_X = []
        self.fr_y = []

        for i in range(len_fr):
            X, y = fr_ds.__getitem__(i)
            X = X.reshape((376, 376, 1))
            X = np.array(X)

            self.fr_X.append(X)
            self.fr_y.append(y)

    def fr_save(self):
        '''
        
        export tensors of all the french dataset as X and relatives labels as y.
        
        '''
        self.fr_X_y()
        self.fr_X = np.array(self.fr_X)
        np.save('tensors/fr_X', self.fr_X)

        self.fr_y = np.array(self.fr_y)
        np.save('tensors/fr_y', self.fr_y)
    
    def it_X_y(self):
        '''
        
        apply SoundDS on italian dataset.
        
        '''
        it_ds = SoundDS(self.datasets.ita_datas(), 
                self.datasets.ita)
        len_it = it_ds.__len__()

        self.it_X = []
        self.it_y = []

        for i in range(len_it):
            X, y = it_ds.__getitem__(i)
            X = X.reshape((376, 376, 1))
            X = np.array(X)

            self.it_X.append(X)
            self.it_y.append(y)


            self.samples = expand_dims(X, 0)
            self.datagen = ImageDataGenerator(rotation_range=10, 
                                              width_shift_range=0.1, 
                                              height_shift_range=0.1, 
                                              zoom_range=0.1, 
                                              fill_mode='wrap')
            self.iter = self.datagen.flow(self.samples, 
                                          batch_size = 1)
            for i in range(40):
                self.batch = self.iter.next()
                self.image = self.batch[0].astype('uint8')
                self.it_X.append(self.image)
                self.it_y.append(y)

    def it_save(self):
        '''
        
        export tensors of all the italian dataset as X and relatives labels as y.
        
        '''
        self.it_X_y()
        self.it_X = np.array(self.it_X)
        np.save('tensors/it_X', self.it_X)

        self.it_y = np.array(self.it_y)
        np.save('tensors/it_y', self.it_y)

# Arrays().lan_save()
# Arrays().ge_save()
# Arrays().en_save()
# Arrays().fr_save()
# Arrays().it_save()