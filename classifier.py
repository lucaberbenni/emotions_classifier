from tensorflow import keras
import os
import time
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

from datas import AudioUtil
from recorder import Rec

class Classification:
    '''
    
    class for language prediction and feelings language based prediction of an audio recorded with microphone.
    
    '''
    def __init__(self):
        '''
        
        define self.path as models folder.
        
        '''
        self.path = 'models/'

    def models(self):
        '''
        
        import all models in self.path.
        
        '''
        for model in os.listdir(self.path):
            if model.endswith('h5'):
                if model.startswith('lan'):
                    self.lan_model = keras.models.load_model(self.path + model)

                if model.startswith('en'):
                    self.en_model = keras.models.load_model(self.path + model)

                if model.startswith('fr'):
                    self.fr_model = keras.models.load_model(self.path + model)

                if model.startswith('ge'):
                    self.ge_model = keras.models.load_model(self.path + model)

                if model.startswith('it'):
                    self.it_model = keras.models.load_model(self.path + model)

    def input(self):
        '''

        load and transform wav file and return it as an array.

        '''
        self.file = AudioUtil.open('input.wav')
        self.re_file = AudioUtil.resample(self.file, 16000)
        self.re_chan = AudioUtil.rechannel(self.re_file, 1)
        self.du_file = AudioUtil.pad_trunc(self.re_chan, 6000)
        self.spec = AudioUtil.spectro_gram(self.du_file, 
                                           n_mels= 376, 
                                           n_fft = 512, 
                                           hop_len = None)
        
        self.spec = self.spec.reshape((1, 376, 376, 1))
        self.spec = np.array(self.spec)

        return self.spec

    def lan_predict(self):
        '''
        
        predict language and return language label.
        
        '''
        self.models()

        self.lan_y = np.load('tensors/lan_y.npy')
        self.lan_classes = list(np.unique(self.lan_y))

        self.lan_pred = self.lan_model.predict(self.input())

        self.lan_dict_pred = dict(zip(self.lan_classes, self.lan_pred[0]))
        self.language = max(self.lan_dict_pred, key = self.lan_dict_pred.get)

        return self.language
    
    def en_predict(self):
        '''
        
        predict sentiment and return sentiment label.
        
        '''
        self.models()

        self.en_y = np.load('tensors/en_y.npy')
        self.en_classes = list(np.unique(self.en_y))

        self.en_pred = self.en_model.predict(self.input())

        self.en_dict_pred = dict(zip(self.en_classes, self.en_pred[0]))
        self.en_feel = max(self.en_dict_pred, key = self.en_dict_pred.get)

        return self.en_feel
    
    def ge_predict(self):
        '''
        
        predict sentiment and return sentiment label.
        
        '''
        self.models()

        self.ge_y = np.load('tensors/ge_y.npy')
        self.ge_classes = list(np.unique(self.ge_y))

        self.ge_pred = self.ge_model.predict(self.input())

        self.ge_dict_pred = dict(zip(self.ge_classes, self.ge_pred[0]))
        self.ge_feel = max(self.ge_dict_pred, key = self.ge_dict_pred.get)

        return self.ge_feel
    
    def it_predict(self):
        '''
        
        predict sentiment and return sentiment label.
        
        '''
        self.models()

        self.it_y = np.load('tensors/it_y.npy')
        self.it_classes = list(np.unique(self.it_y))

        self.it_pred = self.it_model.predict(self.input())

        self.it_dict_pred = dict(zip(self.it_classes, self.it_pred[0]))
        self.it_feel = max(self.it_dict_pred, key = self.it_dict_pred.get)

        return self.it_feel
    
    def fr_predict(self):
        '''
        
        predict sentiment and return sentiment label.
        
        '''
        self.models()

        self.fr_y = np.load('tensors/fr_y.npy')
        self.fr_classes = list(np.unique(self.fr_y))

        self.fr_pred = self.fr_model.predict(self.input())

        self.fr_dict_pred = dict(zip(self.fr_classes, self.fr_pred[0]))
        self.fr_feel = max(self.fr_dict_pred, key = self.fr_dict_pred.get)

        return self.fr_feel
    
    def run(self):
            '''
            
            activate the record function and return the sentiment prediction based on the language prediction.
            
            '''
            Rec().wav()

            self.lan_predict()
            self.en_predict()
            self.ge_predict()
            self.fr_predict()
            self.it_predict()
            self.time = datetime.now()
            self.lan_df = pd.DataFrame(columns = self.lan_dict_pred)
            self.en_df = pd.DataFrame(columns = self.en_dict_pred)
            self.ge_df = pd.DataFrame(columns = self.ge_dict_pred)
            self.it_df = pd.DataFrame(columns = self.it_dict_pred)
            self.fr_df = pd.DataFrame(columns = self.fr_dict_pred)

            if self.lan_predict() == 'en':
                self.lan_df = self.lan_df.append(self.lan_dict_pred, ignore_index = True)
                self.en_df = self.en_df.append(self.en_dict_pred, ignore_index = True)
            
            if self.lan_predict() == 'de':
                self.lan_df = self.lan_df.append(self.lan_dict_pred, ignore_index = True)
                self.ge_df = self.ge_df.append(self.ge_dict_pred, ignore_index = True)
            
            if self.lan_predict() == 'it':
                self.lan_df = self.lan_df.append(self.lan_dict_pred, ignore_index = True)
                self.it_df = self.it_df.append(self.it_dict_pred, ignore_index = True)
            
            if self.lan_predict() == 'fr':
                self.lan_df = self.lan_df.append(self.lan_dict_pred, ignore_index = True)
                self.fr_df = self.fr_df.append(self.fr_dict_pred, ignore_index = True)
        
            return self.lan_df, self.en_df, self.ge_df, self.it_df, self.fr_df

lan = pd.DataFrame()
en = pd.DataFrame()
ge = pd.DataFrame()
it = pd.DataFrame()
fr = pd.DataFrame()

st.title('Emotions classification')
st.write('Neural Network chain for emotions classification from speech in four different languages.')
st.markdown('---')

st.header('Goals')
st.write('-Spectrograms analysis from WAV datas')
st.write('-CNNs to detect language')
st.write('-CNNs to detect emotions')
st.write('-Live prediction from live recordings')
st.write('-Stremlit app to display results')
st.markdown('---')

st.header('Architecture')
st.image('flowcharts/str.png')
st.markdown('---')

st.header('Datas')
st.subheader('Common Voice')
col_1, col_2 = st.columns(2)
col_1.write('-4 languages')
col_1.write('-40.000 mp3 files')
col_2.write('The database is part of Common Voice, an open source, multi-language dataset.')
col_1, col_2, col_3 = st.columns(3)
col_2.markdown('[Link to Common Voice dataset](https://commonvoice.mozilla.org/en/datasets)')
st.subheader('CREMA-D')
col_1, col_2 = st.columns(2)
col_1.write('-6 emotions')
col_1.write('-7742 wav files')
col_2.write('The database is part of CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset, and is composed sentences spoken in a range of basic emotional states recorded from 91 actors with diverse ethnic backgrounds.')
col_1, col_2, col_3 = st.columns(3)
col_2.markdown('[Link to CREMA-D dataset](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4313618/)')
st.subheader('EmoDB')
col_1, col_2 = st.columns(2)
col_1.write('-7 emotions')
col_1.write('-535 wav files')
col_2.write('The database is created by the Institute of Communication Science, Technical University, Berlin, Germany. Ten professional speakers (five males and five females) participated in data recording.The EMODB database comprises of seven emotions: 1) anger; 2) boredom; 3) anxiety; 4) happiness; 5) sadness; 6) disgust; and 7) neutral.')
col_1, col_2, col_3 = st.columns(3)
col_2.markdown('[Link to EmoDB dataset](https://paperswithcode.com/dataset/emodb-dataset)')
st.subheader('EMOVO')
col_1, col_2 = st.columns(2)
col_1.write('-7 emotions')
col_1.write('-588 wav files')
col_2.write('The database is built from the voices of up to 6 actors who played 14 sentences simulating 6 emotional states (disgust, fear, anger, joy, surprise, sadness) plus the neutral state.')
col_1, col_2, col_3 = st.columns(3)
col_2.markdown('[Link to EMOVO dataset](https://paperswithcode.com/dataset/emovo)')
st.subheader('Att-Hack')
col_1, col_2 = st.columns(2)
col_1.write('-36.634 wav files')
col_1.write('-4 emotions')
col_2.write('The database is composed by 100 phrases with multiple versions/repetitions (3 to 5) in four social attitudes : friendly, distant, dominant and seductive.')
col_1, col_2, col_3 = st.columns(3)
col_2.markdown('[Link to Att-Hack dataset](http://www.openslr.org/88/)')
st.markdown('---')

st.header('Data transformation')
st.image('flowcharts/trans.png')
st.markdown('---')

st.header('Language classification')
st.image('flowcharts/lan.png')
lan_over = st.empty()

st.header('English emotions classification')
st.image('flowcharts/en.png')
en_over = st.empty()

st.header('German emotions classification')
st.image('flowcharts/ge.png')
ge_over = st.empty()

st.header('Italian emotions classification')
st.image('flowcharts/it.png')
it_over = st.empty()

st.header('French emotions classification')
st.image('flowcharts/fr.png')
fr_over = st.empty()
st.markdown('---')

st.header('Future steps')
st.write('-Hyperparameters optimization')
st.write('-Transfer learning')
st.write('-Recursive neural network')
st.markdown('---')

st.header('Tech stack')
col_1, col_2, col_3 = st.columns(3)
col_1.image('logos/python.png')
col_2.image('logos/pandas.png')
col_3.image('logos/numpy.png')
col_3.image('logos/tensorflow.png')
col_2.image('logos/torchaudio.png')
col_1.image('logos/streamlit.png')

while True:
    Classification().run()
    time.sleep(3)
    lan_df, en_df, ge_df, it_df, fr_df = Classification().run()
    lan = lan.append(lan_df, ignore_index = True)
    lan = lan.fillna(0)
    print(lan)
    with lan_over.container():
        
        st.dataframe(lan)
        st.bar_chart(lan)

    en = en.append(en_df, ignore_index = True)
    en = en.fillna(0)
    print(en)
    with en_over.container():

        st.dataframe(en)
        st.bar_chart(en)

    ge = ge.append(ge_df, ignore_index = True)
    ge = ge.fillna(0)
    print(ge)
    with ge_over.container():
        
        st.dataframe(ge)
        st.bar_chart(ge)

    it = it.append(it_df, ignore_index = True)
    it = it.fillna(0)
    print(it)
    with it_over.container():
        
        st.dataframe(it)
        st.bar_chart(it)

    fr = fr.append(fr_df, ignore_index = True)
    fr = fr.fillna(0)
    print(fr)
    with fr_over.container():
        
        st.dataframe(fr)
        st.bar_chart(fr)