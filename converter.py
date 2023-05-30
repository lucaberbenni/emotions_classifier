import os
from pydub import AudioSegment

def converter(path):
    '''
    
    convert all mp3 files in path into wav files.
    
    '''
    for folder in os.listdir(path):

        if not folder.endswith('Store'):

            for file in os.listdir(path + folder):

                if file.endswith('mp3'):

                    name = file[:-4]
                    sound = AudioSegment.from_mp3(f'{path}{folder}/{file}')
                    sound.export(f'{path}{folder}/{name}.wav', format = 'wav')

# converter('data_languages/')