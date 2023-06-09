This project is a language and emotions classification system that utilizes speech data to predict the language and emotional state of a speaker. The system is built using TensorFlow and Keras, and it incorporates various machine learning techniques, including spectrogram analysis and convolutional neural networks (CNNs).

## Project Overview

The main goal of this project is to develop a robust system for classifying languages and emotions from speech recordings. The system follows a pipeline that includes data transformation, language classification, and emotion classification. The data transformation step involves converting the input audio files into spectrograms, which are visual representations of the audio signal. These spectrograms are then used as input for the classification models.

The language classification model is trained on a dataset comprising recordings in multiple languages, including English, German, Italian, and French. It utilizes CNNs to predict the language of the input speech based on the spectrogram features. The emotion classification models are trained separately for each language and are designed to predict emotions such as anger, happiness, sadness, and more.

## Dataset

The project utilizes several datasets for training and evaluation, including:

- Common Voice: A multi-language dataset with thousands of recorded speech samples.
- CREMA-D: The Crowd-sourced Emotional Multimodal Actors Dataset, containing recordings of sentences spoken in various emotional states.
- EmoDB: The Emotion Database, consisting of recordings from professional speakers expressing different emotions.
- EMOVO: A dataset containing recordings of actors simulating emotional states.
- Att-Hack: A dataset comprising recordings with multiple versions in different social attitudes.

## Architecture

The system follows a modular architecture, with separate components for data processing, model training, and prediction. The code includes classes and functions for loading models, transforming audio data, and performing language and emotion predictions. The project also includes a Streamlit app that provides a user-friendly interface for live emotion prediction from recorded audio.

## Future Steps

The project has several areas for future improvement and expansion. Some potential next steps include:

- Hyperparameter optimization to fine-tune the performance of the classification models.
- Exploring transfer learning techniques to leverage pre-trained models for improved accuracy.
- Investigating the use of recursive neural networks (RNNs) to capture temporal dependencies in speech data.

## Tech Stack

The project is implemented using the following technologies and libraries:

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Streamlit
