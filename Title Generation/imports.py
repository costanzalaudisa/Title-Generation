from reproducible import *

### Libraries import ###

# Utils
import re

# Data processing
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import csv
from collections import Counter 

# Machine learning
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))
print("Is GPU available:", tf.test.is_gpu_available())
print("Is Tensorflow built with CUDA:", tf.test.is_built_with_cuda())

from tensorflow import keras                    # pip install tensorflow
from keras.preprocessing.text import Tokenizer  # pip install keras
from keras.layers.experimental.preprocessing import StringLookup, TextVectorization
from sklearn.feature_extraction.text import CountVectorizer

# Language processing
from string import punctuation
import nltk                                 # pip install ntlk
nltk.download('stopwords')
from nltk.corpus import stopwords
from unidecode import unidecode             # pip install Unidecode
from enchant.checker import SpellChecker    # pip install -U pyenchant

from sklearn.model_selection import train_test_split


### Options ###

# Print option to infinity
np.set_printoptions(threshold=np.inf)

# Force CPU only (uncomment for CPU, comment for GPU if available)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


### Functions ###

# Function to check for english text using a spellchecker 
max_error_count = 0     # 1 error as threshold
min_text_length = 0
def is_in_english(quote): 
    d = SpellChecker("en_US")
    d.set_text(quote)
    errors = [err.word for err in d]
    return False if ((len(errors) > max_error_count) or len(quote.split()) < min_text_length) else True

# Function to plot model's history
def plot_history(history):
    # Plot loss
    plt.plot(history.history['loss'], label='Train. loss')
    plt.plot(history.history['val_loss'], label='Val. loss')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc="upper left")
    plt.show()

    # Plot loss
    plt.plot(history.history['accuracy'], label='Train. accuracy')
    plt.plot(history.history['val_accuracy'], label='Val. accuracy')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc="upper left")
    plt.show()