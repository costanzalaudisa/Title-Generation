### Libraries import ###

# Utils
import numpy as np
import re

# Data processing
import pandas as pd
import matplotlib.pyplot as plt
import csv
from collections import Counter 

# Machine learning
import tensorflow as tf                     
from tensorflow import keras                    # pip install tensorflow
from keras.preprocessing.text import Tokenizer  # pip install keras
from keras.layers.experimental.preprocessing import StringLookup
from sklearn.feature_extraction.text import CountVectorizer

# Language processing
from string import punctuation
import nltk                                 # pip install ntlk
nltk.download('stopwords')
from nltk.corpus import stopwords
from unidecode import unidecode             # pip install Unidecode
from enchant.checker import SpellChecker    # pip install -U pyenchant

import tensorflow_datasets as tfds
import tensorflow as tf

from sklearn.model_selection import train_test_split

# Options
np.set_printoptions(threshold=np.inf) # print option to infinity

# Functions
max_error_count = 0
min_text_length = 0

def is_in_english(quote): # checks for english text (1 error as threshold)
    d = SpellChecker("en_US")
    d.set_text(quote)
    errors = [err.word for err in d]
    return False if ((len(errors) > max_error_count) or len(quote.split()) < min_text_length) else True