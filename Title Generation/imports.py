### Libraries import ###

# Utils
import numpy as np

# Data processing
import pandas as pd
import matplotlib.pyplot as plt

# Machine learning
import tensorflow as tf
from tensorflow import keras
import csv
from collections import Counter 
from string import punctuation
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer

# Options
np.set_printoptions(threshold=np.inf)