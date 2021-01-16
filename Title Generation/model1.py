from imports import *
from dataset import *

def createModel1(input, output):

    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=input,
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(output)
    ])

    return model
