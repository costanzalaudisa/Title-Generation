from imports import *
from dataset import *

def createTitleModel(output):
    MAX_TOKENS = 5000
    EMBEDDING_DIMS = 64

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,), dtype=tf.string),
        TextVectorization(max_tokens = MAX_TOKENS, output_mode='int', output_sequence_length=None),
        tf.keras.layers.Embedding(
            input_dim=MAX_TOKENS+1,
            output_dim=EMBEDDING_DIMS,
            mask_zero=True),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(output, activation='softmax')
    ])

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer='adam',
              metrics=['accuracy'])

    return model

def createPlotModel(input, output):

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(256, input_dim=input, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(output, activation='softmax'))

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])

    return model