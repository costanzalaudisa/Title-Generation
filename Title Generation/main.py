from imports import *
from dataset import *
from model1 import *
 
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import tensorflow as tf
print("Is GPU available: ", tf.test.is_gpu_available())
print("Is Tensorflow buit with CUDA: ", tf.test.is_built_with_cuda())

# Gather dataset
#df = pd.read_csv('wiki_movie_plots_deduped.csv')
#print("Original dataset shape:", df.shape)

#writeCleanedCsv(df)
#balanceDataSet('modified_ds.csv')

#df = pd.read_csv('modified_ds.csv', sep=";")
#print("Modified dataset shape:", df.shape)

#getTitleVectors(df)

#df = pd.read_csv('modified_ds.csv', sep=";")
#vocab = buildVocab(df)

# Read the balanced data set, create the input and output vectors and train the model

df = pd.read_csv('balanced_ds.csv', sep=";")
X = getPlotVectors(df)
Y = getGenreVectors(df)

print(len(X.columns), len(Y.columns))

model1 = createModel1(len(X.columns), len(Y.columns))
model2 = createModel2(6971, len(Y.columns))

#BUFFER_SIZE = 10000
#BATCH_SIZE = 64

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

history = model2.fit(X_train, y_train, epochs=10)

test_loss, test_acc = model2.evaluate(X_test, y_test)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))