from imports import *
from dataset import *
from model1 import *
 
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Gather dataset
df = pd.read_csv('wiki_movie_plots_deduped.csv')
print("Original dataset shape:", df.shape)

#df = splitGenres(df) # not in use at the moment

#writeCleanedCsv(df)

#df = pd.read_csv('modified_ds.csv', sep=";")
#print("Modified dataset shape:", df.shape)

#getTitleVectors(df)

df = pd.read_csv('modified_ds.csv', sep=";")
#vocab = buildVocab(df)

X = getPlotVectors(df)

Y = getGenreVectors(df)

#new_frame = combineDataFrames(X, Y)

print(len(X.columns), len(Y.columns))

model1 = createModel1(len(X.columns), len(Y.columns))
model1.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

BUFFER_SIZE = 10000
BATCH_SIZE = 64

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

history = model1.fit(X_train, y_train, epochs=10)

test_loss, test_acc = model1.evaluate(X_test, y_test)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))