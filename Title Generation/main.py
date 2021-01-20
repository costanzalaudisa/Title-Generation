from imports import *
from dataset import *
from model1 import *

# Gather dataset
#df = pd.read_csv('wiki_movie_plots_deduped.csv')

# Write cleaned version of dataset
#writeCleanedCsv(df)
#balanceDataSet('modified_ds.csv')

# Read modified non-balanced dataset
#df = pd.read_csv('modified_ds.csv', sep=";")
#print("Modified dataset shape:", df.shape)

# Read balanced data set
df = pd.read_csv('balanced_ds.csv', sep=";")

# Define plot vocabulary
#vocab = buildVocab(df)

# Create the input and output vectors
X = getPlotVectors(df)
#X = df['Title']
Y = getGenreVectors(df)

# Split dataset into training, validation, test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.10, stratify = Y, random_state = seed_value)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.25, stratify = y_train, random_state = seed_value)

print("Training set -> samples: ", X_train.shape, ", labels:", y_train.shape)
print("Validation set -> samples: ", X_val.shape, ", labels:", y_val.shape)
print("Test set -> samples: ", X_test.shape, ", labels:", y_test.shape)

# Standardize data (not necessary?)
#scaler = StandardScaler().fit(X_train)
#X_train = scaler.transform(X_train)
#X_val = scaler.transform(X_val)
#X_test = scaler.transform(X_test)

# Define and train models
INPUT_DIM = len(X.columns)
OUTPUT_DIM = len(Y.columns)
BUFFER_SIZE = 10000
BATCH_SIZE = 32
EPOCH_NUM = 25

#titleModel = createTitleModel(output_dim)
plotModel = createPlotModel(INPUT_DIM, OUTPUT_DIM)

history = plotModel.fit(X_train, y_train, 
                     epochs=EPOCH_NUM,
                     batch_size=BATCH_SIZE,
                     validation_data=(X_val, y_val))

plot_history(history)

test_loss, test_acc = plotModel.evaluate(X_test, y_test)
print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))