from imports import *
from dataset import *
from model1 import *
from confusion_matrix import *

# Gather dataset
#df = pd.read_csv('wiki_movie_plots_deduped.csv')

# Write cleaned version of dataset
#writeCleanedCsv(df)
#balanceDataSet('modified_ds_red.csv')

# Read modified non-balanced dataset
#df = pd.read_csv('modified_ds.csv', sep=";")
#print("Modified dataset shape:", df.shape)

# Read balanced data set
df = pd.read_csv('balanced_ds_red.csv', sep=";")

# Define plot vocabulary
#vocab = buildVocab(df)

# Create the input and output vectors
X = getPlotVectors(df)
Y = getGenreVectors(df)

#standardize or normalize the input
normalized_X=(X-X.mean())/X.std()

# Split dataset into training, validation, test set
X_train, X_test, y_train, y_test = train_test_split(normalized_X, Y, test_size = 0.10, stratify = Y, random_state = seed_value)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.25, stratify = y_train, random_state = seed_value)

print("Training set -> samples: ", X_train.shape, ", labels:", y_train.shape)
print("Validation set -> samples: ", X_val.shape, ", labels:", y_val.shape)
print("Test set -> samples: ", X_test.shape, ", labels:", y_test.shape)

# Define and train models
INPUT_DIM = len(X.columns)
OUTPUT_DIM = len(Y.columns)
BUFFER_SIZE = 10000
BATCH_SIZE = 32
EPOCH_NUM = 30


#####################################################################################
#####   Single model training and prediction with graphs                        #####
#####################################################################################

#titleModel = createTitleModel(output_dim)
#plotModel = createPlotModel(INPUT_DIM, OUTPUT_DIM)

## Add tensorboard for analysis
#log_dir = "logs/fit/"
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#history = plotModel.fit(X_train, y_train, 
#                     epochs=EPOCH_NUM,
#                     batch_size=BATCH_SIZE,
#                     validation_data=(X_val, y_val),
#                     callbacks=[tensorboard_callback])

#plot_history(history)

##display confusion matrix
#y_pred = plotModel.predict(X_test).argmax(axis=-1)
#y_labels = y_test.idxmax(axis=1)
#cm_analysis(y_labels,y_pred ,range(10), ymap = {0: "action", 1:"science-fiction",2: "drama",3:"comedy",4:"horror",5: "thriller",6:"crime",7: "western",8:"adventure",9:"music"}
#, figsize=(20,14))

#test_loss, test_acc = plotModel.evaluate(X_test, y_test)
#print('Test Loss: {}'.format(test_loss))
#print('Test Accuracy: {}'.format(test_acc))

# Try out K-fold cross validation
#plotModel = createPlotModel(INPUT_DIM, OUTPUT_DIM)

###############################################################################################
#####   Cross-validation method. The model is specified in the function in model1.py      #####
###############################################################################################

crossValidation(INPUT_DIM, OUTPUT_DIM, normalized_X, Y, BATCH_SIZE, EPOCH_NUM)


