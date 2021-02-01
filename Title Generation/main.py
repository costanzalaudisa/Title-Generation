from imports import *
from dataset import *
from model1 import *

#############################################
##### Loading and preprocessing dataset #####
#############################################

# Gather dataset [no need to execute]
#df = pd.read_csv('wiki_movie_plots_deduped.csv')

# Write cleaned version of dataset [no need to execute as files are already provided]
#writeCleanedCsv(df)
#balanceDataSet('modified_ds_red.csv')

# Read dataset (choose one)
df = pd.read_csv('balanced_ds.csv', sep=";")

# Define plot vocabulary [no need to execute as files are already provided]
#vocab = buildVocab(df)

# Create the input and output vectors
#X = getPlotVectors(df)
X = getTitleVectors(df)
Y = getGenreVectors(df)

# Standardize the input (only if reading plots)
normalized_X = (X-X.mean())/X.std()

# Split dataset into training, validation, test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.10, stratify = Y, random_state = seed_value) 
#X_train, X_test, y_train, y_test = train_test_split(normalized_X, Y, test_size = 0.10, stratify = Y, random_state = seed_value) # if using normalized dataset
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.25, stratify = y_train, random_state = seed_value)

print("Training set -> samples: ", X_train.shape, ", labels:", y_train.shape)
print("Validation set -> samples: ", X_val.shape, ", labels:", y_val.shape)
print("Test set -> samples: ", X_test.shape, ", labels:", y_test.shape)

# Define parameters and train models
INPUT_DIM = len(X.columns)
OUTPUT_DIM = len(Y.columns)
BUFFER_SIZE = 10000
BATCH_SIZE = 16
EPOCH_NUM = 25


############################################################
##### Single model training and prediction with graphs #####
############################################################

### Train models on plots ###

# Build and train model on plots
plotModel = createPlotModel(INPUT_DIM, OUTPUT_DIM)

history = plotModel.fit(X_train, y_train, 
                     epochs=EPOCH_NUM,
                     batch_size=BATCH_SIZE,
                     validation_data=(X_val, y_val))

plot_history(history)

test_loss, test_acc = plotModel.evaluate(X_test, y_test)
print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

# Display confusion matrix
y_pred = plotModel.predict(X_test).argmax(axis=-1)
y_labels = y_test.idxmax(axis=1)
cm_analysis(y_labels,y_pred ,range(10), ymap = {0: "action", 1:"science-fiction",2: "drama",3:"comedy",4:"horror",5: "thriller",6:"crime",7: "western",8:"adventure",9:"music"}
, figsize=(20,14))

### Train models on titles ###

## Build and train model on titles
#titleModel = createTitleModel(INPUT_DIM, OUTPUT_DIM)

#history = titleModel.fit(X_train, y_train, 
#                     epochs=EPOCH_NUM,
#                     batch_size=BATCH_SIZE,
#                     validation_data=(X_val, y_val))

#plot_history(history)

#test_loss, test_acc = titleModel.evaluate(X_test, y_test)
#print('Test Loss: {}'.format(test_loss))
#print('Test Accuracy: {}'.format(test_acc))

## Display confusion matrix
#y_pred = titleModel.predict(X_test).argmax(axis=-1)
#y_labels = y_test.idxmax(axis=1)
#cm_analysis(y_labels,y_pred ,range(10), ymap = {0: "action", 1:"science-fiction",2: "drama",3:"comedy",4:"horror",5: "thriller",6:"crime",7: "western",8:"adventure",9:"music"}
#, figsize=(20,14))


#################################################################################
##### Cross-validation method. Model is specified in functions in model1.py #####
#################################################################################

crossValidation(INPUT_DIM, OUTPUT_DIM, X, Y, BATCH_SIZE, EPOCH_NUM)


#####################################################
##### Single models training and merging models #####
#####################################################

plot_X = getPlotVectors(df)
title_X = getTitleVectors(df)
Y = getGenreVectors(df)

mergeModels(plot_X, title_X, Y)  
