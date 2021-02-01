from imports import *
from dataset import *
from confusion_matrix import *


def createTitleModel(input, output):
    MAX_TOKENS = 100
    EMBEDDING_DIMS = 512

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(MAX_TOKENS, EMBEDDING_DIMS, input_length=input),
        tf.keras.layers.LSTM(512),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(output, activation='softmax')
    ])

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

    return model


def createPlotModel(input, output):
    # Model 1
    model1 = tf.keras.Sequential()
    model1.add(tf.keras.layers.Dense(256, input_dim=input, activation='relu'))
    model1.add(tf.keras.layers.Dense(128, activation='relu'))
    model1.add(tf.keras.layers.Dense(64, activation='relu'))
    model1.add(tf.keras.layers.Dense(32, activation='relu'))
    model1.add(tf.keras.layers.Dense(output, activation='softmax'))

    model1.compile(loss=tf.keras.losses.CategoricalCrossentropy(), 
                  optimizer=tf.keras.optimizers.Adam(), 
                  metrics=['accuracy'])

    # Model 2
    model2 = tf.keras.Sequential()
    model2.add(tf.keras.layers.Dense(256, input_dim=input, activation='relu'))
    model2.add(tf.keras.layers.Dropout(0.8))
    model2.add(tf.keras.layers.Dense(128, activation='relu'))
    model2.add(tf.keras.layers.Dropout(0.5))
    model2.add(tf.keras.layers.Dense(64, activation='relu'))
    model2.add(tf.keras.layers.Dropout(0.2))
    model2.add(tf.keras.layers.Dense(32, activation='relu'))
    model2.add(tf.keras.layers.Dropout(0.1))
    model2.add(tf.keras.layers.Dense(output, activation='softmax'))

    model2.compile(loss=tf.keras.losses.CategoricalCrossentropy(), 
                  optimizer=tf.keras.optimizers.Adam(), 
                  metrics=['accuracy'])

    # Model 3
    model3 = tf.keras.Sequential()
    model3.add(tf.keras.layers.Dense(32, input_dim=input, activation='relu'))
    model3.add(tf.keras.layers.Dropout(0.8))
    model3.add(tf.keras.layers.Dense(64, activation='relu'))
    model3.add(tf.keras.layers.Dropout(0.5))
    model3.add(tf.keras.layers.Dense(128, activation='relu'))
    model3.add(tf.keras.layers.Dropout(0.2))
    model3.add(tf.keras.layers.Dense(256, activation='relu'))
    model3.add(tf.keras.layers.Dropout(0.1))
    model3.add(tf.keras.layers.Dense(output, activation='softmax'))

    model3.compile(loss=tf.keras.losses.CategoricalCrossentropy(), 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  metrics=['accuracy'])

    # Model 3
    model3d = tf.keras.Sequential()
    model3d.add(tf.keras.layers.Dense(32, input_dim=input, activation='relu'))
    model3d.add(tf.keras.layers.Dense(64, activation='relu'))
    model3d.add(tf.keras.layers.Dense(128, activation='relu'))
    model3d.add(tf.keras.layers.Dense(256, activation='relu'))
    model3d.add(tf.keras.layers.Dense(output, activation='softmax'))

    model3d.compile(loss=tf.keras.losses.CategoricalCrossentropy(), 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  metrics=['accuracy'])

    # Model 4
    model4 = tf.keras.Sequential()
    model4.add(tf.keras.layers.Dense(256, input_dim=input, activation='relu'))
    model4.add(tf.keras.layers.Dropout(0.8))
    model4.add(tf.keras.layers.Dense(256, activation='relu'))
    model4.add(tf.keras.layers.Dropout(0.8))
    model4.add(tf.keras.layers.Dense(256, activation='relu'))
    model4.add(tf.keras.layers.Dropout(0.5))
    model4.add(tf.keras.layers.Dense(256, activation='relu'))
    model4.add(tf.keras.layers.Dropout(0.5))
    model4.add(tf.keras.layers.Dense(output, activation='softmax'))

    model4.compile(loss=tf.keras.losses.CategoricalCrossentropy(), 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  metrics=['accuracy'])

    # Model 5
    model5 = tf.keras.Sequential()
    model5.add(tf.keras.layers.Dense(256, input_dim=input, activation='relu'))
    model5.add(tf.keras.layers.Dropout(0.8))
    model5.add(tf.keras.layers.Dense(128, activation='relu'))
    model5.add(tf.keras.layers.Dropout(0.5))
    model5.add(tf.keras.layers.Dense(64, activation='relu'))
    model5.add(tf.keras.layers.Dropout(0.2))
    model5.add(tf.keras.layers.Dense(32, activation='relu'))
    model5.add(tf.keras.layers.Dropout(0.1))
    model5.add(tf.keras.layers.Dense(output, activation='softmax'))

    model5.compile(loss=tf.keras.losses.CategoricalCrossentropy(), 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), 
                  metrics=['accuracy'])

    # Model 6
    model6 = tf.keras.Sequential()
    model6.add(tf.keras.layers.Dense(256, input_dim=input, activation='relu'))
    model6.add(tf.keras.layers.Dropout(0.8))
    model6.add(tf.keras.layers.Dense(128, activation='relu'))
    model6.add(tf.keras.layers.Dropout(0.5))
    model6.add(tf.keras.layers.Dense(64, activation='relu'))
    model6.add(tf.keras.layers.Dropout(0.2))
    model6.add(tf.keras.layers.Dense(32, activation='relu'))
    model6.add(tf.keras.layers.Dropout(0.1))
    model6.add(tf.keras.layers.Dense(output, activation='softmax'))

    # Different optimizers and learning rates
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                                                        initial_learning_rate=1e-2,
                                                        decay_steps=10000,
                                                        decay_rate=0.9)
    sgd_opt = keras.optimizers.SGD(learning_rate=lr_schedule)

    model6.compile(loss=tf.keras.losses.CategoricalCrossentropy(), 
                  optimizer=sgd_opt, 
                  metrics=['accuracy'])    

    # Regularized model
    model_reg = tf.keras.Sequential()
    model_reg.add(tf.keras.layers.Dense(1024, input_dim=input, activation='relu', kernel_regularizer='l2'))
    model_reg.add(tf.keras.layers.Dense(512, activation='relu', kernel_regularizer='l2'))
    model_reg.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer='l2'))
    model_reg.add(tf.keras.layers.Dense(output, activation='softmax'))

    model_reg.compile(loss=tf.keras.losses.CategoricalCrossentropy(), 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  metrics=['accuracy'])

    # Choose model to return
    return model2


def mergeModels(plot_X, title_X, Y):
    # Split all data properly
    plot_X_train, plot_X_test, title_X_train, title_X_test, y_train, y_test = train_test_split(plot_X, title_X, Y, test_size = 0.10, stratify = Y, random_state = seed_value)
    
    print("--------------------PLOT--------------------")
    print("Training set -> samples: ", plot_X_train.shape, ", labels:", y_train.shape)
    print("Test set -> samples: ", plot_X_test.shape, ", labels:", y_test.shape)

    print("--------------------TITLE--------------------")
    print("Training set -> samples: ", title_X_train.shape, ", labels:", y_train.shape)
    print("Test set -> samples: ", title_X_test.shape, ", labels:", y_test.shape)

    # Define parameters
    INPUT_P_DIM = len(plot_X.columns)
    INPUT_T_DIM = len(title_X.columns)
    OUTPUT_DIM = len(Y.columns)

    MAX_TOKENS = 100
    EMBEDDING_DIMS = 512

    BUFFER_SIZE = 10000
    BATCH_SIZE = 32
    EPOCH_NUM = 25

    # Define the two sets of inputs
    input_P = tf.keras.layers.Input(shape=(INPUT_P_DIM,))
    input_T = tf.keras.layers.Input(shape=(INPUT_T_DIM,))
    
    # First branch operates on the first input (plots)
    x = tf.keras.layers.Dense(256, activation="relu")(input_P)
    x = tf.keras.layers.Dropout(0.8)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.Model(inputs=input_P, outputs=x)
    
    # Second branch operates on the second input (titles)
    y = tf.keras.layers.Embedding(MAX_TOKENS, EMBEDDING_DIMS)(input_T)
    y = tf.keras.layers.LSTM(512)(y)
    y = tf.keras.layers.Dense(512, activation='relu')(y)
    y = tf.keras.Model(inputs=input_T, outputs=y)

    # Combine the output of the two branches
    combined = tf.keras.layers.concatenate([x.output, y.output])
    z = tf.keras.layers.Dense(OUTPUT_DIM, activation="softmax")(combined)

    # Model accepts 2 inputs and outputs predictions
    model = tf.keras.Model(inputs=[x.input, y.input], outputs=z)

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), 
                  optimizer=tf.keras.optimizers.Adam(), 
                  metrics=['accuracy'])

    history = model.fit([plot_X_train, title_X_train], y_train, 
                         epochs=EPOCH_NUM,
                         batch_size=BATCH_SIZE,
                         validation_data=([plot_X_test, title_X_test], y_test))

    plot_history(history)

    # Display confusion matrix
    y_pred = model.predict([plot_X_test, title_X_test]).argmax(axis=-1)
    y_labels = y_test.idxmax(axis=1)
    #cm_analysis(y_labels,y_pred ,range(10), ymap = {0: "action", 1:"science-fiction",2: "drama",3:"comedy",4:"horror",5: "thriller",6:"crime",7: "western",8:"adventure",9:"music"}, figsize=(20,14)) # 10 GENRES
    cm_analysis(y_labels,y_pred ,range(5), ymap = {0:"science-fiction", 1: "drama", 2: "horror", 3: "crime", 4: "western"}, figsize=(20,14)) # 5 BEST-SCORING GENRES


def crossValidation(INPUT_DIM, OUTPUT_DIM, X, Y, BATCH_SIZE, EPOCH_NUM):
    # Convert to Numpy arrays
    X = X.to_numpy()
    Y = Y.to_numpy()

    # Initialize lists to store results
    train_acc_per_fold = []
    train_loss_per_fold = []
    acc_per_fold = []
    loss_per_fold = []


    # Define cross-validation values
    n_splits = 5
    kfold = KFold(n_splits=n_splits, shuffle=True)
    fold_no = 1

    for train, test in kfold.split(X, Y):

        # Build and train model
        model = createPlotModel(INPUT_DIM, OUTPUT_DIM)

        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no}...')

        history = model.fit(X[train], Y[train],
                  batch_size=BATCH_SIZE,
                  epochs=EPOCH_NUM,
                  validation_data=(X[test], Y[test]))
        
        # Store results
        train_acc_per_fold.append(history.history['accuracy'][-1] * 100)
        train_loss_per_fold.append(history.history['loss'][-1])

        # Evaluate on validation model and 
        scores = model.evaluate(X[test], Y[test], verbose=0)
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        # Increase fold
        fold_no += 1
            
    # Calculate average values
    avg_train_acc = sum(train_acc_per_fold)/n_splits
    avg_train_loss = sum(train_loss_per_fold)/n_splits
    avg_acc = sum(acc_per_fold)/n_splits
    avg_loss = sum(loss_per_fold)/n_splits

    print('------------------------------------------------------------------------')
    print("Average training loss: ", avg_train_loss)
    print("Average training accuracy: ", avg_train_acc)
    print('------------------------------------------------------------------------')
    print("Average validation loss: ", avg_loss)
    print("Average validation accuracy: ", avg_acc)