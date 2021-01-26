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

    #model.add(tf.keras.layers.Dense(32, input_dim=input, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.8))
    #model.add(tf.keras.layers.Dense(64, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.5))
    #model.add(tf.keras.layers.Dense(128, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.2))
    #model.add(tf.keras.layers.Dense(64, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.1))

    #model.add(tf.keras.layers.Dense(256, input_dim=input, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.8))
    #model.add(tf.keras.layers.Dense(256, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.5))
    #model.add(tf.keras.layers.Dense(256, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.2))
    #model.add(tf.keras.layers.Dense(256, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.1))

    #model.add(tf.keras.layers.Dense(32, input_dim=input, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.8))
    #model.add(tf.keras.layers.Dense(64, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.5))
    #model.add(tf.keras.layers.Dense(128, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.2))
    #model.add(tf.keras.layers.Dense(256, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.1))

    model.add(tf.keras.layers.Dense(256, input_dim=input, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.8))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.1))

    model.add(tf.keras.layers.Dense(output, activation='softmax'))

    ### Different optimizers and learning rates
    #opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    #lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #                                                    initial_learning_rate=1e-2,
    #                                                    decay_steps=10000,
    #                                                    decay_rate=0.9)
    #optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

    return model

def crossValidation(INPUT_DIM, OUTPUT_DIM, X, Y, BATCH_SIZE, EPOCH_NUM):
    X = X.to_numpy()
    Y = Y.to_numpy()
    n_splits = 5
    kfold = KFold(n_splits=n_splits, shuffle=True)
    acc_per_fold = []
    loss_per_fold = []

    fold_no = 1
    for train, test in kfold.split(X, Y):
        model = createPlotModel(INPUT_DIM, OUTPUT_DIM)
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        history = model.fit(X[train], Y[train],
                  batch_size=BATCH_SIZE,
                  epochs=EPOCH_NUM,
                  validation_data=(X[test], Y[test]))
        # Generate generalization metrics
        scores = model.evaluate(X[test], Y[test], verbose=0)
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        fold_no += 1
        
    plot_history(history)

    avg_acc = sum(acc_per_fold)/n_splits
    avg_loss = sum(loss_per_fold)/n_splits
    print('------------------------------------------------------------------------')
    print("Final average accuracy: ", avg_acc)
    print('------------------------------------------------------------------------')
    print("Final average loss: ", avg_loss)

def createHyperModels(input, output):

    model_list = []

    model_1 = model = tf.keras.Sequential()
    model_1.add(tf.keras.layers.Dense(256, input_dim=input, activation='relu'))
    model_1.add(tf.keras.layers.Dropout(0.8))
    model_1.add(tf.keras.layers.Dense(128, activation='relu'))
    model_1.add(tf.keras.layers.Dropout(0.5))
    model_1.add(tf.keras.layers.Dense(64, activation='relu'))
    model_1.add(tf.keras.layers.Dropout(0.2))
    model_1.add(tf.keras.layers.Dense(32, activation='relu'))
    model_1.add(tf.keras.layers.Dropout(0.1))
    model_1.add(tf.keras.layers.Dense(output, activation='softmax'))
    model_1.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    model_list.append(model_1)


    return model_list

