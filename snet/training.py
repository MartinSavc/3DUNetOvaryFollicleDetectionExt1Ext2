import tensorflow.keras as keras

def get_callbacks(model_file, logging_file="training.log"):
    callbacks = list()
    callbacks.append(keras.callbacks.ModelCheckpoint(model_file, save_best_only=True))
    callbacks.append(keras.callbacks.ModelCheckpoint(model_file+'_last.h5'))
    callbacks.append(keras.callbacks.CSVLogger(logging_file, append=True))

    return callbacks


def train_model(model, model_file, training_generator, validation_generator, n_epochs=200):
    model.fit(training_generator,
            epochs=n_epochs,
            steps_per_epoch=len(training_generator),
            validation_data=validation_generator,
            verbose=1,
            callbacks=get_callbacks(model_file))
