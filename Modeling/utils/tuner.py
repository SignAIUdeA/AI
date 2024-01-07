import keras_tuner
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters
import keras
from .metrics import f1, precision, recall
import numpy as np
import tensorflow as tf
import os

class CVTuner(keras_tuner.engine.tuner.Tuner):

    def __init__(self, data_cv, goal, hypermodel, oracle, project_name, directory, overwrite):
        self.data_cv = data_cv
        self.goal = goal
        self.trial_scores = []
        keras_tuner.engine.tuner.Tuner.__init__(self,
                                                hypermodel=hypermodel,
                                                oracle=oracle,
                                                project_name=project_name,
                                                directory=directory,
                                                overwrite=overwrite)

    def save_model(self, trial_id, model, step=0):
        filepath = os.path.join("models",self.get_trial_dir(trial_id), "model.h5")
        model.save(filepath)

    def run_trial(self, trial, x, y, batch_size=64, epochs=1):

        val_f1 = []
        val_precision = []
        val_recall = []

        for train_index, test_index in self.data_cv.split(x,y):

            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = self.hypermodel.build(trial.hyperparameters)

            callback = [keras.callbacks.EarlyStopping(monitor=self.goal, mode = 'max',patience=10, verbose=1)]

            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callback, verbose=1)
            y_pred = model.predict(x_test)

            f1 = model.evaluate(x_test, y_test)

            prec = precision(y_test, y_pred)
            rec = recall(y_test, np.round(y_pred))

            val_f1.append(f1)
            val_precision.append(prec)
            val_recall.append(rec)

            self.trial_scores.append(
                {
                    'trial_id': trial.trial_id,
                    'hyperparameters': trial.hyperparameters.values,
                    'f1': f1,
                    'f1_std': np.std(val_f1),
                    'precision': np.mean(val_precision),
                    'precision_std': np.std(val_precision),
                    'recall': np.mean(val_recall),
                    'recall_std': np.std(val_recall),
                }
            )

        self.oracle.update_trial(trial.trial_id, {self.goal: np.mean(val_f1)})
        self.save_model(trial.trial_id, model)



