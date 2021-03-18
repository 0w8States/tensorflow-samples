
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Plots the loss of the model over the Epochs
def plot_loss(history, path):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [weight]')
  plt.legend()
  plt.grid(True)
  plt.savefig(path, dpi=300)
  plt.clf()


# Plots the precition of a single variable based regressor
def plot_predict_randomsample(x, y, path, train_features, train_labels):
  plt.scatter(train_features['output_force'], train_labels, label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('output_force')
  plt.ylabel('weight')
  plt.legend()
  plt.savefig(path, dpi=300)
  plt.clf()

# Plots the precition of a mutli-variable based regressor
def plot_predict(test_predictions, test_labels, path):
  a = plt.axes(aspect='equal')
  plt.scatter(test_labels, test_predictions)
  plt.xlabel('True Values [Weight]')
  plt.ylabel('Predictions [Weight]')
  lims = [0, 1400]
  plt.xlim(lims)
  plt.ylim(lims)
  plt.plot(lims, lims)
  plt.savefig(path, dpi=300)
  plt.clf()

# Plots the precition of a mutli-variable based regressor
def plot_predict_error(test_predictions, test_labels, path):
  error = test_predictions - test_labels
  plt.hist(error, bins=50)
  plt.xlabel('Prediction Error [Weight]')
  plt.ylabel('Count')
  plt.savefig(path, dpi=300)
  plt.clf()

# Export prediction results to CSV
def csv_prediction_export(predictions, test_labels, path):
  dataset = pd.DataFrame(test_labels)
  dataset['prediction'] = predictions[:, 0]
  dataset.to_csv(path, index=False)