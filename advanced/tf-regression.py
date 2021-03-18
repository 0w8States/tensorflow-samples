import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shutil
import os

# Supress tensorflow interanal logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

import reporter as rpt
import time

t0 = time.time()

# Constants
EPOCHS = 100
DATASET_FRACTION = 0.995 # Use 99.5% for training, 0.5% for testing

DIRECOTRY = "lpf_5_Regression_Test"
MODELS = f"{DIRECOTRY}/models"
REPORTS = f"{DIRECOTRY}/reports"
DATA = f"{DIRECOTRY}/reports/data"
PLOTS = f"{DIRECOTRY}/reports/plots"

# Configure Test Folder
if os.path.exists(DIRECOTRY):
	shutil.rmtree(DIRECOTRY)
	
# Configure the reporting folders
os.mkdir(DIRECOTRY)
os.mkdir(MODELS) 
os.mkdir(REPORTS) 
os.mkdir(DATA)  
os.mkdir(PLOTS) 
print("Directory {} created".format(DIRECOTRY)) 


# Print some Tensorflow diagnostics stuff
print("\nTensorflow Version: {}\n".format(tf.__version__))
print("\n# GPUs Available: {}\n".format(len(tf.config.experimental.list_physical_devices('GPU'))))

path = "weights_dataset.csv"
column_names = ['estimated_power', 'actual_position', 'position_error', 'output_force', 'weight']

raw_dataset = pd.read_csv(path, names=column_names, low_memory=False)

dataset = raw_dataset.copy()
print("\nDataset Head:")
print(dataset.head())

print("\nTotal Rows of Data: {}\n".format(len(dataset.index)))

print("\nNaN Evaluation:")
print(dataset.isna().sum())

print("\n\nSplitting Dataset...")
train_dataset = dataset.sample(frac=DATASET_FRACTION, random_state=0)
test_dataset = dataset.drop(train_dataset.index)  

# Diagnostics Plot
#sns.pairplot(train_dataset[['estimated_power', 'actual_position', 'position_error', 'output_force']], diag_kind='kde')

print("\n\nTransposing Dataset - Overall Statistics:")
print(train_dataset.describe().transpose())

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('weight')
test_labels = test_features.pop('weight')

# Normalize the data - Sample of how it looks
normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))

# Print demo for normalized values
#first = np.array(train_features[:1])
#with np.printoptions(precision=2, suppress=True):
#  print('\nFirst example:', first)
#  print()
#  print('\nNormalized:', normalizer(first).numpy())





#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#### Single Variable Linear Regression ###
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
# Apply a linear transformation (y=mx+b) to produce 1 output using layers.Dense.

# Normalize Output Force
outputforce = np.array(train_features['output_force'])
outputforce_normalizer = preprocessing.Normalization(input_shape=[1,])
outputforce_normalizer.adapt(outputforce)

# Build the sequential model
single_regress_model = tf.keras.Sequential([
    outputforce_normalizer,
    layers.Dense(units=1)
])

print("\nSingle Variable Linear Regression Model Details:\n")
print(single_regress_model.summary())

single_regress_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')


history = single_regress_model.fit(
    train_features['output_force'], train_labels,
    epochs=EPOCHS,
    # suppress logging
    verbose=1,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)

# Visualize the model's training progress
#hist = pd.DataFrame(history.history)
#hist['epoch'] = history.epoch
#hist.tail()

# Save the test results for later
test_results = {}
test_results['single_regress_model'] = single_regress_model.evaluate(
    test_features['output_force'],
    test_labels, verbose=0)

# Take a look at some sample predictions of outputforce 10 to 28, 250 samples
#x = tf.linspace(11.5, 27, 250)
#y = single_regress_model.predict(x)
#rpt.plot_predict_randomsample(x,y, 'reports/plots/SingleVarRegression_Prediction.png', train_features, train_labels)

# Plot the model loss
rpt.plot_loss(history, f'{PLOTS}/SingleVarRegression_Training.png')

# Export some rought prediction results to CSV
single_regress_model_prediction = single_regress_model.predict(test_features['output_force'])
rpt.csv_prediction_export(single_regress_model_prediction, test_labels, f'{DATA}/SingleVarRegression_Predictions.csv')

# Plot the predictions
single_regress_model_prediction = single_regress_model.predict(test_features['output_force']).flatten()
rpt.plot_predict(single_regress_model_prediction, test_labels, f'{PLOTS}/SingleVarRegression_Predictions.png')

# Plot the error
rpt.plot_predict_error(single_regress_model_prediction, test_labels, f'{PLOTS}/SingleVarRegression_Error.png')

# Export the model
single_regress_model.save(f'{MODELS}/single_regress_model')



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#### Multi-Variable Linear Regression ####
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#

# Apply a linear transformation (y=mx+b) except that m is a matrix and b is a vector.

multi_regress_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

print("\nMulti-Variable Linear Regression Model Details:\n")
print(multi_regress_model.summary())

multi_regress_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

# Train the model
history = multi_regress_model.fit(
    train_features, train_labels, 
    epochs=EPOCHS,
    # suppress logging
    verbose=1,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)

# Save for later
test_results['multi_regress_model'] = multi_regress_model.evaluate(
    test_features, test_labels, verbose=0)

# Plot the model loss
rpt.plot_loss(history, f'{PLOTS}/MultiVarRegression_Training.png')

# Export some rought prediction results to CSV
multi_regress_model_prediction = multi_regress_model.predict(test_features)
rpt.csv_prediction_export(multi_regress_model_prediction, test_labels, f'{DATA}/MultiVarRegression_Predictions.csv')

# Plot the predictions
multi_regress_model_prediction = multi_regress_model.predict(test_features).flatten()
rpt.plot_predict(multi_regress_model_prediction, test_labels, f'{PLOTS}/MultiVarRegression_Predictions.png')

# Plot the error
rpt.plot_predict_error(multi_regress_model_prediction, test_labels, f'{PLOTS}/MultiVarRegression_Error.png')

# Export the model
multi_regress_model.save(f'{MODELS}/multi_regress_model')



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
##### Single Variable DNN Regression #####
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#

def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

print("\nSingle Variable DNN Regression Model Details:\n")
single_dnn_model = build_and_compile_model(outputforce_normalizer)
print(single_dnn_model.summary())

history = single_dnn_model.fit(
    train_features['output_force'], train_labels,
    validation_split=0.2,
    verbose=1, epochs=EPOCHS)

# Save the model for later
test_results['single_dnn_model'] = single_dnn_model.evaluate(
    test_features['output_force'], test_labels,
    verbose=0)

# Plot the model loss
rpt.plot_loss(history, f'{PLOTS}/SingleVarDNN_Training.png')

# Export some rought prediction results to CSV
single_dnn_model_prediction = single_dnn_model.predict(test_features['output_force'])
rpt.csv_prediction_export(single_dnn_model_prediction, test_labels, f'{DATA}/SingleVarDNN_Predictions.csv')

# Plot the predictions
single_dnn_model_prediction = single_dnn_model.predict(test_features['output_force']).flatten()
rpt.plot_predict(single_dnn_model_prediction, test_labels, f'{PLOTS}/SingleVarDNN_Predictions.png')

# Plot the error
rpt.plot_predict_error(single_dnn_model_prediction, test_labels, f'{PLOTS}/SingleVarDNN_Error.png')

# Export the model
single_dnn_model.save(f'{MODELS}/single_dnn_model')




#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
##### Mutli Variable DNN Regression ######
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#

multi_dnn_model = build_and_compile_model(normalizer)

print("\nMulti-Variable DNN Regression Model Details:\n")
print(multi_dnn_model.summary())

history = multi_dnn_model.fit(
    train_features, train_labels,
    validation_split=0.2,
    verbose=1, epochs=EPOCHS)

# Save the model for later
test_results['multi_dnn_model'] = multi_dnn_model.evaluate(test_features, test_labels, verbose=0)

# Plot the model loss
rpt.plot_loss(history, f'{PLOTS}/MultiVarDNN_Training.png')

# Export some rought prediction results to CSV
multi_dnn_model_prediction = multi_dnn_model.predict(test_features)
rpt.csv_prediction_export(multi_dnn_model_prediction, test_labels, f'{DATA}/MultiVarDNN_Predictions.csv')

# Plot the predictions
multi_dnn_model_prediction = multi_dnn_model.predict(test_features).flatten()
rpt.plot_predict(multi_dnn_model_prediction, test_labels, f'{PLOTS}/MultiVarDNN_Predictions.png')

# Plot the error
rpt.plot_predict_error(multi_dnn_model_prediction, test_labels, f'{PLOTS}/MultiVarDNN_Error.png')

# Export the model
multi_dnn_model.save(f'{MODELS}/multi_dnn_model')




#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
############## Test Reports ##############
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#

print("\nAll Training Results:\n")
results = pd.DataFrame(test_results, index=['Mean absolute error [Weight]']).T
print(results)
results.to_csv(f'{DATA}/Model_Evaluation.csv')

t1 = time.time()
total_time = t1-t0
print("\nTotal Processing Time (s): {}".format(total_time))

