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

# Tunable Variables
EPOCHS = 100
DATASET_FRACTION = 0.995 # Use 99.5% for training, 0.5% for testing

DIRECOTRY = "lpf_05_DNNSingle_Test_v2"
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
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)


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


# Normalize Output Force
outputforce = np.array(train_features['output_force'])
outputforce_normalizer = preprocessing.Normalization(input_shape=[1,])
outputforce_normalizer.adapt(outputforce)


# Save the test results for later
test_results = {}

# determine the number of input features
n_features = train_features.shape[1]
print(f'Number of features: {n_features}')

################################
####### Detect Hardware ########
################################
try:
  tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
except ValueError:
  tpu = None
  gpus = tf.config.list_logical_devices("GPU")
    
# Select appropriate distribution strategy for hardware
if tpu:
  tf.config.experimental_connect_to_cluster(tpu)
  tf.tpu.experimental.initialize_tpu_system(tpu)
  strategy = tf.distribute.TPUStrategy(tpu)
  print('Running on TPU ', tpu.master())  
elif len(gpus) > 0:
  strategy = tf.distribute.MirroredStrategy(gpus) # this works for 1 to multiple GPUs
  print('Running on ', len(gpus), ' GPU(s) ')
else:
  strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
  print('Running on CPU')

# How many accelerators do we have ?
print("Number of accelerators: ", strategy.num_replicas_in_sync)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
##### Single Variable DNN Regression #####
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
def single_dnn():
  with strategy.scope():
    # define model
    model = keras.Sequential()
    model.add(layers.Dense(64, activation='relu', kernel_initializer='he_normal', input_shape=(1,), name="layer1"))
    model.add(layers.Dense(64, activation='relu', kernel_initializer='he_normal', name="layer2"))
    model.add(layers.Dense(1))
    # compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_absolute_error')
    print("\nSingle-Variable DNN Regression Model Details:\n")
    print(model.summary())
    # fit the model
    history = model.fit(train_features['output_force'], train_labels, epochs=EPOCHS, verbose=1, validation_split=0.2) # batch_size=32

	# Save the model for later
  test_results['single_dnn_model'] = model.evaluate(test_features['output_force'], test_labels, verbose=0)

	# Plot the model loss
  rpt.plot_loss(history, f'{PLOTS}/SingleVarDNN_Training.png')

	# Export some rought prediction results to CSV
  single_dnn_model_prediction = model.predict(test_features['output_force'])
  rpt.csv_prediction_export(single_dnn_model_prediction, test_labels, f'{DATA}/SingleVarDNN_Predictions.csv')

	# Plot the predictions
  single_dnn_model_prediction = model.predict(test_features['output_force']).flatten()
  rpt.plot_predict(single_dnn_model_prediction, test_labels, f'{PLOTS}/SingleVarDNN_Predictions.png')

	# Plot the error
  rpt.plot_predict_error(single_dnn_model_prediction, test_labels, f'{PLOTS}/SingleVarDNN_Error.png')

	# Export the model
  model.save(f'{MODELS}/single_dnn_model', overwrite=True, include_optimizer=False, save_format='tf')

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
##### Single Variable DNN Regression #####
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
def single_dnn_normalized():
  with strategy.scope():
    # define model
    model = keras.Sequential()
    model.add(layers.Dense(64, activation='relu', kernel_initializer='he_normal', input_shape=(1,), name="layer1"))
    #model.add(layers.BatchNormalization())
    model.add(layers.Dense(64, activation='relu', kernel_initializer='he_normal', name="layer2"))
    #model.add(layers.Dense(1, activation='sigmoid'))
    model.add(layers.Dense(1))
    
    # compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_absolute_error')
    print("\nSingle-Variable DNN Regression Model Details:\n")
    print(model.summary())
    # configure early stopping
    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    # fit the model
    history = model.fit(train_features['output_force'], train_labels, epochs=EPOCHS, verbose=1, validation_split=0.2, callbacks=[es]) # batch_size=32

	# Save the model for later
  test_results['single_dnn_norm_model'] = model.evaluate(test_features['output_force'], test_labels, verbose=0)

	# Plot the model loss
  rpt.plot_loss(history, f'{PLOTS}/SingleVarDNN_Norm_Training.png')

	# Export some rought prediction results to CSV
  single_dnn_model_prediction = model.predict(test_features['output_force'])
  rpt.csv_prediction_export(single_dnn_model_prediction, test_labels, f'{DATA}/SingleVarDNN_Norm_Predictions.csv')

	# Plot the predictions
  single_dnn_model_prediction = model.predict(test_features['output_force']).flatten()
  rpt.plot_predict(single_dnn_model_prediction, test_labels, f'{PLOTS}/SingleVarDNN_Norm_Predictions.png')

	# Plot the error
  rpt.plot_predict_error(single_dnn_model_prediction, test_labels, f'{PLOTS}/SingleVarDNN_Norm_Error.png')

	# Export the model
  model.save(f'{MODELS}/single_dnn_normalized', overwrite=True, include_optimizer=False, save_format='tf')

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
##### Mutli Variable DNN Regression ######
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
def multi_dnn():
    with strategy.scope():
        # define model
        model = keras.Sequential()
        model.add(layers.Dense(64, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
        model.add(layers.Dense(64, activation='relu', kernel_initializer='he_normal'))
        model.add(layers.Dense(32, activation='relu', kernel_initializer='he_normal'))
        model.add(layers.Dense(16, activation='relu', kernel_initializer='he_normal'))
        model.add(layers.Dense(8, activation='relu', kernel_initializer='he_normal'))
        model.add(layers.Dense(1))
        # compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_absolute_error')

        print("\nMulti-Variable DNN Regression Model Details:\n")
        print(model.summary())

        # fit the model
        history = model.fit(train_features, train_labels, epochs=EPOCHS, verbose=1, validation_split=0.2) # batch_size=32
    	# Save the model for later

    test_results['multi_dnn_model'] = model.evaluate(test_features, test_labels, verbose=0)

	# Plot the model loss
    rpt.plot_loss(history, f'{PLOTS}/MultiVarDNN_Training.png')
    # list all data in history
    #print("History")
    #print(history.history.keys())

	# Export some rought prediction results to CSV
    multi_dnn_model_prediction = model.predict(test_features)
    rpt.csv_prediction_export(multi_dnn_model_prediction, test_labels, f'{DATA}/MultiVarDNN_Predictions.csv')

	# Plot the predictions
    multi_dnn_model_prediction = model.predict(test_features).flatten()
    rpt.plot_predict(multi_dnn_model_prediction, test_labels, f'{PLOTS}/MultiVarDNN_Predictions.png')

	# Plot the error
    rpt.plot_predict_error(multi_dnn_model_prediction, test_labels, f'{PLOTS}/MultiVarDNN_Error.png')

	# Export the model
    model.save(f'{MODELS}/multi_dnn_model', overwrite=True, include_optimizer=False, save_format='tf')#.save(f'{MODELS}/multi_dnn_model')


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
############## Test Reports ##############
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
def print_evaluation():
	print("\nAll Training Results:\n")
	results = pd.DataFrame(test_results, index=['Mean absolute error [Weight]']).T
	print(results)
	results.to_csv(f'{DATA}/Model_Evaluation.csv')

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
############## MAIN Section ##############
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
t0 = time.time()

#single_regression()
#mutli_regression()
#single_dnn()
single_dnn_normalized()
#multi_dnn()


print_evaluation()

t1 = time.time()
total_time = t1-t0
print("\nProgram Complete - Total Processing Time (s): {}".format(total_time))
