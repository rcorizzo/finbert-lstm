import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import keras
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import date, timedelta
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input, Flatten, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from keras.layers import LSTM
from keras.layers import Conv1D, LSTM, BatchNormalization, Dropout, Dense
from keras.layers import GRU, Conv1D, GlobalMaxPooling1D, BatchNormalization, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense, Attention, Input, TimeDistributed
from tensorflow.keras.optimizers import Adam

from google.colab import drive
drive.mount('/content/drive')

triker1 = 'AAPL'
triker = triker1.lower()

trend = 'downtrend'
# trend = 'uptrend'
tmax_trials = 3
tepochs = 15
fepochs = 1
# tmax_trials = 1
# tepochs = 1
# fepochs = 20

# PATH = 'C:/Users/Simon/stock/stock_data/'
# path = 'C:/Users/Simon/stock/'
PATH = '/content/drive/MyDrive/stock_data/'
path = '/content/drive/MyDrive/stock_info/'

ns = ['adbe', 'amt', 'pld', 'vici', 'schw', 'jpm', 'atvi', 'cvs', 'bio', 'jnj']

def se(triker):
    if triker in ns:
      stock = pd.read_csv(PATH + triker + '_Simon/stocks_ts_'+triker1+'_2021-1-4_2022-9-20.csv')
      stock = stock[stock['date']>= '2021-01-01']
      stock = stock.iloc[:,0:25].drop(columns=['symbol', 'NextDayClose', 'NextDayTrend',	'PrevDayTrend'])
    else:
      stock = pd.read_csv(PATH + triker + '_Simon/stocks_ts_'+triker1+'_2021-1-4_2022-9-20.csv')
      stock = stock[stock['date']>= '2021-01-01']
      stock = stock.iloc[:,0:25].drop(columns=['symbol', 'NextDayClose', 'NextDayTrend',	'PrevDayTrend', 'splits'])
    return stock

"""# Load data"""

tt = ['ADBE', 'AMT', 'PLD', 'VICI', 'SCHW', 'JPM', 'ATVI', 'CVS', 'BIO', 'JNJ', 'NVDA', 'NDAQ', 'IBM', 'SBUX', 'NFLX', 'EBAY', 'AMZN', 'GOOG', 'AAPL', 'MSFT', 'TSLA']
tt[0]

appended_data = []
for ele in tt:
    triker_t = str(ele)
    # print(triker)
    triker2 = ele.lower()
    # print(triker1)
    stock = pd.read_csv(PATH + triker2 + '_Simon/stocks_ts_' + triker_t + '_2021-1-4_2022-9-20.csv')
    stock = stock[stock['date']>= '2021-01-01']
    stock = stock[stock['date']<= '2022-09-20']
    # store DataFrame in list
    appended_data.append(stock)
# see pd.concat documentation for more info
appended_data = pd.concat(appended_data)
appended_data = appended_data.drop(['NextDayTrend',	'PrevDayTrend',	'dividends',	'splits'], axis = 1)
# write DataFrame to an excel sheet
appended_data

stock = appended_data

stock['Diff'] = stock['close'] - stock['PrevDayClose']
stock['Diff'] = stock['Diff'].apply(lambda x: 1 if x > 0 else 0)
stock.head(3)

# drop NA
stock = stock.dropna()
stock = stock.drop(columns=['PrevDayClose'])
stock

stock.iloc[:,2:19]



scaler = MinMaxScaler()
stock.iloc[:,2:19] = scaler.fit_transform(stock.iloc[:,2:19])
stock = stock.sort_values(by=["symbol", "date"])
stock

"""# Bidiractional-LSTM"""


lstm_model = Sequential();
lstm_model.add(LSTM(500, return_sequences=True, input_shape=(None,stock.shape[1]-3)));
lstm_model.add(Bidirectional(LSTM(128, dropout=0.8)));
lstm_model.add(Dense(1, activation='sigmoid'))

# lstm_model = Model(lstm_model.inputs, lstm_model.outputs, name="lstm_model")

lstm_model.summary()

plot_model(lstm_model)

# Assuming input_shape is the size of your input sequences: (sequence_length, features)
input_shape = (None, stock.shape[1] - 3)

# Encoder
encoder = Sequential()
encoder.add(Conv1D(filters=128, kernel_size=1, activation='relu', input_shape=input_shape))
encoder.add(MaxPooling1D(pool_size=1, strides=1))
encoder.add(LSTM(128, return_sequences=False))
encoder.add(Dropout(0.8))

# Repeat the encoded representation for each time step in the output sequence
encoder.add(RepeatVector(1))

# Decoder
decoder = Sequential()
# You can experiment with different Conv1D parameters in the decoder
decoder.add(Conv1D(filters=128, kernel_size=1, activation='relu', input_shape=(None, 128)))
decoder.add(MaxPooling1D(pool_size=1, strides=1))
decoder.add(LSTM(128, return_sequences=True))
decoder.add(Dropout(0.8))
decoder.add(TimeDistributed(Dense(1, activation='sigmoid')))

# Combine Encoder and Decoder
cnn_seq2seq_model = Sequential([encoder, decoder])

# Compile the model with an appropriate optimizer and loss for your task
# model.compile(optimizer='your_optimizer', loss='your_loss')

# Display the model summary
cnn_seq2seq_model.summary()

"""#Attention Model

If your task benefits from capturing dependencies between different parts of the input sequence, the attention model might be more appropriate. If a simple repetition of the encoder output works well for your task, the original model might be sufficient.
"""



# Assuming input_shape is the size of your input sequences: (sequence_length, features)
input_shape = (None, stock.shape[1] - 3)

# Define input layer
inputs = Input(shape=input_shape)

# Encoder
encoder = Conv1D(filters=128, kernel_size=1, activation='relu')(inputs)
encoder = MaxPooling1D(pool_size=1, strides=1)(encoder)
encoder = LSTM(128, return_sequences=True)(encoder)
encoder = Dropout(0.8)(encoder)

# Attention Mechanism
attention = Attention()([encoder, encoder])

# Combine attention output with the encoder output
merged = tf.keras.layers.Concatenate(axis=-1)([encoder, attention])

# Decoder
decoder = Conv1D(filters=128, kernel_size=1, activation='relu')(merged)
decoder = MaxPooling1D(pool_size=1, strides=1)(decoder)
decoder = LSTM(128, return_sequences=True)(decoder)
decoder = Dropout(0.8)(decoder)
decoder = TimeDistributed(Dense(1, activation='sigmoid'))(decoder)

# Combine Encoder and Decoder
attention_model = Model(inputs=inputs, outputs=decoder)

# Compile the model with an appropriate optimizer and loss for your task
# model.compile(optimizer='your_optimizer', loss='your_loss')

# Display the model summary
attention_model.summary()

"""#Dilated CNN seq2seq

In summary, the kernel size defines the immediate neighborhood considered by the filter, while the dilation rate influences the spacing between the weights of the filter, affecting the receptive field and the model's ability to capture information from a broader context.
"""

# Assuming input_shape is the size of your input sequences: (sequence_length, features)
input_shape = (None, stock.shape[1] - 3)

# Encoder
encoder = Sequential()
encoder.add(Conv1D(filters=128, kernel_size=1, activation='relu', input_shape=input_shape))
encoder.add(MaxPooling1D(pool_size=1, strides=1))
encoder.add(LSTM(128, return_sequences=False))
encoder.add(Dropout(0.8))

# Repeat the encoded representation for each time step in the output sequence
encoder.add(RepeatVector(1))

# Decoder
decoder = Sequential()
# Replace Conv1D with Dilated Conv1D in the decoder
decoder.add(Conv1D(filters=128, kernel_size=1, activation='relu', dilation_rate=2, input_shape=(None, 128)))
decoder.add(MaxPooling1D(pool_size=1, strides=1))
decoder.add(LSTM(128, return_sequences=True))
decoder.add(Dropout(0.8))
decoder.add(TimeDistributed(Dense(1, activation='sigmoid')))

# Combine Encoder and Decoder
Dilated_cnn_seq2seq_model = Sequential([encoder, decoder])

# Compile the model with an appropriate optimizer and loss for your task
# model.compile(optimizer='your_optimizer', loss='your_loss')

# Display the model summary
Dilated_cnn_seq2seq_model.summary()

"""#CNN-LSTM"""


# Create a Sequential model
CNN_LSTM_model = Sequential()

# One-dimensional convolutional layer
CNN_LSTM_model.add(Conv1D(filters=32, kernel_size=1, strides=1, activation='relu', input_shape=(None, 17)))

# LSTM layer with 128 units and tanh activation function
CNN_LSTM_model.add(LSTM(128, activation='tanh', return_sequences=False))

# Batch normalization layer
CNN_LSTM_model.add(BatchNormalization())

# Dropout layer with a rate of 0.2
CNN_LSTM_model.add(Dropout(0.2))

# Dense layer with a prediction window size
CNN_LSTM_model.add(Dense(1, activation='sigmoid'))  # Replace 'your_prediction_window_size' with the desired size

# Print the model summary
CNN_LSTM_model.summary()

"""#GRU-CNN"""


# Create a Sequential model
GRU_CNN_model = Sequential()

# GRU layer with 128 units and tanh activation function
GRU_CNN_model.add(GRU(128, activation='tanh', return_sequences=True, input_shape=(None, stock.shape[1]-3)))

# One-dimensional convolutional layer with 32 filters of size 3 and a stride of 1
GRU_CNN_model.add(Conv1D(filters=32, kernel_size=1, strides=1, activation='relu'))

# One-dimensional global max-pooling layer
GRU_CNN_model.add(GlobalMaxPooling1D())

# Batch normalization layer
GRU_CNN_model.add(BatchNormalization())

# Dense layer with 10 units and ReLU activation
GRU_CNN_model.add(Dense(units=10, activation='relu'))

# Dropout layer with a rate of 0.2
GRU_CNN_model.add(Dropout(0.2))

# Dense layer with a prediction window size
GRU_CNN_model.add(Dense(1, activation='sigmoid'))  # Replace 'your_prediction_window_size' with the desired size

# Print the model summary
GRU_CNN_model.summary()



"""# Model Compile"""

# optimizer and metric
lstm_model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"]) # default lr = 0.001
cnn_seq2seq_model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"]) # default lr = 0.001
attention_model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"]) # default lr = 0.001
Dilated_cnn_seq2seq_model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"]) # default lr = 0.001
CNN_LSTM_model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"]) # default lr = 0.001
GRU_CNN_model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"]) # default lr = 0.001
# n_epochs = 10

# train_data.element_spec

"""# Early stopping and model saving"""

# checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True, verbose=1)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1) # patience try 5 or 10

# Save and Load model
checkpoint_filepath = '/content/drive/MyDrive/aapl_stock_early_fusion_tuning.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

"""# Exponential Scheduling"""

def exponential_decay_fn(epoch):
    return 0.01 * 0.1**(epoch / 12.5)

def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1**(epoch / s)
    return exponential_decay_fn

exponential_decay_fn = exponential_decay(lr0=0.01, s=12.5)


# callbacks=[lr_scheduler]
lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)


"""
# For Uptrend or Downtrend"""

stock['symbol'] = stock['symbol'].apply(str.upper)
stock = stock[stock['symbol'] == triker1]
# rslt_df = dataframe[dataframe['Stream'].isin(options)]

def startday(trend):
  if trend == 'downtrend':
    data = stock[stock['date']>= '2022-01-01']
  elif trend == 'uptrend':
    data = stock[stock['date']>= '2021-07-01']
    data = data[data['date']< '2022-01-01']
  return data

data_2022 = startday(trend)
data_2022

def mulday(trend):
  if trend == 'downtrend':
    Full_data = stock
  elif trend == 'uptrend':
    Full_data = stock[stock['date']< '2022-01-01']
  return Full_data

Full_data = mulday(trend)
Full_data

data_tail205 = Full_data.tail(len(data_2022)+22)
data_tail205

models = [lstm_model, cnn_seq2seq_model, attention_model, Dilated_cnn_seq2seq_model, CNN_LSTM_model, GRU_CNN_model]
names = ['lstm_model', 'cnn_seq2seq_model', 'attention_model', 'Dilated_cnn_seq2seq_model', 'CNN_LSTM_model', 'GRU_CNN_model']

for name in names:
  print(name)

# Accumulative price from the beggining of the day and evaluate to traning set

# Create an empty DataFrame to store the results
result_table = pd.DataFrame()

for model, name in zip(models, names):
  li = []
  for i in range(0,len(data_tail205)-22):
    train = data_tail205.iloc[0:21+i]
    feature_traindata = train.iloc[:,3:20].to_numpy()
    label_train = train.Diff.values

    val = data_tail205.iloc[0:21+i]
    feature_valdata = val.iloc[:,3:20].to_numpy()
    label_val = val.Diff.values

    test = data_tail205.iloc[21+i:23+i]
    feature_testdata = test.iloc[:,3:20].to_numpy()
    label_test = test.Diff.values

    time_steps = 20+i
    batch_size = 1

    train_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(
        data = feature_traindata,
        targets = label_train,
        length = time_steps,
        sampling_rate=1,
        stride=1,
        start_index=0,
        end_index=None,
        shuffle=False,
        reverse=False,
        batch_size=batch_size
    )

    val_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(
        data = feature_valdata,
        targets = label_val,
        length = time_steps,
        sampling_rate=1,
        stride=1,
        start_index=0,
        end_index=None,
        shuffle=False,
        reverse=False,
        batch_size=batch_size
    )

    test_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(
        data = feature_testdata,
        targets = label_test,
        length = 1,
        sampling_rate=1,
        stride=1,
        start_index=0,
        end_index=None,
        shuffle=False,
        reverse=False,
        batch_size=batch_size
    )

    ###############################################################################################################
    ### CREATE TRAIN FEATURES ###
    #################################################################################################################

    stock_lstm_train_data = tf.data.Dataset.from_generator(
        lambda: train_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=([batch_size, time_steps, feature_traindata.shape[1]], [batch_size,]) # based on batch_size change
      )

    stock_lstm_val_data = tf.data.Dataset.from_generator(
      lambda: val_generator,
      output_types=(tf.float32, tf.float32),
      output_shapes=([batch_size, time_steps, feature_valdata.shape[1]], [batch_size,]) # based on batch_size change
    )

    stock_lstm_test_data = tf.data.Dataset.from_generator(
        lambda: test_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=([batch_size, 1, feature_testdata.shape[1]], [batch_size,]) # based on batch_size change
      )

    train_data = stock_lstm_train_data
    val_data = stock_lstm_val_data
    test_data = stock_lstm_test_data

    # Train the model
    earlyfs_history = model.fit(train_data,
                                      epochs=fepochs,
                                      callbacks=[earlystopping],
                                      validation_data = val_data
                                      # verbose=1
    )

    eval_results = model.predict(test_data)
    print(eval_results)
    li.append(eval_results)
    print(len(data_tail205)-22-i)

  # Add the results to the DataFrame with a column name indicating the model
  result_table[name] = li

# Display the result table
print(result_table)

# Assume df is your existing DataFrame containing prediction results

# Function to convert the entire array to 1 if any value is above 0.5, else 0
def binary_conversion(arr):
    return np.array([1 if val > 0.5 else 0 for val in arr])

# Apply the binary conversion function to each column in the DataFrame
baseline_model = result_table.apply(lambda col: col.apply(binary_conversion))

# Display the modified DataFrame
baseline_model = baseline_model.applymap(lambda x: x[0])
baseline_model[''] = range(len(baseline_model))
baseline_model
baseline_model.to_csv('/content/drive/MyDrive/Base-Line-model/' + triker1 + '_' + trend + '_baseline_model.csv', index=False)
