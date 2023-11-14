import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KERAS_BACKEND']='tensorflow'
import tensorflow as tf
tf.config.list_physical_devices('GPU')
tf.debugging.set_log_device_placement(True)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
import nltk

# from google.colab import drive
# drive.mount('/content/drive')

"""# Load Multimodal data"""

triker1 = 'CVS'
# triker1 = sys.argv[1]
triker = triker1.lower()


trend = 'downtrend'
# trend = sys.argv[2]
# trend = 'Uptrend'
# tmax_trials = 20
# tepochs = 15
# fepochs = 100
# startdate = '2022-01-01' # for downtrend
# startdate = '2021-07-01'# for Uptrend

tmax_trials = 1
tepochs = 1
fepochs = 1
# startday = '2022-07-01'

# PATH = '/home/pinyu/stock/stock_data/'
# path = '/home/pinyu/stock/'

PATH = 'C:/Users/Simon/stock/stock_data/'
path = 'C:/Users/Simon/stock/'

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

stock = se(triker)

print(stock)

stock['Diff'] = stock['close'] - stock['PrevDayClose']
stock['Diff'] = stock['Diff'].apply(lambda x: 1 if x > 0 else 0)
stock.head(3)

news = pd.read_csv(PATH + triker + '_Simon/Single_newsheadline_day_'+triker+'.csv')
news = news[['date', 'Titles']]
news

Single_newsheadline_day_stock = news.merge(stock, on = 'date')
Single_newsheadline_day_stock

Multi_data = Single_newsheadline_day_stock

# drop NA
Multi_data = Multi_data.dropna()
Multi_data = Multi_data.drop(columns=['PrevDayClose'])
Multi_data

Multi_data.iloc[:,2:21]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
Multi_data.iloc[:,2:21] = scaler.fit_transform(Multi_data.iloc[:,2:21])
Multi_data

"""# Text Cleaning

Generating word frequencies
"""

def gen_freq(text):
    #will store all the words in list
    words_list = []
    
    #Loop over all the words and extract word from list
    for word in text.split():
        words_list.extend(word)
        
    #Generate word frequencies using value counts in word_list
    word_freq = pd.Series(words_list).value_counts()
    
    #print top 100 words
    word_freq[:100]
    
    return word_freq

freq = gen_freq(Multi_data.Titles.str)
freq

"""Removing Stopwords"""

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_word_list = stopwords.words('english')

from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.tokenize.toktok import ToktokTokenizer

#Tokenization of text
tokenizer=ToktokTokenizer()

#removing the stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stop_word_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

#Apply function on review column
Multi_data['Titles']= Multi_data['Titles'].apply(remove_stopwords)

"""process of clearing punctuation marks in data
cleaning unnecessary marks in data. </p> </li>
capitalization to lowercase. </p> </li>
cleaning extra spaces. </p> </li>
removal of stopwords in sentences. </p> </li>
"""

import re
#clearing punctuation & unnecessary marks
Multi_data['Titles']= Multi_data['Titles'].apply(lambda x: re.sub("[,’\.!?:-;()–']", '', x))
# Multi_data['Titles']= Multi_data['Titles'].apply(lambda x: re.sub('[^a-zA-Z"]', ' ', x))

#capitalization to lowercase
Multi_data['Titles']= Multi_data['Titles'].apply(lambda x: x.lower())

#cleaning extra spaces
Multi_data['Titles']= Multi_data['Titles'].apply(lambda x: x.strip())

#Removing the square brackets
# def remove_between_square_brackets(text):
#     return re.sub('\[[^]]*\]', '', text)

#Apply function on review column

# Multi_data['Titles']= Multi_data['Titles'].apply(remove_between_square_brackets)

"""# Text Pre-processing for BERT"""

# pip install -q transformers

from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures

# pip install bert-for-tf2

#!pip3 install tensorflow_hub

import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras.models import Model 
import bert

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

max_length = 54

# convert_examples_to_tf_dataset: tokenize the InputExample objects, then create the required input format with the tokenized objects 
# finally, create an input dataset that we can feed to the model.
def convert_examples_to_tf_dataset(examples, tokenizer, max_length=max_length):
    features = [] # -> will hold InputFeatures to be converted later

    for e in examples:
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length, # max_length
            return_token_type_ids=True,
            return_attention_mask=True,
            padding="max_length", # pad_to_max_length=True
            truncation=True
        )
        
        #print(len(input_dict))

        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
            input_dict["token_type_ids"], input_dict['attention_mask'])

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label
            )
        )
    def gen():
        for f in features:
            yield (
                  f.input_ids,
                  f.attention_mask,
                  f.token_type_ids
                ,
                # f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        (tf.int32, tf.int32, tf.int32),
        (
              tf.TensorShape([max_length]),
              tf.TensorShape([max_length]),
              tf.TensorShape([max_length])
            ,
            # tf.TensorShape([]),
        ),
    )

"""# Early fusion Multimodal

LSTM model
"""

# from keras.models import Sequential
# from keras.layers import Dense, LSTM, Dropout
# from keras.layers import LSTM, Flatten

# lstm_model = Sequential();
# lstm_model.add(LSTM(500, return_sequences=True, input_shape=(None,17)));
# lstm_model.add(LSTM(100, dropout=0.1));
# # model.add(Dropout(0.1))
# lstm_model.add(Dense(72))
# # lstm_model.add(Dense(1, activation='sigmoid'))


# lstm_model = Model(lstm_model.inputs, lstm_model.outputs)

from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout
from keras.layers import LSTM

lstm_model = Sequential();
lstm_model.add(LSTM(500, return_sequences=True, input_shape=(None,Multi_data.shape[1]-3)));
lstm_model.add(tf.keras.layers.Normalization());
lstm_model.add(LSTM(450, dropout=0.1));
lstm_model.add(tf.keras.layers.BatchNormalization());
# lstm_model.add(tf.keras.layers.Normalization());
# lstm_model.add(Dense(300));
# lstm_model.add(tf.keras.layers.BatchNormalization());
# lstm_model.add(tf.keras.layers.Normalization());
# lstm_model.add(Dense(350));
# lstm_model.add(tf.keras.layers.BatchNormalization());
# lstm_model.add(tf.keras.layers.Normalization());
# lstm_model.add(Dense(250));
# lstm_model.add(tf.keras.layers.BatchNormalization());
# model.add(Dropout(0.1))
lstm_model.add(Dense(72))

lstm_model = Model(lstm_model.inputs, lstm_model.outputs)

lstm_model.summary()

from tensorflow.keras.utils import plot_model
# plot_model(lstm_model)

"""Text model"""

max_length = 54
input_word_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32,
                                       name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32,
                                   name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32,
                                    name="segment_ids")
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=True)
pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
lstm_out = tf.keras.layers.Bidirectional(LSTM(max_length, name='LSTM'))(sequence_output) # Bidirectional LSTM instead of LSTM

out = tf.keras.layers.Normalization()(lstm_out)
out = tf.keras.layers.Dense(80, activation="relu")(out)
out = tf.keras.layers.Normalization()(out)
out = tf.keras.layers.Dense(36, activation="relu")(out)

#bert_model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])
bert_model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

# Keep the Bert layers trainable
for layer in bert_model.layers:
    layer.trainable = True

bert_model.summary()

# plot_model(bert_model)



"""multimodal"""

from tensorflow.keras import models
# Stacking early-fusion multimodal model
#nClasses = 2 # for multi-class

input_word_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="segment_ids")
lstm_input = tf.keras.layers.Input(shape=(None, Multi_data.shape[1]-3), dtype=tf.float32, name="Price")

lstm_side = lstm_model(lstm_input)
text_side = bert_model([input_word_ids, input_mask, segment_ids])
# Concatenate features from images and texts
merged = tf.keras.layers.Concatenate()([lstm_side, text_side])
merged = tf.keras.layers.Normalization()(merged)
merged = tf.keras.layers.Dense(40, activation="relu")(merged)
merged = tf.keras.layers.Normalization()(merged)
# merged = tf.keras.layers.Dropout(0.5)(merged)
merged = tf.keras.layers.Dense(60, activation="relu")(merged)
# merged = tf.keras.layers.Dense(100, activation="relu")(merged)
# merged = tf.keras.layers.Dropout(0.5)(merged)
output = tf.keras.layers.Dense(1, activation="sigmoid", name="class")(merged)

merge_model = models.Model([lstm_input, input_word_ids, input_mask, segment_ids], output)

merge_model.summary()

# plot_model(merge_model)

"""# Model Compile"""

#import mlflow.tensorflow
from tensorflow.keras.optimizers import Adam

# optimizer and metric
merge_model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"]) # default lr = 0.001
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

import keras

# callbacks=[lr_scheduler]
lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)

"""# Training model

# 10 days in training set 1 day for testing, grad search for hyper parameter tuning, evaluate in next year.

# Function for loop
"""

# convert_data_to_examples: accept train and test datasets and convert each row into an InputExample object.
def convert_train_to_examples(train, DATA_COLUMN, LABEL_COLUMN): 
  train_InputExamples = train.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this case
                                                          text_a = x[DATA_COLUMN], 
                                                          text_b = None, 
                                                          label = x[LABEL_COLUMN]), axis = 1)
  
  return train_InputExamples

# convert_data_to_examples: accept train and test datasets and convert each row into an InputExample object.
def convert_val_to_examples(val, DATA_COLUMN, LABEL_COLUMN): 
  val_InputExamples = val.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this case
                                                          text_a = x[DATA_COLUMN], 
                                                          text_b = None, 
                                                          label = x[LABEL_COLUMN]), axis = 1)
  
  return val_InputExamples

# convert_data_to_examples: accept train and test datasets and convert each row into an InputExample object.
def convert_test_to_examples(test, DATA_COLUMN, LABEL_COLUMN): 
  
  test_InputExamples = test.apply(lambda x: InputExample(guid=None,
                                                          text_a = x[DATA_COLUMN], 
                                                          text_b = None, 
                                                          label = x[LABEL_COLUMN]), axis = 1)  
  
  return test_InputExamples

DATA_COLUMN = 'Titles'
LABEL_COLUMN = 'Diff'

len(Multi_data)



"""# First model"""

training = Multi_data.iloc[0:11]
testing = Multi_data[11:13]
testing

# Accumulative price from the beggining of the day and evaluate to traning set
li = []
train = training
feature_traindata = train.iloc[:,3:20].to_numpy()
label_train = train.Diff.values
  
test = testing
feature_testdata = test.iloc[:,3:20].to_numpy()
label_test = test.Diff.values

time_steps = 10
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

stock_lstm_train_data = tf.data.Dataset.from_generator(
    lambda: train_generator, 
    output_types=(tf.float32, tf.float32), 
    output_shapes=([batch_size, time_steps, feature_traindata.shape[1]], [batch_size,]) # based on batch_size change
  )

stock_lstm_test_data = tf.data.Dataset.from_generator(
    lambda: test_generator, 
    output_types=(tf.float32, tf.float32), 
    output_shapes=([batch_size, 1, feature_testdata.shape[1]], [batch_size,]) # based on batch_size change
  )

  # call above function
train_InputExamples = convert_train_to_examples(train, DATA_COLUMN, LABEL_COLUMN)
  # call above function
test_InputExamples = convert_test_to_examples(test, DATA_COLUMN, LABEL_COLUMN)

stock_train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
stock_train_data = stock_train_data.batch(batch_size, drop_remainder=True)

stock_test_data = convert_examples_to_tf_dataset(list(test_InputExamples), tokenizer)
stock_test_data = stock_test_data.batch(batch_size, drop_remainder=True)

  # price test batch drop remainder
stock_lstm_test_data = stock_lstm_test_data.unbatch()
stock_lstm_test_data = stock_lstm_test_data.batch(batch_size, drop_remainder=True)


  # merged train data
try_data = tf.data.Dataset.zip((stock_lstm_train_data, stock_train_data))
try_data.element_spec
train_data = try_data.map(lambda x, y: ((x[0], y[0], y[1], y[2]), x[1]))
print(train_data.element_spec)

  # merged validation data
try_data2 = tf.data.Dataset.zip((stock_lstm_test_data, stock_test_data))
try_data2.element_spec
test_data = try_data2.map(lambda x, y: ((x[0], y[0], y[1], y[2]), x[1]))
print(test_data.element_spec)

  # Train the model 
earlyfs_history = merge_model.fit(train_data,
                                    epochs=2, # 20
                                    callbacks=[lr_scheduler],
                                    # verbose=1
  )

eval_results = merge_model.predict(test_data)
print(eval_results)
li.append(eval_results)

"""# Tuning data"""

train = Multi_data[0:round(427/2)]
feature_traindata = train.iloc[:,3:20].to_numpy()
label_train = train.Diff.values
  
test = Multi_data.tail(427-round(427/2))
feature_testdata = test.iloc[:,3:20].to_numpy()
label_test = test.Diff.values

time_steps = 10
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

stock_lstm_train_data = tf.data.Dataset.from_generator(
    lambda: train_generator, 
    output_types=(tf.float32, tf.float32), 
    output_shapes=([batch_size, time_steps, feature_traindata.shape[1]], [batch_size,]) # based on batch_size change
  )

stock_lstm_test_data = tf.data.Dataset.from_generator(
    lambda: test_generator, 
    output_types=(tf.float32, tf.float32), 
    output_shapes=([batch_size, 1, feature_testdata.shape[1]], [batch_size,]) # based on batch_size change
  )

  # call above function
train_InputExamples = convert_train_to_examples(train, DATA_COLUMN, LABEL_COLUMN)
  # call above function
test_InputExamples = convert_test_to_examples(test, DATA_COLUMN, LABEL_COLUMN)

stock_train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
stock_train_data = stock_train_data.batch(batch_size, drop_remainder=True)

stock_test_data = convert_examples_to_tf_dataset(list(test_InputExamples), tokenizer)
stock_test_data = stock_test_data.batch(batch_size, drop_remainder=True)

  # price test batch drop remainder
stock_lstm_test_data = stock_lstm_test_data.unbatch()
stock_lstm_test_data = stock_lstm_test_data.batch(batch_size, drop_remainder=True)


  # merged train data
try_data = tf.data.Dataset.zip((stock_lstm_train_data, stock_train_data))
try_data.element_spec
train_data = try_data.map(lambda x, y: ((x[0], y[0], y[1], y[2]), x[1]))
print(train_data.element_spec)

  # merged validation data
try_data2 = tf.data.Dataset.zip((stock_lstm_test_data, stock_test_data))
try_data2.element_spec
test_data = try_data2.map(lambda x, y: ((x[0], y[0], y[1], y[2]), x[1]))
print(test_data.element_spec)

"""# Tuning test"""

# !pip install keras-tuner --upgrade

import keras_tuner
from tensorflow import keras
from keras import backend as K
from tensorflow.keras import layers, losses
import numpy as np
import matplotlib.pyplot as plt
import os

class MyHyperModel(keras_tuner.HyperModel) :
    def build(self, hp, classes=1) : 
        # define model
        input_word_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="segment_ids")
        lstm_input = tf.keras.layers.Input(shape=(None, len(Multi_data.axes[1])-3), dtype=tf.float32, name="Price")

        lstm_side = lstm_model(lstm_input)
        text_side = bert_model([input_word_ids, input_mask, segment_ids])
        # Concatenate features from images and texts
        merged = tf.keras.layers.Concatenate()([lstm_side, text_side])

        # Number of Layers is up to tuning
        for i in range(hp.Int("num_layer", min_value=2, max_value=8, step=2)) :
            # Tune hyperparams of each conv layer separately by using f"...{i}"

            merged = (layers.Normalization())(merged)

            merged = (layers.Dense(hp.Int(name=f"Dense_{i}", min_value=40, max_value=160, step=20),
                                    activation=hp.Choice('layer_opt',['relu'])))(merged)

        
        # Last layer
        output = (layers.Dense(classes, activation='sigmoid'))(merged)

        model = models.Model([lstm_input, input_word_ids, input_mask, segment_ids], output)
        
        # Picking an opimizer and a loss function
        model.compile(optimizer=hp.Choice('optim',['adam']),
                      loss="binary_crossentropy",
                      metrics = ['accuracy'])
        
        # A way to optimize the learning rate while also trying different optimizers
        learning_rate = hp.Choice('lr', [0.0001])
        K.set_value(model.optimizer.learning_rate, learning_rate)
        
        return model
    
    
    def fit(self, hp, model,x, *args, **kwargs) :
        
        return model.fit( x, 
                         *args,
                         shuffle=hp.Boolean("shuffle"),
                         **kwargs)

classes = 1
BATCH_SIZE = 1
hp = keras_tuner.HyperParameters()
hypermodel = MyHyperModel()
model = hypermodel.build(hp, classes)
# hypermodel.fit(hp, model, np.random.rand([lstm_input, input_word_ids, input_mask, segment_ids]), np.random.rand(BATCH_SIZE, classes))

tuner = keras_tuner.BayesianOptimization(
                        hypermodel=MyHyperModel(),
                        objective = "val_accuracy",
                        max_trials = tmax_trials, #max candidates to test
                        overwrite=True,
                        directory='BO_search_dir',
                        project_name='multimodal_lstm_bert')

model.summary()

tuner.search(x=train_data,
             validation_data = test_data,
             epochs=tepochs,
             callbacks=[earlystopping])

tuner.results_summary(1)

tuner.get_best_hyperparameters(5)

# Get the top 2 hyperparameters.
best_hps = tuner.get_best_hyperparameters(1)
# Build the model with the best hp.
h_model = MyHyperModel()
model = h_model.build(best_hps[0])

model.summary()

# history = model.fit(x=train_data, 
#                     validation_data=test_data, 
#                     epochs=100,
#                     callbacks=[earlystopping])

"""
# For 2021-7 or 2022-9"""

def startday(trend):
  if trend == 'downtrend':
    data = Multi_data[Multi_data['date']>= '2022-01-01']
  elif trend == 'uptrend':
    data = Multi_data[Multi_data['date']>= '2021-07-01']
    data = data[data['date']< '2022-01-01']
  print(data)
  return data

data_2022 = startday(trend)
data_2022

def mulday(trend):
  if trend == 'downtrend':
    Full_data = Multi_data
  elif trend == 'uptrend':
    Full_data = Multi_data[Multi_data['date']< '2022-01-01']
  return Full_data

Full_data = mulday(trend)
Full_data

data_tail205 = Full_data.tail(len(data_2022)+22)
data_tail205.head(22)

# Accumulative price from the beggining of the day and evaluate to traning set
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

  # call above function
  train_InputExamples = convert_train_to_examples(train, DATA_COLUMN, LABEL_COLUMN)

  val_InputExamples = convert_val_to_examples(val, DATA_COLUMN, LABEL_COLUMN)
  # call above function
  test_InputExamples = convert_test_to_examples(test, DATA_COLUMN, LABEL_COLUMN)

  stock_train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
  stock_train_data = stock_train_data.batch(batch_size, drop_remainder=True)

  stock_val_data = convert_examples_to_tf_dataset(list(val_InputExamples), tokenizer)
  stock_val_data = stock_val_data.batch(batch_size, drop_remainder=True)

  stock_test_data = convert_examples_to_tf_dataset(list(test_InputExamples), tokenizer)
  stock_test_data = stock_test_data.batch(batch_size, drop_remainder=True)

  # price test batch drop remainder
  stock_lstm_test_data = stock_lstm_test_data.unbatch()
  stock_lstm_test_data = stock_lstm_test_data.batch(batch_size, drop_remainder=True)


  # merged train data
  try_data = tf.data.Dataset.zip((stock_lstm_train_data, stock_train_data))
  try_data.element_spec
  train_data = try_data.map(lambda x, y: ((x[0], y[0], y[1], y[2]), x[1]))
  # print(train_data.element_spec)

  try_valdata = tf.data.Dataset.zip((stock_lstm_val_data, stock_val_data))
  try_valdata.element_spec
  val_data = try_valdata.map(lambda x, y: ((x[0], y[0], y[1], y[2]), x[1]))
  # print(val_data.element_spec)

  # merged validation data
  try_data2 = tf.data.Dataset.zip((stock_lstm_test_data, stock_test_data))
  try_data2.element_spec
  test_data = try_data2.map(lambda x, y: ((x[0], y[0], y[1], y[2]), x[1]))
  # print(test_data.element_spec)

  # Train the model 
  earlyfs_history = merge_model.fit(train_data,
                                    epochs=fepochs,
                                    callbacks=[earlystopping],
                                    validation_data = val_data
                                    # verbose=1
  )

  eval_results = merge_model.predict(test_data)
  print(eval_results)
  li.append(eval_results)
  print(len(data_tail205)-22-i)

li = np.array(li)
y_pred = np.where(li > 0.5, 1, 0)

true_y = data_tail205.tail(len(y_pred)).Diff.values
print(true_y)

y_PRED = []
for i in range(0, len(li)):
  k = y_pred[i][0][0]
  y_PRED.append(k)
y_PRED = np.array(y_PRED)
print(y_PRED)
from sklearn.metrics import confusion_matrix
confusion_matrix(true_y, y_PRED)

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# fpr, tpr, thresholds = roc_curve(true_y, y_PRED)
# roc_auc = auc(fpr, tpr)

# # Plot ROC curve
# plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
# plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
# # plt.xlim([0.0, 1.0])
# # plt.ylim([0.0, 1.0])
# plt.xlabel('False Positive Rate or (1 - Specifity)')
# plt.ylabel('True Positive Rate or (Sensitivity)')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")

from sklearn.metrics import classification_report
print('Results on the test set:')
cr = classification_report(true_y, y_PRED, output_dict=True)
print(cr)

acc = str(round(cr['accuracy']*100, 2))
print(acc)

crdf = pd.DataFrame(cr).transpose()
print(crdf)

crdf.to_csv(path + 'stock_results/stock_info/classreport_'+triker + '_' + trend + '_' + acc + '.csv')

conf_matrix = confusion_matrix(true_y, y_PRED)

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.savefig(path + 'stock_results/stock_info/'+triker + '_' + trend +'_' + acc + 'confusionmatrix.png')
plt.show()

obs = np.arange(0,len(y_PRED))
print(len(obs))

date_2 = data_2022['date']
print(len(date_2))

data = {'': obs,
        'y_PRED': y_PRED,
        'ground_truth': true_y,
        'dates': date_2}

df = pd.DataFrame(data)
print(df)

df.to_csv(path + 'stock_results/stock_info/new_prediction_' + triker + '_' + trend + '_' + acc + '.csv', index=False)

"""# Load model"""

model_structure = merge_model.to_json()
with open(path + 'stock_results/stock_info/' + trend + '_' + triker + acc + '.json', 'w') as json_file:
    json_file.write(model_structure)
merge_model.save_weights(path + 'stock_results/stock_info/' + trend + '_' + triker + acc + '.h5')
print('Model saved.')

# from keras.models import model_from_json
# with open(path + 'stock_results/stock_info/' + triker + acc + '.json', 'r') as json_file:
#     json_string = json_file.read()
# model = model_from_json(json_string)

# model.load_weights('C:/Users/Simon/stock/stock_results/stock_info/' + triker + acc + '.h5')
