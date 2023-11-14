import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KERAS_BACKEND']='tensorflow'
import tensorflow as tf
tf.config.list_physical_devices('GPU')
tf.debugging.set_log_device_placement(True)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import math
from datetime import date
import datetime
from datetime import datetime,timedelta
import random
import time
import matplotlib.pyplot as plt
import pandas as pd
from pandas._libs import index
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
import sklearn.metrics as metrics
from utility_methods import *
##########################################################################################
# Parametrization 
##########################################################################################
forecastingFlag = False
epochsFirstIteration = 600
epochsFollowingIterations = 200
numDaysToTest = 10
minTimeStepsWindow = 365
learningRateInitial = 0.001
learningRateFollowingIterations = 0.0001
ticker = 'aapl'
text_threshold = 0.4
train_from_files = True
slicingFV = 15
slicingFactor = slicingFV
noOfColumns = 18

# SERVER
#PATH = '/home/rcorizzo/git/stocks/'
# LOCAL
PATH = '/Users/robertocorizzo/Desktop/Stocks/'

#drive.mount('/content/drive')
time_range_low = date(2021,1,4)
time_range_high = date(2022,9,20)
##########################################################################################
time_range_low = str(time_range_low.year) +"/"+str(time_range_low.month)+"/"+str(time_range_low.day)
time_range_high = str(time_range_high.year) +"/"+str(time_range_high.month)+"/"+str(time_range_high.day)

# textPolarity contains data for news headlines
#textPolarity = pd.read_csv(os.path.join(PATH +  ticker, 'textPolarity.csv'), encoding = 'utf-8-sig')
textPolarity = pd.read_csv(PATH + '/FULL_DATASET/' + ticker + '/textPolarity.csv')
textPolarity.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)

# dfObserve contains stocks
dfObserve = pd.read_csv(PATH + '/FULL_DATASET/' + ticker + '/stocks_ts_'+ ticker + '_' + time_range_low.replace('/','-') + '_' + time_range_high.replace('/','-') + '.csv',encoding = 'utf-8-sig')
textPolarity.columns
dfObserve.head()

print(dfObserve.index)
testDf = dfObserve
testDf = testDf.set_index(['date','symbol'],append=False)
dfObserve = dfObserve.set_index(['date','symbol'],append=False)

### Scaling
scaled = dfObserve.copy()
dfNextDayClose = dfObserve["NextDayClose"]
scaledData = scalingFunc(dfObserve)
enc1 = scaledData[1]
enc2 = scaledData[2]
scaledDataFrame = scaledData[0]
print(scaledDataFrame.index)

testData = testTrainData(scaledDataFrame,noOfColumns)[0]
allLabels = testTrainData(scaledDataFrame,noOfColumns)[1]

gtLabelsAll = scaledDataFrame.iloc[33:]['PrevDayTrend']

# RUNNER: ALTERNATIVE WAY: INSTEAD OF RUNNING ALL DATES IN FILES, GENERATE RANDOM DATES : CHANGE HERE NUMBER OF DAYS SUBJECT TO EXPERIMENTAL EVALUATION
# from datetime import datetime
##Generate new test dates
# newDates = genNewXTest(numDaysToTest,ticker)
# newDates
##Run with static dates & data
# staticDates = genNewXTestStatic(testDates,ticker)
# newDates = staticDates

#newDates = [datetime.strptime(date, '%m/%d/%y').date() for date in textPolarity['Date']]
newDates = [datetime.strptime(date, '%m/%d/%y').date() for date in textPolarity['Date']]
newDates

print(testData.index)

X_test = []
Y_test = []

for i in range(len(newDates)):
  print(newDates[i])
  tempVal = getXtest(newDates[i],slicingFV,testData,ticker)
  X_test.append(tempVal[0])
  Y_test.append(tempVal[1])

prevDayNorm = scaledDataFrame['PrevDayClose']
#prevDayNorm = scaledDataFrame['PrevDayClose']

print(np.shape(X_test))
print(np.shape(Y_test))
print('Completed data prep \nBeginning Model Training & Predictions')
dfPrevDayClose = dfObserve["PrevDayClose"]
outputDf = runAllMethodsOnAllDates(dfObserve,dfPrevDayClose,enc1,enc2,prevDayNorm,X_test,Y_test,newDates,forecastingFlag,slicingFV,ticker,testData,scaled,allLabels,gtLabelsAll,epochsFirstIteration,learningRateInitial,minTimeStepsWindow,epochsFollowingIterations)
actualDf = outputDf[0]
print('completed model')

calculations = outputDf[1]
completedOutput = outputDf
newOutDf = outputDf[0]
newOutDf

if forecastingFlag:
  # Forecasting metrics
  print("Denormalized LSTM")
  print(calculations[0]) # mae, mse, rmse, r2, mape
  print("Normalized LSTM")
  print(calculations[1]) 
  print("Denormalized ARIMA")
  print(calculations[2]) # mae, mse, rmse, r2, mape
  print("Normalized ARIMA")
  print(calculations[3])

# For forecasting: see how close predictions are to ground truth 

if forecastingFlag:
  newOutDf[['actual','predicted','deNorm_ArimaPred']]

trueTrends = dfObserve['PrevDayTrend']
#testNew = [completedDf,calculations,completedOutput,dfObserve['PrevDayTrend']]
trends = dfObserve['PrevDayTrend']
trends

# We don't scrape text headlines, we load them from file 
tn = textPolarity.copy()
tn = tn.set_index('Date')
print(tn.columns)
print(tn.index)

#We concatenate NLP predictions with ARIMA and LSTM predictions returned above
print(outputDf[0].columns)
print(outputDf[0].index)
outputDfCopy = outputDf[0].copy()
outputDfCopy.reset_index(inplace=True)
print(outputDfCopy.columns)

outputDfCopy.rename(columns={'index': 'Date'}, inplace=True)
print(f'OutputDfCopy.columns: {outputDfCopy.columns}')

outputDfCopy['Date'] = outputDfCopy['Date'].map(lambda x: datetime.strftime(x, '%-m/%-d/%Y'))
outputDfCopy = outputDfCopy.set_index("Date")
print(outputDfCopy.head())
print(tn.head())

# LIMIT ON NUMBER OF DAYS
#tn = tn[0:10]
tn = pd.concat((tn, outputDfCopy), axis='columns')
tn.head()
print(np.shape(tn))
tn
print(len(trends))
print(len(tn))
print(testDf.index)
stackingTraining = prepForStacked(testDf,tn,trends,forecastingFlag,text_threshold,ticker) # inputs: (dataframe), outputs: [(lstm,bert),tre]

# ----------------------------------------------------------------------
# Code to geneerate a stacking dataset with dates never used for testing
# Pick number of desired days and decomment this block to generate it
# Comment it to just evaluate stacking model on testing dates
# ----------------------------------------------------------------------
#stackModelTrain = pd.DataFrame(index=tn.index)
stackModelTrain = pd.DataFrame(index=stackingTraining[7])
stackModelTrain['LSTM Predictions'] = stackingTraining[3]
#stackModelTrain['PosOrNeg Predictions'] = stackingTraining[4]
stackModelTrain['Polarity Predictions'] = stackingTraining[5]
stackModelTrain['Arima Predictions'] = stackingTraining[6]
stackModelTrain['Ground Truth'] = stackingTraining[1]
stackModelTrain['Polarity Scores'] = stackingTraining[8]
stackModelTrain['GBTsPreds'] = stackingTraining[9]
stackModelTrain['predictionsProbLSTM'] = stackingTraining[10]
print(stackModelTrain)
stackModelTrain.to_csv('stackModelTrain.csv',encoding = 'utf-8-sig')

