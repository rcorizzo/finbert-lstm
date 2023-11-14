import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KERAS_BACKEND']='tensorflow'
import pandas as pd
import tensorflow as tf
tf.config.list_physical_devices('GPU')
tf.debugging.set_log_device_placement(True)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
from pandas.tseries.offsets import BDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from pmdarima.arima import auto_arima
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from datetime import date
import datetime
from datetime import datetime,timedelta
import sklearn.metrics as metrics
import numpy as np
##########################################################################################
def runAllMethodsOnAllDates(dfObserve,NextDayInfo,enc1,enc2,nextDayNorm,X_test,Y_test,newDates,forecastingFlag,slicingFV,ticker,testData,scaled,allLabels,gtLabelsAll,epochsFirstIteration,learningRateInitial,minTimeStepsWindow,epochsFollowingIterations):
  predictions = []
  predictionsProbLSTM = []
  predictionsNorm = []
  actual = []
  actualDenorm = []
  arimaPreds  = []
  deNormArimaPreds  = []
  NextDayInfoNorm = []
  GBTpredictions = []

  print('Beginning tpc')
  
  # LIMIT ON NUMBER OF DAYS
  #newDates = newDates[0:10]

  for i in range(len(newDates)):
  #for i in range(10):
    print(i)
    print(newDates[i])

    # LSTM PREDICTIONS
    
    if i==0:
      tData = trainingData([newDates[i]],slicingFV,testData,scaled,allLabels,gtLabelsAll,ticker,minTimeStepsWindow)
    else:
      startingPoint = newDates[i-1]
      tData = trainingData([newDates[i]],slicingFV,testData,scaled,allLabels,gtLabelsAll,ticker,minTimeStepsWindow,startingPoint)

    closingPrices = tData[2]

    print('Sanity check: y_train being fed to LSTM for prediction')
    print(tData[1])
    # Sanity check for ground truth being fed to ARIMA (stock prices up to previous day)
    denorm_ls1 = enc2.inverse_transform(tData[1].reshape(-1, 1))
    denorm_ls2 = enc1.inverse_transform(denorm_ls1)
    print(denorm_ls2)

    print(np.shape(tData[0]))
    print(np.shape(tData[1]))
    print(np.shape(tData[2]))
    print(np.shape(tData[3]))
    print(np.shape(X_test))
    print(np.shape(Y_test))
    print(Y_test[i])
    # Sanity check for y_test: should be current day
    denorm_ytest1 = enc2.inverse_transform(np.array(Y_test[i]).reshape(-1, 1))
    denorm_ytest2 = enc1.inverse_transform(denorm_ytest1)
    print(denorm_ytest2)
    #print("X TEST")
    #print(X_test[i]) # contains all features, indicators, etc

    #model = trainModel(tData[0],tData[1],X_test,np.array(Y_test))

    if (i==0):
      model = trainModel(tData[0],tData[1],epochsFirstIteration,learningRateInitial)
    else:
      model = trainModel(tData[0],tData[1],epochsFollowingIterations,learningRateInitial,model)

    test_Predict = model.predict(X_test[i])
    
    print('TEST PREDICT')
    print(test_Predict)


    # GBTs
    print(len(tData[3]))
    print('GBT Training') 
    gbts = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=10, random_state=0).fit(tData[3], tData[1])
    GBTpredictions.append(int(gbts.predict(X_test[i][0][-1].reshape(1, -1))[0]))

    # ARIMA
    print('ARIMA Training') 
    # Sanity check for ground truth being fed to ARIMA (stock prices up to previous day)

    print()
    print(closingPrices)
    print()
    
    # Training on normalized data
    Arima_model = auto_arima(closingPrices,start_p=1, d=None, start_q=1, max_p=2, max_d=2, max_q=2, start_P=1, D=None, start_Q=1, max_P=2, max_D=1, max_Q=2, max_order=2)
    predArima = pd.DataFrame(Arima_model.predict(1))
    #print(predArima)

    print(str(i) + ' Predictions Complete')

    denorm_gt1 = enc2.inverse_transform(closingPrices.reshape(-1, 1))
    denorm_gt2 = enc1.inverse_transform(denorm_gt1)

    ##undo normalization & append
    if (forecastingFlag==True):
      denorm_step_1 = enc2.inverse_transform(test_Predict)
      denorm_step_2 = enc1.inverse_transform(denorm_step_1)
      predictionsNorm.append(test_Predict[0][0])
      predictions.append(denorm_step_2[0][0])
    else:
      if np.isnan(test_Predict[0][0]):
        test_Predict[0][0] = 0.0
      predictionsNorm.append(test_Predict[0][0])
      predictions.append(round(test_Predict[0][0])) # Wrong but maybe unused in classification
      predictionsProbLSTM.append(test_Predict[0][0])


    ##get actual date info
    x = newDates[i]
    dateL = [int(x.strftime('%Y')),int(x.strftime('%m')),int(x.strftime('%d'))]
    
    # NextDayInfo.loc[date(dateL[0],dateL[1],dateL[2])]
    # actual.append(justOne.loc[date(dateL[0],dateL[1],dateL[2])]['close'][0])

    actual.append(dfObserve.loc[dfObserve.index.get_level_values('date') == datetime.strftime(newDates[i],'%Y-%m-%d')]['close'])
    #NextDayInfoNorm.append(nextDayNorm.loc[date(dateL[0],dateL[1],dateL[2])][0])
    NextDayInfoNorm.append(nextDayNorm.loc[datetime.strftime(date(dateL[0],dateL[1],dateL[2]),'%Y-%m-%d')][0])

    #denorm arima vals
    denorm_step_1 = enc2.inverse_transform(predArima)
    denorm_step_2_arima = enc1.inverse_transform(denorm_step_1)
    arimaPreds.append(predArima[0][0])
    deNormArimaPreds.append(denorm_step_2_arima[0][0])

  print('Predictions Complete')

  print(len(actual))

  outputDf = pd.DataFrame(index=newDates)
  outputDf['actual'] = actual
  outputDf['predicted'] = predictions
  outputDf['NextDayInfoNorm'] = NextDayInfoNorm
  outputDf['predictedNorm'] = predictionsNorm
  outputDf['arimaPred'] = arimaPreds
  outputDf['deNorm_ArimaPred'] = deNormArimaPreds
  outputDf['GBTsPreds'] = GBTpredictions
  outputDf['predictionsProbLSTM'] = predictionsProbLSTM

  calculationsDenorm = runCalculations(actual,predictions)
  calculationsNorm = runCalculations(NextDayInfoNorm,predictionsNorm) # needs normalized actual (NextDayInfo)

  calculationsDeNormArima = runCalculations(actual,deNormArimaPreds) 
  calculationsNormArima = runCalculations(NextDayInfoNorm,arimaPreds) # needs normalized actual for Arima?

  calculationsDenormGBTs = runCalculations(actual,GBTpredictions)
  calculationsNormGBTs = runCalculations(NextDayInfoNorm,GBTpredictions)

  calculations = [calculationsDenorm,calculationsNorm,calculationsDeNormArima,calculationsNormArima,calculationsNormGBTs]
  return(outputDf, calculations, predArima)
##########################################################################################
"""# Get Stock Data
##########################################################################################
### Day Picker
"""
def str_time_prop(start, end, time_format, prop):
  stime = time.mktime(time.strptime(start, time_format))
  etime = time.mktime(time.strptime(end, time_format))
  ptime = stime + prop * (etime - stime)
  return time.strftime(time_format, time.localtime(ptime))
##########################################################################################
def random_date(start, end, prop):
  return str_time_prop(start, end, '%Y/%m/%d', prop)

bdays=BDay()
##########################################################################################
def is_business_day(date):
  return date == date + 0*bdays

cal = USFederalHolidayCalendar()
holidays = cal.holidays(start="2021/1/4", end="2022/9/15").to_pydatetime()
print(holidays[0])
# np.append(holidays,datetime(2022, 6, 20))
print(holidays[-2])
for i in range(len(holidays)):
  holidays[i] = holidays[i].date()

print(holidays[-3])
##########################################################################################
def genRandTestDays(numbOfDays):
  daysList = []
  while len(daysList) < numbOfDays:
    tempDay = datetime.strptime(random_date("2021/1/4", "2022/9/15", random.random()),'%Y/%m/%d').date()
    if(is_business_day(tempDay)):
      # checkIfNew = daysOfInterest.count(tempDay)
      if(tempDay not in holidays):
        if(checkIfNew == 0):
          daysList.append(tempDay)

  daysList.sort()
  print(len(daysList))
  return daysList
##########################################################################################
##########################################################################################
def scalingFunc(dfObserve):
  closing_only = dfObserve[["close"]]
  encoder_closing_1 = QuantileTransformer(output_distribution="normal")
  # print(closing_only)
  closing_only = encoder_closing_1.fit_transform(closing_only)  

  encoder_closing_2 = MinMaxScaler()
  closing_only = encoder_closing_2.fit_transform(closing_only)
  scaled = dfObserve.copy()
  encoder1 = QuantileTransformer(output_distribution="normal")
  scaled.iloc[:,0:] = encoder1.fit_transform(scaled.iloc[:,0:])  

  encoder2 = MinMaxScaler()
  scaled.iloc[:,0:] = encoder2.fit_transform(scaled.iloc[:,0:])

  dfObserve = scaled
  return [dfObserve,encoder_closing_1,encoder_closing_2]
##########################################################################################
def testTrainData(dfObserve,noOfColumns):
  testData  = dfObserve.iloc[33:,0:noOfColumns]
  # testData  = dfObserve.iloc[33:,0:]

  #allLabels = dfObserve.loc[date(2010,8,16):,"NextDayClose"]

  allLabels = dfObserve.loc[dfObserve.index.get_level_values(level = 'date') >= '2010-08-16', "NextDayClose"]

  # pL = 1
  # end = daysOfInterest[pL] - timedelta(days=1)
  # start = end - timedelta(days=100)
  # dateRange = (testData.loc[start:end])
  # labelRange = (allLabels.loc[start:end])

  testData.reset_index(inplace=True)
  # print(testData)
  slices = [testData,allLabels]
  return slices
##########################################################################################
def trainData(testDay,slicingFactor,ticker):
  X_train = []
  y_train = []
  
  X_test = []
  X_testOne = []

  testDay = testData[testData['date'] == testDay].index[0]

  if ticker=='aapl':
    tempDf = testData.drop(['dividends', 'splits', 'date', 'symbol'], axis=1)
  else:
    tempDf = testData.drop(['splits', 'date', 'symbol'], axis=1)

  # X_testDay = tempDf[tempDf['date'] == daysOfInterest[0]].index[0]

  # tempDf = tempDf.drop('date', axis=1)
  for i in range(testDay-slicingFactor, testDay):
    X_test.append(tempDf[i:slicingFactor+i])
  X_testOne.append(X_test[1])
  X_test = np.array(X_testOne)

  # print(tempDf)
  for i in range(slicingFactor, testDay-slicingFactor):
    X_train.append(tempDf[i:slicingFactor+i])
    y_train.append(allLabels[i])
    # print(X_train)
  X_train, y_train = np.array(X_train), np.array(y_train)

  return(X_train,y_train,X_test)
##########################################################################################
def trainingData(trainingDays,slicingFactor,testData,scaled,allLabels,gtLabelsAll,ticker,minTimeStepsWindow,startingDay=False):
  X_train = []
  y_train = []
  classes = []
  
  trainIDs = []
  classesIDs = []
  
  symbol = ticker

  print(trainingDays[0])
  trainingDays[0] = datetime.strftime(trainingDays[0], '%Y-%m-%d')
  print(trainingDays[0])

  testData.head()

  upperBound = testData[testData['date'] == trainingDays[0]].index[0] 

  if (startingDay):
    sliceFrom = testData[testData['date'] == datetime.strftime(startingDay, '%Y-%m-%d')].index[0]

    # We ensure at least minTimeStepsWindow windows if the current date is too close to the previous test date, which would cause overfiting on very few windows 
    if ((upperBound-sliceFrom) < minTimeStepsWindow):
      sliceFrom = upperBound - minTimeStepsWindow
  else:
    sliceFrom = 0

  print("startingDay: " + str(startingDay))
  print("sliceFrom: " + str(sliceFrom))

  # TODO: This for loop actually appends a single entry and should be fixed
  for i in range(len(trainingDays)):
    print(type(trainingDays[i]))
    print(scaled.loc[trainingDays[i],symbol]['PrevDayTrend'])
    trainIDs.append(testData[testData['date'] == trainingDays[i]].index[0])
    # print("AAAA")
    # print(scaled["date"])
    # print(scaled[scaled['date'] == trainingDays[i]])
    classesIDs.append(scaled.loc[trainingDays[i],symbol]['PrevDayTrend'])
    # WARNING :ASSUMES SINGLE STOCK IN SCALED

  print(len(classesIDs))

  if ticker=='aapl':
    tempDf = testData.drop(['dividends', 'splits', 'date', 'symbol'], axis=1)
  else:
    tempDf = testData.drop(['splits', 'date', 'symbol'], axis=1)

  for j in range(len(trainIDs)):
    print('sssss')
    print(trainIDs[j])
    for i in range(sliceFrom, trainIDs[j]-1):
      X_train.append(tempDf[i:slicingFactor+i])
      y_train.append(allLabels[i])
      classes.append(gtLabelsAll[i])

  X_train, y_train = np.array(X_train), np.array(y_train)
  classes = np.array(classes)
  print(X_train.shape)

  X_train_full = tempDf[sliceFrom:trainIDs[0]-1]

  #return(X_train,y_train)
  return(X_train,classes,y_train,X_train_full)
##########################################################################################
def getXtest(testDay,slicingFactor,testData,ticker):
  X_test = []
  X_testOne = []
  Y_test_closing_only = []

  # print(testDay)

  #testDay = testData[testData.get_level_values(level = 'date') == testDay].index
  testDay = datetime.strftime(testDay, '%Y-%m-%d')
  # print(type(testDay))
  # print(testDay)
  # print(type(testData['date'][0]))
  # print(testData[testData['date'] == testDay])
  # print(testData[testData['date'] == testDay].index.values)
  testDay = testData[testData['date'] == testDay].index[0]
  
  # print(testDay)
  #tempDf = testData.drop(['dividends', 'splits', 'date', 'symbol'], axis=1)

  if ticker=='aapl':
    tempDf = testData.drop(['dividends', 'splits', 'date', 'symbol'], axis=1)
  else:
    tempDf = testData.drop(['splits', 'date', 'symbol'], axis=1)

  # print(tempDf)
  # tempDf = testData.drop(['dividends', 'splits', 'symbol'], axis=1)

  for i in range(testDay-slicingFactor, testDay):
    X_test.append(tempDf[i:slicingFactor+i])
    Y_test_closing_only.append(tempDf['close'][i])
    # print(tempDf['close'][i])

  # print(X_test[0])
  # print(X_test[1])
  X_testOne.append(X_test[0]) ##0 current day excluded, 1 included
  X_test = np.array(X_testOne)

  return(X_test,Y_test_closing_only)
##########################################################################################
"""tweak learning rate?"""
def scheduler(epoch,lr):
  if epoch< 10:
    return lr
  else:
    return lr* tf.math.exp(-0.1)
##########################################################################################
def trainModel(X_train,y_train,epochs,learningRateInitial,model=False,):
  verbose, epochs, batch_size = 2, epochs, 32
  if (model==False):  
    myopt=tf.keras.optimizers.Adam(learning_rate=learningRateInitial)
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], 1# y_train.shape[1]
    model = Sequential();
    model.add(LSTM(500, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])));
    model.add(LSTM(100, dropout=0.1));
    # model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    loss=tf.keras.losses.BinaryCrossentropy() 
    model.compile(loss=loss, optimizer=myopt, metrics=['mse'])
  
  es = tf.keras.callbacks.EarlyStopping(patience=25, monitor='loss')
  lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

  loss=tf.keras.losses.BinaryCrossentropy() 

  print("X_train shape: ", np.shape(X_train))
  print("y_train shape: ", np.shape(y_train))

  # Manual schedule for LR with reduction
  # myopt=tf.keras.optimizers.Adam(learning_rate=learningRateFollowingIterations)

  # Automatic schedule for LR with exponential reduction
  myopt=tf.keras.optimizers.Adam(learning_rate=learningRateInitial)
  model.compile(loss=loss, optimizer=myopt, metrics=['mse'])
  model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[es,lr], verbose=True, shuffle=False)
  
  return model
##########################################################################################
def trainModelForecasting(X_train,y_train,learningRateInitial):
  myopt=tf.keras.optimizers.Adam(learning_rate=learningRateInitial)
  verbose, epochs, batch_size = 2, 30, 32
  n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], 1# y_train.shape[1]
  model = Sequential();
  model.add(LSTM(500, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])));
  model.add(LSTM(100, dropout=0.1));
  model.add(Dense(1));
  loss=tf.keras.losses.MeanSquaredError();
  model.compile(loss=loss, optimizer=myopt, metrics=['mse'])
  # fit network
  #model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=True, shuffle=False)
  model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=True, shuffle=False)
  # evaluate model
  # _, accuracy = model.evaluate(X_test,y_test, batch_size=batch_size, verbose=0)
  return model
##########################################################################################
def runCalculations(actual,predictions):
  mae = metrics.mean_absolute_error(actual, predictions)
  mse = metrics.mean_squared_error(actual, predictions)
  rmse = np.sqrt(mse) # or mse**(0.5)  
  r2 = metrics.r2_score(actual,predictions)

  mape = mean_absolute_percentage_error(actual, predictions)

  return [mae, mse, rmse, r2, mape]

##########################################################################################
def prepForStacked(testDf,tn,trends,forecastingFlag,text_threshold,ticker):
  tnI = tn.index
  arimaPreds = []
  gbtPreds = []
  lstmDiff = [] #1 if predicted is higher, 0 is predicted is lower
  bertScore = [] #1 if positive / neutral, 0 if negative
  trueTrend = []
  predictionsProbLSTM = []

  actualEntriesTNI = []

  polarityArr = []
  polarityScores = []
  posOrNeg = []
  
  print(tn.iloc[0])

  # Classification metrics can be run just feeding GT_classes and LSTM_from_prices_to_classes
  
  for i in range(len(tn)): 

    tempVar = tn.iloc[i] # Here we are not asssessing the model as correct or wrong, but just noting if it predicted an up or downtrend
    
    print(tempVar)

    tn['Date'] = tn.index
    
    #tempPrevDay = datetime.strptime(tn.iloc[i]['Date'], '%m/%d/%Y') - timedelta(days=1)
    
    tempPrevDay = datetime.strptime(tn.iloc[i]['Date'], '%m/%d/%Y') - timedelta(days=1)
    print(tempPrevDay)
    #___________________
    
    #print(is_business_day(tempPrevDay))
    #print(tempPrevDay-timedelta(days=1))
    #print(is_business_day(tempPrevDay-timedelta(days=1) ))

    while(is_business_day(tempPrevDay) == False):
      tempPrevDay= tempPrevDay-timedelta(days=1)
      print(tempPrevDay)

    #print(tempPrevDay)

    # tempPrevDay = datetime.strptime(tempPrevDay, '%m/%d/%Y')
    tempPrevDay = tempPrevDay.strftime("%Y-%m-%d")
    print("Prev day ", str(tempPrevDay))

    try:
      prevDayInfo = testDf.loc[tempPrevDay,ticker]

      if(forecastingFlag==True):           
        if(prevDayInfo['close']<tempVar['predicted']): #actual is actually NextDayInfo
          lstmDiff.append(1) # LSTM is predicting an uptrend : 
        else:
          lstmDiff.append(0) # LSTM is predicting a downtrend
      else:
        # print(tempVar['predicted'])
        lstmDiff.append(tempVar['predicted']) 


      gbtPreds.append(tempVar['GBTsPreds'])
      predictionsProbLSTM.append(tempVar['predictionsProbLSTM'])

      # print("SANITY", str(prevDayInfo['close']))
      # print("SANITY", str(tempVar['deNorm_ArimaPred']))
        
      if(prevDayInfo['close']<tempVar['deNorm_ArimaPred']): #actual is actually NextDayInfo
        arimaPreds.append(1) # LSTM is predicting an uptrend : 
      else:
        arimaPreds.append(0) # LSTM is predicting a downtrend

      if(tempVar['PosOrNeg'] == 'Positive'):
        posOrNeg.append(1)
      elif(tempVar['PosOrNeg'] == 'Negative'):
        posOrNeg.append(0)
      
      print(tn.index[i])

      #currentDay = testDf.loc[tn.index[i],ticker]
      currentDay = testDf.loc[datetime.strptime(tn.iloc[i]['Date'], '%m/%d/%Y').strftime("%Y-%m-%d"),ticker]
      print("currentDay trend: ", str(currentDay['PrevDayTrend']))

      trueTrend.append(int(currentDay['PrevDayTrend']))

      # for thresh in range(0,1,0.01):
      if(tempVar['Polarity']>text_threshold): #actual is actually NextDayInfo
        polarityArr.append(1);
      else:
        polarityArr.append(0)

      polarityScores.append(tempVar['Polarity'])

      actualEntriesTNI.append(tnI[i])
      print()
    except:
      print("Date not found")
      print()
      continue

  print(len(actualEntriesTNI))
  print(len(tnI))
  print(len(lstmDiff))
  print(len(posOrNeg))
  print(len(polarityArr))
  print(len(trueTrend))
  print(len(arimaPreds))

    # if(tempVar['Positive']>tempVar['Negative']):
    #     posOrNeg.append(1)
    # else:
    #     posOrNeg.append(-1)

      # polarityArr.append(tempVar['Polarity'])
  stackingTraining = pd.DataFrame(index=actualEntriesTNI)
  stackingTraining['lstmDiff'] = lstmDiff
  #stackingTraining['PosOrNeg'] = posOrNeg
  stackingTraining['arimaPreds'] = arimaPreds
  stackingTraining['HeadlinesPol'] = polarityArr
  stackingTraining['trueTrend'] = trueTrend
  stackingTraining['gbtPreds'] = gbtPreds
  stackingTraining['predictionsProbLSTM'] = predictionsProbLSTM
  
  allArr = [lstmDiff,polarityArr,trueTrend,arimaPreds]
  allPD = pd.DataFrame()
  allPD['lstmDiff'] = lstmDiff 
  #allPD['PosOrNeg'] = posOrNeg
  allPD['HeadlinesPol'] = polarityArr
  allPD['trueTrend'] = trueTrend
  print(trueTrend)

  return [stackingTraining,trueTrend,allPD,lstmDiff,posOrNeg,polarityArr,arimaPreds,actualEntriesTNI,polarityScores,gbtPreds,predictionsProbLSTM]
##########################################################################################
def generateStackingDataset(stackingTraining, filename='stackModelTrain.csv'):
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
  stackModelTrain.to_csv(filename, encoding = 'utf-8-sig')
##########################################################################################
# Find the best threshold for binary predictions of sentiment model: unused now, and partially unfair, but it didn't lead to good results 
##########################################################################################
def bestPredsTextModel(tn):
  polarityArr = []
  f_scores = [] 
  thresholds = [] 
  trueTrend = []

  for i in range(len(tn)): 
    tempVar = tn.iloc[i] # Here we are not asssessing the model as correct or wrong, but just noting if it predicted an up or downtrend
    
    tempPrevDay = tn.index[i] - timedelta(days=1)  
    print(tempPrevDay)

    while(is_business_day(tempPrevDay) == False):
      tempPrevDay= tempPrevDay-timedelta(days=1)

    try:
      currentDay = dfArr[0].loc[tn.index[i],ticker]
      trueTrend.append(int(currentDay['PrevDayTrend']))
    except:
      print("Date not found")
      continue

  thresh = 0.01

  while(thresh < 0.99):
    polarityArr = []

    for i in range(len(tn)): 
      if(tempVar['Polarity']>thresh): #actual is actually NextDayInfo
        polarityArr.append(1)
      else:
        polarityArr.append(0)

    print('Threshold ', str(thresh))
    print(precision_recall_fscore_support(trueTrend,polarityArr, average='macro'))
    print(precision_recall_fscore_support(trueTrend,polarityArr, average='micro'))
    print(precision_recall_fscore_support(trueTrend,polarityArr, average='weighted'))
    print()
    f_scores.append(precision_recall_fscore_support(trueTrend,polarityArr, average='weighted')[2])
    thresholds.append(thresh)
    thresh += 0.05

  max_value = max(f_scores)
  max_index = f_scores.index(max_value)
  best_thresh = thresholds[max_index]

  print('Best threshold ', str(best_thresh))

  polarityArr = []

  for i in range(len(tn)): 
    if(tempVar['Polarity']>best_thresh): #actual is actually NextDayInfo
      polarityArr.append(1)
    else:
      polarityArr.append(0)

  print('Preds ', str(polarityArr))

  return polarityArr
##########################################################################################
def genNewXTest(days,ticker):
  X_test = []
  Y_test = []
  newTestDays = genRandTestDays(days)
  for i in range(len(newTestDays)):
    print(i)
    tempVal = getXtest(newTestDays[i],slicingFV,ticker)
    X_test.append(tempVal[0])
    Y_test.append(tempVal[1])
  return X_test,Y_test,newTestDays
##########################################################################################
def genNewXTestStatic(dates,ticker):
  X_test = []
  Y_test = []

  for i in range(len(dates)):
    print(i)
    tempVal = getXtest(dates[i],slicingFV,ticker)
    X_test.append(tempVal[0])
    Y_test.append(tempVal[1])
  return X_test,Y_test,dates