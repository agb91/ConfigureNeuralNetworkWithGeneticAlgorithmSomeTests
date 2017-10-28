from numpy import loadtxt
import numpy as np
import tensorflow as tf
import pandas as pd
import math
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


path = "/home/andrea/Desktop/python/tensorflowTutorialGenerical/"
abalone_train = path + "training"
abalone_test = path + "test"
abalone_toPredict = path + "toPredict"

CSV_COLUMNS = [ "a","b","c","d","e","f","g","target"]
# Training examples
dataset_train = pd.read_csv(abalone_train, names=CSV_COLUMNS, skipinitialspace=True)
dataset_test = pd.read_csv(abalone_test, names=CSV_COLUMNS, skipinitialspace=True)
dataset_toPredict = pd.read_csv(abalone_toPredict, names=CSV_COLUMNS, skipinitialspace=True)


#print (dataset)

X_train = dataset_train[["a","b","c","d","e","f","g"]]
Y_train = dataset_train["target"]

X_test = dataset_test[["a","b","c","d","e","f","g"]]
Y_test = dataset_test["target"]

X_toPredict = dataset_toPredict[["a","b","c","d","e","f","g"]]
Y_toPredict = dataset_toPredict["target"]

#print(Y)



# fit model no training data
model = XGBRegressor()
model.fit(X_train, Y_train)

print( model )

y_pred = model.predict( X_test )
y_pred_brief = model.predict( X_toPredict )
#print( y_pred )

MSE = 0


for i in range( 0, len(Y_test) ):
	toAdd = (Y_test[i] - y_pred[i])**2
	toAdd = math.sqrt( toAdd )
	MSE = MSE + toAdd



	
MSE = MSE / len(Y_test)

print("MSE :    " + str(MSE) )


for i in range ( 0 , len(Y_toPredict)  ):
	print( "expected:  " + str( y_pred_brief[i] ) + ";   vs real:   " + str( Y_toPredict[i] ) )

print("finished")





