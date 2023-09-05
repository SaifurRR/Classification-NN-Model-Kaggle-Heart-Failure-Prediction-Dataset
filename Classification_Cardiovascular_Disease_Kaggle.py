import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
import numpy as np

#Load dataset:
data=pd.read_csv('heart_failure.csv')
pd.set_option('display.max_columns', None)
print(data.info())
#label classification
label_distribution=Counter(data['death_event'])
print(f'Classes and number of values in dataset: {label_distribution}')
#label column
y=data['death_event'] #pd.Series
#features column
x=data[['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time']] #pd.Dataframe: as multiple cols

#Data Pre-processing:
x=pd.get_dummies(x) #one-hot coding -> only categorical
#train-test-split
X_train, X_test, Y_train, Y_test=train_test_split(x,y,test_size=0.25, random_state=43)
#Scaling: Numerical Features
ct=ColumnTransformer([('standardscaler', StandardScaler(),X_train.columns)], remainder='passthrough')
#train
X_train=ct.fit_transform(X_train) #scaled vector[[#of features: len 17][#of features: len 17][...]]
#test
X_test=ct.transform(X_test) #vector same length & scaler factor as train

#Labels for Classification:
le=LabelEncoder()
Y_train=le.fit_transform(Y_train.astype(str)) #binary coded
#[1 0 0 0 0 1] -> one-long vector
Y_test=le.transform(Y_test.astype(str)) #binary coded ->  [1 0 0 0 0 1] 
#print(Y_test)

#split long vector into vectors of category types 
Y_train=to_categorical(Y_train, dtype='int64')
Y_test=to_categorical(Y_test, dtype='int64')
#print(len(Y_test)) # [[1 0] [0 1] [1 0] [0 1]]

#Design Model:
model=Sequential()
no_features=X_train.shape[1] # number of features/nodes
#input layer
model.add(InputLayer(input_shape=(no_features,)))
#hidden layer
model.add(Dense(12, activation='relu'))
#output layer
no_classes=len(le.classes_) #2 labels
model.add(Dense(no_classes, activation='softmax')) #softmax probability distribution
#compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #measure loss, optimize on a given metric

#Train & Evaluate Model:
model.fit(X_train, Y_train, epochs=100, batch_size=16, verbose=0)
#batch_size: # of training examples used in parallel in each iteration

#Evaluate:
categorical_loss, accuracy= model.evaluate(X_test, Y_test, verbose=0)
print(f'Categorical Crossentropy: {categorical_loss}\n',f'Accuracy: {accuracy}')

#Classification Report:
#estimate
y_estimate=model.predict(X_test, verbose=0)
y_estimate=np.argmax(y_estimate, axis=1) #select indices
#print(y_estimate)
print()
#true
y_true=np.argmax(Y_test, axis=1)

#classification report:
classification_report=classification_report(y_true, y_estimate)
print(classification_report)

