import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

data = pd.read_csv("storepurchasedata.csv")
print(data.describe())

X = data.iloc[:,:-1].values 
Y = data.iloc[:,-1].values

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size= 0.2, random_state=0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier 

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

#model training 
classifier.fit(x_train, y_train)

y_predict = classifier.predict(x_test)
y_prob = classifier.predict_proba(x_test)[:,1]

#print(y_predict)
#print(y_test)

cm = confusion_matrix(y_test, y_predict)
print(cm)

print(accuracy_score(y_test, y_predict))

print(classification_report(y_test, y_predict))

prediction = classifier.predict(sc.transform(np.array([[40,20000]])))
print(prediction)
new_proba = classifier.predict_proba(sc.transform(np.array([[40,20000]])))[:,0]
print(new_proba)

#----

import pickle

model_file = "classifier.pickle"
pickle.dump(classifier,open(model_file,'wb'))

scaler_file = "sc.pickle"
pickle.dump(sc, open(scaler_file,'wb'))



