import pickle 
import numpy as np  


local_classifier = pickle.load(open('classifier.pickle', 'rb'))
local_sc = pickle.load(open('sc.pickle', 'rb'))

new_prediction = local_classifier.predict(local_sc.transform(np.array([[42,50000]])))
new_prediction_prob = local_classifier.predict_proba(local_sc.transform(np.array([[42,50000]])))[:,0]
