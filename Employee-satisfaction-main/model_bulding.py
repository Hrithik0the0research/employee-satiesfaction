# %%
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.models import Sequential
#import xgboost as xgb
import lime
from lime.lime_tabular import LimeTabularExplainer
import lime.lime_tabular
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# %%
data=pd.read_excel("new_data.xlsx")
data

# %%
class_value=data["PRESENT JOB FEELING"]
features=data.columns[7:]
features_value=data[features]
print(features_value,class_value)
x=features_value
y=class_value

# %%
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1, random_state=42)

# %%
x_test.shape

# %%
classifier = SVC(C=10,kernel='poly',degree=5,gamma=10,probability=True)  
y_score=classifier.fit(x_train, y_train)


# %%

y_pred=classifier.predict(x_test)
print(y_pred,y_test)
print(metrics.accuracy_score(y_test,y_pred))


# %%
"""best model"""
filename = 'finalized_model.sav'
pickle.dump(classifier, open(filename, 'wb'))

# %%
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
xgb_classifier = xgb.XGBClassifier(n_estimators = 400, learning_rate = 0.6, max_depth = 6)
xgb_classifier.fit(x_train,y_train)

# %%
predictions = xgb_classifier.predict(x_test)
pred=[]
for i in predictions:
    pred.append(i+1)
pred=np.array(pred)
metrics.accuracy_score(y_test,pred)

# %%
clf = RandomForestClassifier(max_depth=5, random_state=42)
clf.fit(x_train, y_train)


# %%

print(clf.predict(x_test))
print(metrics.accuracy_score(y_test,clf.predict(x_test)))

# %%
s=["strongly disagree","disagree","neutral","agree"," strongly agree"]
explainer = lime.lime_tabular.LimeTabularExplainer(x_train.values,feature_names=features, class_names=s,mode = 'classification')

# %%
print(x_test.iloc[0].values)
exp = explainer.explain_instance(x_test.iloc[0].values,xgb_classifier.predict_proba, num_features = 4)
exp.show_in_notebook(show_table = True, show_all = False)

# %%
print(x_test.iloc[0].values)
exp = explainer.explain_instance(x_test.iloc[0].values,classifier.predict_proba, num_features = 17)
exp.show_in_notebook(show_table = True, show_all = False)

# %%
exp.save_to_file("index.html")

# %%
print(x_test.iloc[0].values)
exp = explainer.explain_instance(x_test.iloc[0].values,clf.predict_proba, num_features = 4)
exp.show_in_notebook(show_table = True, show_all = False)

# %%
import shap
shap.initjs()

# %%
""""
for svc
"""
shap_explainer = shap.KernelExplainer(classifier.predict_proba,x_train)
shap_values = shap_explainer.shap_values(x_test)

# %%
shap.summary_plot(shap_values,x_test,class_names=s, feature_names =features)

# %%
s=shap.force_plot(shap_explainer.expected_value[0],shap_values[0],x_test)
shap.save_html("force.html",s)

# %%
s1=shap.decision_plot(shap_explainer.expected_value[1], shap_values[1], x_test)

# %%
shap.dependence_plot('DOWNHRTED', shap_values[1], x_test[0], interaction_index='DOWNHRTED') 

# %%
r_probs=[0 for _ in range(len(y_test))]
rf_probs=classifier.predict_proba(x_test)
nb_probs=xgb_classifier.predict_proba(x_test)

# %%
