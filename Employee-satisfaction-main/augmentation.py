# %%
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE 
import pandas as pd
import numpy as np 


# %%
data=pd.read_excel("new_data.xlsx")
print(data)

# %%
class_value=data["PRESENT JOB FEELING"]
class_value
feature_name_list=data.columns[7:]
feature_name_list
feature_final_list=[]
for i in feature_name_list:
    if "Unnamed: " not in i:
        feature_final_list.append(i)
print(feature_final_list)
features=data[feature_final_list]
features.shape
features=features.dropna()
features.shape

# %%

class_value.shape
features=features.astype("int")
class_value=class_value.astype("int")
class_value=class_value.dropna()
class_value.shape

x=features
y=class_value

sm = SMOTE(random_state=122)
X_res, y_res = sm.fit_resample(data, y)
X_res
