#%%
#MOR SVR
from scipy.stats import loguniform
from sklearn.multioutput import MultiOutputRegressor
from pandas import read_csv
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
import pandas as pd
import json

data=pd.read_excel("dataset.xlsx")
x=data.sample(frac=1).reset_index(drop=True)
HOM_mean,TTT_mean=x.mean()[:2]
HOM_std,TTT_std=x.std()[:2]
x["HOM"]=(x["HOM"]-HOM_mean)/HOM_std
x["TTT"]=(x["TTT"]-TTT_mean)/TTT_std

###Train-test split and only second index of j changes when output changes
X_train,X_test=x.iloc[:450,:2],x.iloc[450:,:2]
y_train,y_test=x.iloc[:450,2:],x.iloc[450:,2:]
X_train,X_test=X_train.values,X_test.values
y_train,y_test=y_train.values,y_test.values


###Output mean and standard deviations
y_tr_means,y_ts_means=y_train.mean(axis=0),y_test.mean(axis=0)
y_tr_stds,y_ts_stds=y_train.std(axis=0),y_test.std(axis=0)

#####Normalization of output
y_train,y_test=(y_train-y_tr_means)/y_tr_stds,(y_test-y_ts_means)/y_ts_stds


#define model
model=SVR()
mor=MultiOutputRegressor(model)

#define evaluation
cv=RepeatedKFold(n_splits=10,random_state=1)
space = {'estimator__kernel': ('linear', 'rbf','poly'), 'estimator__C':[1,5,10],'estimator__gamma': [1e-4,1e-2],'estimator__epsilon':[0.1,0.2,0.5,0.3]}


#define search
search = GridSearchCV(mor, space, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv,verbose=1)
#execute search 
result=search.fit(X_train,y_train)
best_model=result.best_estimator_

#make prediction
preds=best_model.predict(X_test)

#Printing
best_parameters_for_SVR=result.best_params_

with open('SVR_params.txt', 'w') as convert_file:
     convert_file.write(json.dumps(best_parameters_for_SVR))
#%%
#MOR Kernel Ridge
from scipy.stats import loguniform
from sklearn.multioutput import MultiOutputRegressor
from pandas import read_csv
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
import pandas as pd
import json

data=pd.read_excel("dataset.xlsx")
x=data.sample(frac=1).reset_index(drop=True)
HOM_mean,TTT_mean=x.mean()[:2]
HOM_std,TTT_std=x.std()[:2]
x["HOM"]=(x["HOM"]-HOM_mean)/HOM_std
x["TTT"]=(x["TTT"]-TTT_mean)/TTT_std

###Train-test split and only second index of j changes when output changes
X_train,X_test=x.iloc[:450,:2],x.iloc[450:,:2]
y_train,y_test=x.iloc[:450,2:],x.iloc[450:,2:]
X_train,X_test=X_train.values,X_test.values
y_train,y_test=y_train.values,y_test.values


###Output mean and standard deviations
y_tr_means,y_ts_means=y_train.mean(axis=0),y_test.mean(axis=0)
y_tr_stds,y_ts_stds=y_train.std(axis=0),y_test.std(axis=0)

#####Normalization of output
y_train,y_test=(y_train-y_tr_means)/y_tr_stds,(y_test-y_ts_means)/y_ts_stds


#define model
model=KernelRidge()
mor=MultiOutputRegressor(model)

#define evaluation
cv=RepeatedKFold(n_splits=10,random_state=1)
space = {'estimator__kernel': ('linear', 'rbf','poly','sigmoid'), 'estimator__degree':[1,2,3],'estimator__alpha':[0.1,0.5,1,2]}


#define search
search = GridSearchCV(mor, space, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv,verbose=1)
#execute search 
result=search.fit(X_train,y_train)
best_model=result.best_estimator_

#make prediction
preds=best_model.predict(X_test)

#Printing
best_parameters_for_KRR=result.best_params_

with open('KRR_params.txt', 'w') as convert_file:
     convert_file.write(json.dumps(best_parameters_for_KRR))

#%%
#MOR GBR
from scipy.stats import loguniform
from sklearn.multioutput import MultiOutputRegressor
from pandas import read_csv
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
import pandas as pd
import json

data=pd.read_excel("dataset.xlsx")
x=data.sample(frac=1).reset_index(drop=True)
HOM_mean,TTT_mean=x.mean()[:2]
HOM_std,TTT_std=x.std()[:2]
x["HOM"]=(x["HOM"]-HOM_mean)/HOM_std
x["TTT"]=(x["TTT"]-TTT_mean)/TTT_std

###Train-test split and only second index of j changes when output changes
X_train,X_test=x.iloc[:450,:2],x.iloc[450:,:2]
y_train,y_test=x.iloc[:450,2:],x.iloc[450:,2:]
X_train,X_test=X_train.values,X_test.values
y_train,y_test=y_train.values,y_test.values


###Output mean and standard deviations
y_tr_means,y_ts_means=y_train.mean(axis=0),y_test.mean(axis=0)
y_tr_stds,y_ts_stds=y_train.std(axis=0),y_test.std(axis=0)

#####Normalization of output
y_train,y_test=(y_train-y_tr_means)/y_tr_stds,(y_test-y_ts_means)/y_ts_stds


#define model
model=GBR()
mor=MultiOutputRegressor(model)

#define evaluation
cv=RepeatedKFold(n_splits=10,random_state=1)
space={'estimator__n_estimators': [1,5,10,50,100,200,500], 'estimator__max_depth':[1,3,5],
       'estimator__learning_rate': [0.01,0.1,0.3]}
#define search
search = GridSearchCV(mor, space, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv,verbose=1)
#execute search 
result=search.fit(X_train,y_train)
best_model=result.best_estimator_

#make prediction
preds=best_model.predict(X_test)

#Printing
best_parameters_for_GBR=result.best_params_

with open('GBR_params.txt', 'w') as convert_file:
     convert_file.write(json.dumps(best_parameters_for_GBR))

#%%
#MOR AdaBoostRegressor
from sklearn.ensemble import AdaBoostRegressor as ABR
from sklearn.multioutput import MultiOutputRegressor
from pandas import read_csv
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
import pandas as pd
import json

data=pd.read_excel("dataset.xlsx")
x=data.sample(frac=1).reset_index(drop=True)
HOM_mean,TTT_mean=x.mean()[:2]
HOM_std,TTT_std=x.std()[:2]
x["HOM"]=(x["HOM"]-HOM_mean)/HOM_std
x["TTT"]=(x["TTT"]-TTT_mean)/TTT_std

###Train-test split and only second index of j changes when output changes
X_train,X_test=x.iloc[:450,:2],x.iloc[450:,:2]
y_train,y_test=x.iloc[:450,2:],x.iloc[450:,2:]
X_train,X_test=X_train.values,X_test.values
y_train,y_test=y_train.values,y_test.values


###Output mean and standard deviations
y_tr_means,y_ts_means=y_train.mean(axis=0),y_test.mean(axis=0)
y_tr_stds,y_ts_stds=y_train.std(axis=0),y_test.std(axis=0)

#####Normalization of output
y_train,y_test=(y_train-y_tr_means)/y_tr_stds,(y_test-y_ts_means)/y_ts_stds


#define model
model=ABR()
mor=MultiOutputRegressor(model)

#define evaluation
cv=RepeatedKFold(n_splits=10,random_state=1)
space={'estimator__n_estimators': [10,50,100,200,500],
       'estimator__learning_rate': [0.001,0.03,0.1,0.3,1]}
#define search
search = GridSearchCV(mor, space, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv,verbose=1)
#execute search 
result=search.fit(X_train,y_train)
best_model=result.best_estimator_

#make prediction
preds=best_model.predict(X_test)

#Printing
best_parameters_for_ABR=result.best_params_

with open('ABR_params.txt', 'w') as convert_file:
     convert_file.write(json.dumps(best_parameters_for_ABR))

#%%
#MOR MLPR
from scipy.stats import loguniform
from sklearn.multioutput import MultiOutputRegressor
from pandas import read_csv
import numpy as np
from sklearn.neural_network import MLPRegressor as MLPR
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
import pandas as pd
import json

data=pd.read_excel("dataset.xlsx")
x=data.sample(frac=1).reset_index(drop=True)
HOM_mean,TTT_mean=x.mean()[:2]
HOM_std,TTT_std=x.std()[:2]
x["HOM"]=(x["HOM"]-HOM_mean)/HOM_std
x["TTT"]=(x["TTT"]-TTT_mean)/TTT_std

###Train-test split and only second index of j changes when output changes
X_train,X_test=x.iloc[:450,:2],x.iloc[450:,:2]
y_train,y_test=x.iloc[:450,2:],x.iloc[450:,2:]
X_train,X_test=X_train.values,X_test.values
y_train,y_test=y_train.values,y_test.values


###Output mean and standard deviations
y_tr_means,y_ts_means=y_train.mean(axis=0),y_test.mean(axis=0)
y_tr_stds,y_ts_stds=y_train.std(axis=0),y_test.std(axis=0)

#####Normalization of output
y_train,y_test=(y_train-y_tr_means)/y_tr_stds,(y_test-y_ts_means)/y_ts_stds


#define model
model=MLPR()
mor=MultiOutputRegressor(model)

#define evaluation
cv=RepeatedKFold(n_splits=10,random_state=1)
space={'estimator__hidden_layer_sizes': [(5,),(50,),(100,)],'estimator__max_iter': [500,1000,1500]}
#define search
search = GridSearchCV(mor, space, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv,verbose=1)
#execute search 
result=search.fit(X_train,y_train)
best_model=result.best_estimator_

#make prediction
preds=best_model.predict(X_test)

#Printing
best_parameters_for_MLPR=result.best_params_

with open('MLPR_params.txt', 'w') as convert_file:
     convert_file.write(json.dumps(best_parameters_for_MLPR))
     
#%%
#MOR KNNRegressor
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.multioutput import MultiOutputRegressor
from pandas import read_csv
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
import pandas as pd
import json

data=pd.read_excel("dataset.xlsx")
x=data.sample(frac=1).reset_index(drop=True)
HOM_mean,TTT_mean=x.mean()[:2]
HOM_std,TTT_std=x.std()[:2]
x["HOM"]=(x["HOM"]-HOM_mean)/HOM_std
x["TTT"]=(x["TTT"]-TTT_mean)/TTT_std

###Train-test split and only second index of j changes when output changes
X_train,X_test=x.iloc[:450,:2],x.iloc[450:,:2]
y_train,y_test=x.iloc[:450,2:],x.iloc[450:,2:]
X_train,X_test=X_train.values,X_test.values
y_train,y_test=y_train.values,y_test.values


###Output mean and standard deviations
y_tr_means,y_ts_means=y_train.mean(axis=0),y_test.mean(axis=0)
y_tr_stds,y_ts_stds=y_train.std(axis=0),y_test.std(axis=0)

#####Normalization of output
y_train,y_test=(y_train-y_tr_means)/y_tr_stds,(y_test-y_ts_means)/y_ts_stds


#define model
model=KNR()
mor=MultiOutputRegressor(model)

#define evaluation
cv=RepeatedKFold(n_splits=10,random_state=1)
n_neighbors=range(1,21,2)
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'minkowski']

space={'estimator__n_neighbors': n_neighbors,
       'estimator__metric': metric,
       'estimator__weights':weights}
#define search
search = GridSearchCV(mor, space, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv,verbose=1)
#execute search 
result=search.fit(X_train,y_train)
best_model=result.best_estimator_

#make prediction
preds=best_model.predict(X_test)

#Printing
best_parameters_for_KNR=result.best_params_

with open('KNR_params.txt', 'w') as convert_file:
     convert_file.write(json.dumps(best_parameters_for_KNR))

#%%
#CatBoostRegressor
from scipy.stats import loguniform
from sklearn.multioutput import MultiOutputRegressor
from pandas import read_csv
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
import pandas as pd
import json
from catboost import CatBoostRegressor as CBR

np.random.seed(13)

data=pd.read_excel("dataset.xlsx")
x=data.sample(frac=1).reset_index(drop=True)
HOM_mean,TTT_mean=x.mean()[:2]
HOM_std,TTT_std=x.std()[:2]
x["HOM"]=(x["HOM"]-HOM_mean)/HOM_std
x["TTT"]=(x["TTT"]-TTT_mean)/TTT_std

###Train-test split and only second index of j changes when output changes
X_train,X_test=x.iloc[:450,:2],x.iloc[450:,:2]
y_train,y_test=x.iloc[:450,2:],x.iloc[450:,2:]
X_train,X_test=X_train.values,X_test.values
y_train,y_test=y_train.values,y_test.values


###Output mean and standard deviations
y_tr_means,y_ts_means=y_train.mean(axis=0),y_test.mean(axis=0)
y_tr_stds,y_ts_stds=y_train.std(axis=0),y_test.std(axis=0)

#####Normalization of output
y_train,y_test=(y_train-y_tr_means)/y_tr_stds,(y_test-y_ts_means)/y_ts_stds


#define model
model=CBR()
mor=MultiOutputRegressor(model)

#define evaluation
cv=RepeatedKFold(n_splits=10,random_state=1)
space={'estimator__max_depth':[6,10],'estimator__learning_rate':[0.009,0.03,0.09,0.3],'estimator__n_estimators':[100,200,500]}
#define search
search = GridSearchCV(mor, space, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv,verbose=1)
#execute search 
result=search.fit(X_train,y_train)
best_model=result.best_estimator_

#make prediction
preds=best_model.predict(X_test)

#Printing
best_parameters_for_CB=result.best_params_

with open('CB_params.txt', 'w') as convert_file:
     convert_file.write(json.dumps(best_parameters_for_CB))
#%%
#Results
#%%
#MOR SVR
from scipy.stats import loguniform
from sklearn.multioutput import MultiOutputRegressor
from pandas import read_csv
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
import pandas as pd
import json

np.random.seed(13)

data=pd.read_excel("dataset.xlsx")
x=data.sample(frac=1).reset_index(drop=True)
HOM_mean,TTT_mean=x.mean()[:2]
HOM_std,TTT_std=x.std()[:2]
x["HOM"]=(x["HOM"]-HOM_mean)/HOM_std
x["TTT"]=(x["TTT"]-TTT_mean)/TTT_std

###Train-test split and only second index of j changes when output changes
X_train,X_test=x.iloc[:450,:2],x.iloc[450:,:2]
y_train,y_test=x.iloc[:450,2:],x.iloc[450:,2:]
X_train,X_test=X_train.values,X_test.values
y_train,y_test=y_train.values,y_test.values


###Output mean and standard deviations
y_tr_means,y_ts_means=y_train.mean(axis=0),y_test.mean(axis=0)
y_tr_stds,y_ts_stds=y_train.std(axis=0),y_test.std(axis=0)

#####Normalization of output
y_train,y_test=(y_train-y_tr_means)/y_tr_stds,(y_test-y_ts_means)/y_ts_stds


#define model
model=SVR(C=10,epsilon=0.1,gamma=0.01,kernel="rbf")
mor=MultiOutputRegressor(model)

#Fit and predict
fitted=mor.fit(X_train,y_train)
preds=fitted.predict(X_test)

#Computations
y_test,y_hat=(y_test*y_ts_stds)+y_ts_means,(preds*y_ts_stds)+y_ts_means
Max_E=np.max(np.abs(y_test-y_hat),axis=0)
MAE_vals=np.mean(np.abs(y_test-y_hat),axis=0)
MSE_vals=np.mean(np.power((y_test-y_hat),2),axis=0)
#%%
#mor KRR
from scipy.stats import loguniform
from sklearn.multioutput import MultiOutputRegressor
from pandas import read_csv
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
import pandas as pd
import json

np.random.seed(13)

data=pd.read_excel("dataset.xlsx")
x=data.sample(frac=1).reset_index(drop=True)
HOM_mean,TTT_mean=x.mean()[:2]
HOM_std,TTT_std=x.std()[:2]
x["HOM"]=(x["HOM"]-HOM_mean)/HOM_std
x["TTT"]=(x["TTT"]-TTT_mean)/TTT_std

###Train-test split and only second index of j changes when output changes
X_train,X_test=x.iloc[:450,:2],x.iloc[450:,:2]
y_train,y_test=x.iloc[:450,2:],x.iloc[450:,2:]
X_train,X_test=X_train.values,X_test.values
y_train,y_test=y_train.values,y_test.values


###Output mean and standard deviations
y_tr_means,y_ts_means=y_train.mean(axis=0),y_test.mean(axis=0)
y_tr_stds,y_ts_stds=y_train.std(axis=0),y_test.std(axis=0)

#####Normalization of output
y_train,y_test=(y_train-y_tr_means)/y_tr_stds,(y_test-y_ts_means)/y_ts_stds


#define model
model=KernelRidge(alpha=0.1,degree=1,kernel="rbf")
mor=MultiOutputRegressor(model)

#Fit and predict
fitted=mor.fit(X_train,y_train)
preds=fitted.predict(X_test)

#Computations
y_test,y_hat=(y_test*y_ts_stds)+y_ts_means,(preds*y_ts_stds)+y_ts_means

MAE_vals=np.mean(np.abs(y_test-y_hat),axis=0)
MSE_vals=np.mean(np.power((y_test-y_hat),2),axis=0)

#%%
#MOR GBR
from scipy.stats import loguniform
from sklearn.multioutput import MultiOutputRegressor
from pandas import read_csv
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
import pandas as pd
import json

np.random.seed(13)

data=pd.read_excel("dataset.xlsx")
x=data.sample(frac=1).reset_index(drop=True)
HOM_mean,TTT_mean=x.mean()[:2]
HOM_std,TTT_std=x.std()[:2]
x["HOM"]=(x["HOM"]-HOM_mean)/HOM_std
x["TTT"]=(x["TTT"]-TTT_mean)/TTT_std

###Train-test split and only second index of j changes when output changes
X_train,X_test=x.iloc[:450,:2],x.iloc[450:,:2]
y_train,y_test=x.iloc[:450,2:],x.iloc[450:,2:]
X_train,X_test=X_train.values,X_test.values
y_train,y_test=y_train.values,y_test.values


###Output mean and standard deviations
y_tr_means,y_ts_means=y_train.mean(axis=0),y_test.mean(axis=0)
y_tr_stds,y_ts_stds=y_train.std(axis=0),y_test.std(axis=0)

#####Normalization of output
y_train,y_test=(y_train-y_tr_means)/y_tr_stds,(y_test-y_ts_means)/y_ts_stds


#define model
model=GBR(learning_rate=0.1,max_depth=5,n_estimators=500)
mor=MultiOutputRegressor(model)

#Fit and predict
fitted=mor.fit(X_train,y_train)
preds=fitted.predict(X_test)

#Computations
y_test,y_hat=(y_test*y_ts_stds)+y_ts_means,(preds*y_ts_stds)+y_ts_means

Max_E=np.max(np.abs(y_test-y_hat),axis=0)
MAE_vals=np.mean(np.abs(y_test-y_hat),axis=0)
MAE_vals=np.around(MAE_vals,5)
MSE_vals=np.mean(np.power((y_test-y_hat),2),axis=0)
#%%
#MOR ABR
from sklearn.ensemble import AdaBoostRegressor as ABR
from sklearn.multioutput import MultiOutputRegressor
from pandas import read_csv
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
import pandas as pd
import json

np.random.seed(13)

data=pd.read_excel("dataset.xlsx")
x=data.sample(frac=1).reset_index(drop=True)
HOM_mean,TTT_mean=x.mean()[:2]
HOM_std,TTT_std=x.std()[:2]
x["HOM"]=(x["HOM"]-HOM_mean)/HOM_std
x["TTT"]=(x["TTT"]-TTT_mean)/TTT_std

###Train-test split and only second index of j changes when output changes
X_train,X_test=x.iloc[:450,:2],x.iloc[450:,:2]
y_train,y_test=x.iloc[:450,2:],x.iloc[450:,2:]
X_train,X_test=X_train.values,X_test.values
y_train,y_test=y_train.values,y_test.values


###Output mean and standard deviations
y_tr_means,y_ts_means=y_train.mean(axis=0),y_test.mean(axis=0)
y_tr_stds,y_ts_stds=y_train.std(axis=0),y_test.std(axis=0)

#####Normalization of output
y_train,y_test=(y_train-y_tr_means)/y_tr_stds,(y_test-y_ts_means)/y_ts_stds


#define model
model=ABR(learning_rate=0.3,n_estimators=200)
mor=MultiOutputRegressor(model)

#Fit and predict
fitted=mor.fit(X_train,y_train)
preds=fitted.predict(X_test)

#Computations
y_test,y_hat=(y_test*y_ts_stds)+y_ts_means,(preds*y_ts_stds)+y_ts_means

MAE_vals=np.mean(np.abs(y_test-y_hat),axis=0)
MSE_vals=np.mean(np.power((y_test-y_hat),2),axis=0)
#%%
#MOR KNNR
from scipy.stats import loguniform
from sklearn.multioutput import MultiOutputRegressor
from pandas import read_csv
import numpy as np
from sklearn.neighbors import KNeighborsRegressor as KNNR
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
import pandas as pd
import json

np.random.seed(13)

data=pd.read_excel("dataset.xlsx")
x=data.sample(frac=1).reset_index(drop=True)
HOM_mean,TTT_mean=x.mean()[:2]
HOM_std,TTT_std=x.std()[:2]
x["HOM"]=(x["HOM"]-HOM_mean)/HOM_std
x["TTT"]=(x["TTT"]-TTT_mean)/TTT_std

###Train-test split and only second index of j changes when output changes
X_train,X_test=x.iloc[:450,:2],x.iloc[450:,:2]
y_train,y_test=x.iloc[:450,2:],x.iloc[450:,2:]
X_train,X_test=X_train.values,X_test.values
y_train,y_test=y_train.values,y_test.values


###Output mean and standard deviations
y_tr_means,y_ts_means=y_train.mean(axis=0),y_test.mean(axis=0)
y_tr_stds,y_ts_stds=y_train.std(axis=0),y_test.std(axis=0)

#####Normalization of output
y_train,y_test=(y_train-y_tr_means)/y_tr_stds,(y_test-y_ts_means)/y_ts_stds


#define model
model=KNNR(n_neighbors=1,weights="uniform",metric="euclidean")
mor=MultiOutputRegressor(model)

#Fit and predict
fitted=mor.fit(X_train,y_train)
preds=fitted.predict(X_test)

#Computations
y_test,y_hat=(y_test*y_ts_stds)+y_ts_means,(preds*y_ts_stds)+y_ts_means

MAE_vals=np.mean(np.abs(y_test-y_hat),axis=0)
MSE_vals=np.mean(np.power((y_test-y_hat),2),axis=0)
#%%
#MOR MLPR
from scipy.stats import loguniform
from sklearn.multioutput import MultiOutputRegressor
from pandas import read_csv
import numpy as np
from sklearn.neural_network import MLPRegressor as MLPR
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
import pandas as pd
import json

np.random.seed(13)

data=pd.read_excel("dataset.xlsx")
x=data.sample(frac=1).reset_index(drop=True)
HOM_mean,TTT_mean=x.mean()[:2]
HOM_std,TTT_std=x.std()[:2]
x["HOM"]=(x["HOM"]-HOM_mean)/HOM_std
x["TTT"]=(x["TTT"]-TTT_mean)/TTT_std

###Train-test split and only second index of j changes when output changes
X_train,X_test=x.iloc[:450,:2],x.iloc[450:,:2]
y_train,y_test=x.iloc[:450,2:],x.iloc[450:,2:]
X_train,X_test=X_train.values,X_test.values
y_train,y_test=y_train.values,y_test.values


###Output mean and standard deviations
y_tr_means,y_ts_means=y_train.mean(axis=0),y_test.mean(axis=0)
y_tr_stds,y_ts_stds=y_train.std(axis=0),y_test.std(axis=0)

#####Normalization of output
y_train,y_test=(y_train-y_tr_means)/y_tr_stds,(y_test-y_ts_means)/y_ts_stds


#define model
model=MLPR(max_iter=1500,hidden_layer_sizes=100)
mor=MultiOutputRegressor(model)

#Fit and predict
fitted=mor.fit(X_train,y_train)
preds=fitted.predict(X_test)

#Computations
y_test,y_hat=(y_test*y_ts_stds)+y_ts_means,(preds*y_ts_stds)+y_ts_means

MAE_vals=np.mean(np.abs(y_test-y_hat),axis=0)
MSE_vals=np.mean(np.power((y_test-y_hat),2),axis=0)
#%%
#KNR
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.multioutput import MultiOutputRegressor
from pandas import read_csv
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
import pandas as pd
import json

data=pd.read_excel("dataset.xlsx")
x=data.sample(frac=1).reset_index(drop=True)
HOM_mean,TTT_mean=x.mean()[:2]
HOM_std,TTT_std=x.std()[:2]
x["HOM"]=(x["HOM"]-HOM_mean)/HOM_std
x["TTT"]=(x["TTT"]-TTT_mean)/TTT_std

###Train-test split and only second index of j changes when output changes
X_train,X_test=x.iloc[:450,:2],x.iloc[450:,:2]
y_train,y_test=x.iloc[:450,2:],x.iloc[450:,2:]
X_train,X_test=X_train.values,X_test.values
y_train,y_test=y_train.values,y_test.values


###Output mean and standard deviations
y_tr_means,y_ts_means=y_train.mean(axis=0),y_test.mean(axis=0)
y_tr_stds,y_ts_stds=y_train.std(axis=0),y_test.std(axis=0)

#####Normalization of output
y_train,y_test=(y_train-y_tr_means)/y_tr_stds,(y_test-y_ts_means)/y_ts_stds


#define model
model=KNR(n_neighbors=1,weights="uniform",metric="euclidean")
mor=MultiOutputRegressor(model)

#Fit and predict
fitted=mor.fit(X_train,y_train)
preds=fitted.predict(X_test)

#Computations
y_test,y_hat=(y_test*y_ts_stds)+y_ts_means,(preds*y_ts_stds)+y_ts_means

MAE_vals=np.mean(np.abs(y_test-y_hat),axis=0)
MSE_vals=np.mean(np.power((y_test-y_hat),2),axis=0)
#%%
#MOR CatBoost
from scipy.stats import loguniform
from sklearn.multioutput import MultiOutputRegressor
from pandas import read_csv
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
import pandas as pd
import json
from catboost import CatBoostRegressor as CBR

np.random.seed(13)

data=pd.read_excel("dataset.xlsx")
x=data.sample(frac=1).reset_index(drop=True)
HOM_mean,TTT_mean=x.mean()[:2]
HOM_std,TTT_std=x.std()[:2]
x["HOM"]=(x["HOM"]-HOM_mean)/HOM_std
x["TTT"]=(x["TTT"]-TTT_mean)/TTT_std

###Train-test split and only second index of j changes when output changes
X_train,X_test=x.iloc[:450,:2],x.iloc[450:,:2]
y_train,y_test=x.iloc[:450,2:],x.iloc[450:,2:]
X_train,X_test=X_train.values,X_test.values
y_train,y_test=y_train.values,y_test.values


###Output mean and standard deviations
y_tr_means,y_ts_means=y_train.mean(axis=0),y_test.mean(axis=0)
y_tr_stds,y_ts_stds=y_train.std(axis=0),y_test.std(axis=0)

#####Normalization of output
y_train,y_test=(y_train-y_tr_means)/y_tr_stds,(y_test-y_ts_means)/y_ts_stds


#define model
model=CBR(learning_rate=0.09,max_depth=6,n_estimators=500)
mor=MultiOutputRegressor(model)

#Fit and predict
fitted=mor.fit(X_train,y_train)
preds=fitted.predict(X_test)

#Computations
y_test,y_hat=(y_test*y_ts_stds)+y_ts_means,(preds*y_ts_stds)+y_ts_means

MAE_vals=np.mean(np.abs(y_test-y_hat),axis=0)
MSE_vals=np.mean(np.power((y_test-y_hat),2),axis=0)
