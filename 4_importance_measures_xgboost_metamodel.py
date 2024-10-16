# -*- coding: utf-8 -*-
import pandas as pd
import yaml
import numpy as np
import math
from scipy import special
from sklearn import metrics
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import shap
import os

#folders
data_folder='./realization_datasets'
results_folder='./results'

#input filename
filename='nkm_3d_restart_6000.xlsx'

#problem description: parameter names and ranges, sample size

#parts for results filename
#identifier of analyzed dataset
filename_prefix=filename.replace('.xlsx','')
#identifier of current script
test_name='ml_imp_measures'

#load dataset
#the parameters and qois are saved in one table, first columns=parameters, last column(s) - qois
qoi_num=1
dataset = pd.read_excel(os.path.join(data_folder,filename), index_col=0)
#extract list of parameters (in this case: all columns except last one)
parameter_names = list(dataset.columns)[:-qoi_num]
parameter_num=len(parameter_names)
sample_df=dataset[parameter_names]
sample_size=len(sample_df)

#problem description: parameter names and ranges
#with open(file=(os.path.join(data_folder,'problem.yaml')), mode='r') as f:
#        problem=yaml.safe_load(f)
#check sample matches the problem description
#print (f"Parameter number in the sample matches the problem description: {problem['num_vars']==parameter_num}")
#print (f"Parameter names in the sample match the problem description: {problem['names']==parameter_names}")



#extract qois (in this case only last column)
qois=list(dataset.columns)[-qoi_num:]
qoi_df=dataset[qois]


#metamodel options
#avalable_methods = ['XGBRegressor', 'RandomForestRegressor', 'SVR', 'GaussianProcessRegressor']
#select metamodel
method = 'XGBRegressor'




for i,qoi in enumerate(qois):
    writer = pd.ExcelWriter(os.path.join(results_folder,f'{test_name}_{filename_prefix}_metamodel_{method}.xlsx'))
    #preparing structure for results (quality of metamodel approximation)
    result_df = result = pd.DataFrame(index=np.arange(0, 3), columns = ['method']+parameter_names)

    #transformation of the output into standard normal variables
    #transformation of inputs and outputs using algorithm from T.Mara (similar to quantile transformation)
    cdf = np.array([(i+0.5)/sample_size for i in range(sample_size)])
    y_n=np.zeros(sample_size)
    y=qoi_df[qoi].to_numpy()
    y_r = np.sort(y)
    Iy=sorted(range(len(y)), key=lambda k: y[k])
    for j in range(sample_size):
        y_n[Iy[j]] = math.sqrt(2.0)*special.erfinv(2*cdf[j]-1) 
      
    qoi_trans = y_n
    
    #XGBoost initialization 
    regr = XGBRegressor(n_estimators=100)
    


    #metamodel fitting
    #split into train and test sample
    #x_train, x_test, y_train, y_test = train_test_split(sample_df, qoi_trans, test_size=0.1, random_state=42)
    #fit train dataset
    #regr.fit(x_train, y_train)
    
    #fit whole dataset without splitting
    x=sample_df
    y=qoi_trans
    regr.fit(sample_df, qoi_trans)
    
    
    
    #Gini importance
    importance=[regr.feature_importances_[i] for i in range(parameter_num)] 
    importance.insert(0,'Gini')
    result_df.loc[0] =  importance
  
#permutation importance
    perm_importance = permutation_importance(regr, x, y)
    importance=[perm_importance.importances_mean[i] for i in range(parameter_num)]
    importance.insert(0,'Permutations')
    result_df.loc[1] = importance

#SHAP importance
    explainer = shap.TreeExplainer(regr)
    shap_values = explainer.shap_values(x)
    sh = [np.mean([abs(shap_values[k,j]) for k in range(len(x))]) for j in range(parameter_num)]
    sh.insert(0,'SHAP')
    result_df.loc[2] = sh
    result_df.to_excel(writer,sheet_name=f'{qoi}',index=False)
    writer.close()