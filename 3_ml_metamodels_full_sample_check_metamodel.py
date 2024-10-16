# -*- coding: utf-8 -*-
import pandas as pd
import yaml
import numpy as np
import math
from scipy import special
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn import svm
from xgboost import XGBRegressor
import os

#folders
data_folder='./realization_datasets'
results_folder='./results'

#input filename
filename='nkm_3d_restart_6000.xlsx'

#parts for results filename
#identifier of analyzed dataset
filename_prefix=filename.replace('.xlsx','')
#identifier of current script
test_name='ml_check_metamodel'

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
avalable_methods = ['XGBRegressor', 'RandomForestRegressor', 'SVR', 'GaussianProcessRegressor']
#select metamodel
#method = 'GaussianProcessRegressor'

for i,qoi in enumerate(qois):
    writer = pd.ExcelWriter(os.path.join(results_folder,f'{test_name}_{filename_prefix}_qoi_{qoi}.xlsx'))
    
    for method in avalable_methods:
        #preparing structure for results (quality of metamodel approximation)
        regr_quality_df = pd.DataFrame(index=np.arange(0, qoi_num), columns=['qoi','MSE_train', 'R2_train', 'MSE_test', 'R2_test'])
    


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
        
        match method :
            
            case 'XGBRegressor': regr = XGBRegressor(n_estimators=100)
            case 'RandomForestRegressor' :  regr = RandomForestRegressor(max_depth=5, random_state=0)
            case 'SVR' : regr = svm.SVR(kernel='rbf')
            case 'GaussianProcessRegressor' :  regr = GaussianProcessRegressor(kernel=RBF(1.0, length_scale_bounds="fixed"))
    
        x_train, x_test, y_train, y_test = train_test_split(sample_df, qoi_trans, test_size=0.1, random_state=42)
    
        #metamodel fitting
        regr.fit(x_train, y_train) 
       
        #metamodel predictions and estimation of quality
        #for training dataset
        y_train_pr = regr.predict(x_train)  
        mse_tr = metrics.mean_squared_error(y_train,y_train_pr)
        R2_train=regr.score(x_train, y_train)
    
        #for test dataset
        y_test_pr = regr.predict(x_test) 
        mse = metrics.mean_squared_error(y_test, y_test_pr)
        R2_test=regr.score(x_test, y_test)
        
        regr_quality_df.loc[i] = [qoi, mse_tr, R2_train, mse, R2_test]
        regr_quality_df.to_excel(writer,sheet_name=f'{method}',index=False)
    writer.close()