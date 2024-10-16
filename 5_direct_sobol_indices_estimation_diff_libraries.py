# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
from scipy import special
import openturns as ot
ot.Log.Show(ot.Log.NONE)
import yaml
from SALib.sample import saltelli
from SALib.analyze import sobol
from xgboost import XGBRegressor
import os

#folders
data_folder='./realization_datasets'
results_folder='./results'

#input filename
filename='nkm_3d_restart_6000.xlsx'


#problem description: parameter names and ranges, sample size

with open(file=(os.path.join(data_folder,'problem.yaml')), mode='r') as f:
        problem=yaml.safe_load(f)

#parts for results filename
#identifier of analyzed dataset
filename_prefix=filename.replace('.xlsx','')
#identifier of current script
test_name='direct_sobol_diff_libs'

#load dataset
#the parameters and qois are saved in one table, first columns=parameters, last column(s) - qois
qoi_num=1
dataset = pd.read_excel(os.path.join(data_folder,filename), index_col=0)
#extract list of parameters (in this case: all columns except last one)
parameter_names = list(dataset.columns)[:-qoi_num]
parameter_num=len(parameter_names)
sample_df=dataset[parameter_names]
sample_size=len(sample_df)

#check sample matches the problem description
print (f"Parameter number in the sample matches the problem description: {problem['num_vars']==parameter_num}")
print (f"Parameter names in the sample match the problem description: {problem['names']==parameter_names}")

#display names for parameters
parameters_dict=problem['names_display_dictionary']
#extract qois (in this case only last column)
qois=list(dataset.columns)[-qoi_num:]
qoi_df=dataset[qois]

#analysis options
##
seed = 42
#extended size of sample for xgboost
ext_sample_size_per_parameter = 4096


# structure for openturns sample generation
ot.RandomGenerator.SetSeed(seed)
marginals=[]
for i in range (parameter_num):
    marginals=marginals+[ot.Uniform(problem['bounds'][i][0],problem['bounds'][i][1])]
    
distribution = ot.ComposedDistribution(marginals, ot.IndependentCopula(parameter_num))



for qoi in qois:
    writer = pd.ExcelWriter(os.path.join(results_folder,f'{test_name}_{filename_prefix}_qoi_{qoi}.xlsx'))
    #Sobol sensitivity analysis by OpenTurns for initial dataset
    sample_size_per_parameter = int(sample_size/(parameter_num+2))
    
    input_design = ot.Sample(np.array(sample_df))
    output_sample = ot.Sample.BuildFromDataFrame(qoi_df)
    sensitivityAnalysis = ot.SaltelliSensitivityAlgorithm(input_design, output_sample, sample_size_per_parameter)
    
    Si = sensitivityAnalysis.getFirstOrderIndices(0)
    S1 = [Si[i] for i in range(parameter_num)]
    Si = sensitivityAnalysis.getTotalOrderIndices(0)
    ST = [Si[i] for i in range(parameter_num)]
        
    si_df=pd.DataFrame({'S1' : S1, 'ST' : ST, 'parameter' : parameter_names})
    si_df['parameter_nice']=si_df['parameter'].map(parameters_dict)
    si_df.to_excel(writer, sheet_name = "OpenTURNS_from_init_data")
    
    
    #Sobol Sensitivity analysis by SAlib for initial dataset
    input_samples = saltelli.sample(problem, sample_size_per_parameter)
    
    Si = sobol.analyze(problem, qoi_df[qoi].values, calc_second_order=False)
    si_df=pd.concat({k: pd.DataFrame(v) for k, v in Si.items()}, axis=1)
    si_df.columns = si_df.columns.droplevel(level=1)
    si_df['parameter']=sample_df.columns
    si_df['parameter_nice']=si_df['parameter'].map(parameters_dict)
    si_df.to_excel(writer, sheet_name = "SAlib_from_init_data")
    
    #openTURNS Sobol' sensitivity analysis by OpenTURNS for XGBoost approximation
    #generate sample for OpenTURNS
    si_exp = ot.SobolIndicesExperiment(distribution, ext_sample_size_per_parameter)
    ext_input_design = si_exp.generate()
    ext_sample_df = ext_input_design.asDataFrame()
    ext_sample_df.columns = parameter_names
    #generate sample for SALib
    ext_input_samples = saltelli.sample(problem, ext_sample_size_per_parameter)
    
    
    ##
    #Model outputs transformation for XGBoost using algorithm from T.Mara (similar to quantile transformation)
    cdf = np.array([(i+0.5)/sample_size for i in range(sample_size)])
    y_n=np.zeros(sample_size)
    y=qoi_df[qoi].to_numpy()
    y_r = np.sort(y)
    Iy=sorted(range(len(y)), key=lambda k: y[k])
    for j in range(sample_size):
            y_n[Iy[j]] = math.sqrt(2.0)*special.erfinv(2*cdf[j]-1) 
          
    qoi_trans = y_n
    y_n_r = np.sort(y_n) #for inverse transformation
        
    #XGBoost initialization and fitting
    regr = XGBRegressor(n_estimators=100)
    # X_train, X_test, y_train, y_test = train_test_split(scaled_in, scaled_tr, test_size=0.1, random_state=42)
    # regr.fit(X_train, y_train)
    regr.fit(sample_df, qoi_trans) 
        
    y_OT = regr.predict(ext_sample_df.to_numpy()) 
    y_SALib = regr.predict(ext_input_samples)
      
    #Inverse transformation the XGBoost prediction for OpenTURNS sample
    yy = np.zeros(len(y_OT))
    for j in range(len(y_OT)):
        Ind = np.where(y_n_r >= y_OT[j])[0]
        if (len(Ind) > 1): ind = Ind[0]
        if ((ind == 0) | (ind >= len(y_n_r))):
              yy[j] = y_r[ind]
        else:
              yy[j] = y_r[ind-1]+((y_r[ind]-y_r[ind-1])/(y_n_r[ind]-y_n_r[ind-1]))*(y_OT[j]-y_n_r[ind-1])
                
    ext_result_df = pd.DataFrame(yy, columns = [qoi])
    ext_output_sample = ot.Sample.BuildFromDataFrame(ext_result_df.loc[:])
    
    sensitivityAnalysis = ot.SaltelliSensitivityAlgorithm(ext_input_design, ext_output_sample, ext_sample_size_per_parameter)
    
    Si = sensitivityAnalysis.getFirstOrderIndices(0)
    S1 = [Si[i] for i in range(parameter_num)]
    Si = sensitivityAnalysis.getTotalOrderIndices(0)
    ST = [Si[i] for i in range(parameter_num)]
        
    si_df=pd.DataFrame({'S1' : S1, 'ST' : ST, 'parameter' : parameter_names})
    si_df['parameter_nice']=si_df['parameter'].map(parameters_dict)
    si_df.to_excel(writer, sheet_name = "OpenTURNS_from_XGBoost")
    
    
    #Inverse transformation of the XGBoost prediction for SALib samples
    yy = np.zeros(len(y_SALib))
    for j in range(len(y_SALib)):
        Ind = np.where(y_n_r >= y_SALib[j])[0]
        if (len(Ind) > 1): ind = Ind[0]
        if ((ind == 0) | (ind >= len(y_n_r))):
              yy[j] = y_r[ind]
        else:
              yy[j] = y_r[ind-1]+((y_r[ind]-y_r[ind-1])/(y_n_r[ind]-y_n_r[ind-1]))*(y_SALib[j]-y_n_r[ind-1])

    Si = sobol.analyze(problem, yy, calc_second_order=False)
    si_df=pd.concat({k: pd.DataFrame(v) for k, v in Si.items()}, axis=1)
    si_df.columns = si_df.columns.droplevel(level=1)
    si_df['parameter']=parameter_names
    si_df['parameter_nice']=si_df['parameter'].map(parameters_dict)
    si_df.to_excel(writer, sheet_name = "SAlib_from_XGBoost")
    writer.close()