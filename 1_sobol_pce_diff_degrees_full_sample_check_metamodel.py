# -*- coding: utf-8 -*-

import pandas as pd
import openturns as ot
import openturns.viewer as viewer
ot.Log.Show(ot.Log.NONE)
import yaml
from sklearn.model_selection import train_test_split
import pylab as pl
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
test_name='pce_check_metamodel'

#load dataset
#the parameters and qois are saved in one table, first columns=parameters, last column(s) - qois
qoi_num=1
dataset = pd.read_excel(os.path.join(data_folder,filename), index_col=0)
#extract list of parameters (in this case: all columns except last one)
parameter_names = list(dataset.columns)[:-qoi_num]
parameter_num=len(parameter_names)
sample_df=dataset[parameter_names]
sample_size=len(sample_df)

#problem description: parameter names and ranges, sample size
#with open(file=('problem.yaml'), mode='r') as f:
#        problem=yaml.safe_load(f)
#check sample matches the problem description
#print (f"Parameter number in the sample matches the problem description: {problem['num_vars']==parameter_num}")
#print (f"Parameter names in the sample match the problem description: {problem['names']==parameter_names}")

#extract qois (in this case only last column)
qois=list(dataset.columns)[-qoi_num:]
qoi_df=dataset[qois]


#analysis options
##
#sparse_or_full='full'
sparse_or_full='sparse'
##
#maximal polynomial order
maxDegree = 7
##
seed = 42 

#split dataset into train and test for checking metamodel quality
sample_df_train, sample_df_test, qoi_df_train, qoi_df_test = train_test_split(sample_df, qoi_df, test_size=0.1,random_state=seed,shuffle=True)


#PCE metamodel functions
#construct metamodel
def ComputeLeastSquaresChaos(
    inputTrain, outputTrain, multivariateBasis, totalDegree, myDistribution,sparse_or_full
):
###If full the the basis size is computed from the total degree.
#The next lines use the LeastSquaresStrategy class with default parameters (the default is the PenalizedLeastSquaresAlgorithmFactory class). 
#This creates a full polynomial chaos expansion, i.e. we keep all the candidate coefficients produced by the enumeration rule. 
# In order to create a sparse polynomial chaos expansion, we use LeastSquaresMetaModelSelectionFactory class instead.
    if sparse_or_full=='full':
        #full
        projectionStrategy = ot.LeastSquaresStrategy()
    else:
        ###sparse
        selectionAlgorithm = ot.LeastSquaresMetaModelSelectionFactory()
        projectionStrategy = ot.LeastSquaresStrategy(selectionAlgorithm)
    
    enumfunc = multivariateBasis.getEnumerateFunction()
    P = enumfunc.getStrataCumulatedCardinal(totalDegree)
    adaptiveStrategy = ot.FixedStrategy(multivariateBasis, P)
    chaosalgo = ot.FunctionalChaosAlgorithm(
        inputTrain, outputTrain, myDistribution, adaptiveStrategy, projectionStrategy
    )
    chaosalgo.run()
    result = chaosalgo.getResult()
    return result

def computeSparsityRate(multivariateBasis, totalDegree, chaosResult):
    """Compute the sparsity rate, assuming a FixedStrategy."""
    # Get P, the maximum possible number of coefficients
    enumfunc = multivariateBasis.getEnumerateFunction()
    P = enumfunc.getStrataCumulatedCardinal(totalDegree)
    # Get number of coefficients in the selection
    indices = chaosResult.getIndices()
    nbcoeffs = indices.getSize()
    # Compute rate
    sparsityRate = 1.0 - nbcoeffs / P
    return sparsityRate

def computeR2Chaos(chaosResult, inputTest, outputTest):
    """Compute the R2 of a chaos."""
    metamodel = chaosResult.getMetaModel()
    val = ot.MetaModelValidation(outputTest, metamodel(inputTest))
    R2 = val.computeR2Score()[0]
    R2 = max(R2, 0.0)  # We are not lucky every day.
    return R2

def printChaosStats(multivariateBasis, chaosResult, inputTest, outputTest, totalDegree):
    """Print statistics of a chaos."""
    sparsityRate = computeSparsityRate(multivariateBasis, totalDegree, chaosResult)
    R2 = computeR2Chaos(chaosResult, inputTest, outputTest)
    
    metamodel = chaosResult.getMetaModel()
    val = ot.MetaModelValidation(outputTest, metamodel(inputTest))
    graph = val.drawValidation().getGraph(0, 0)
    legend1 = "D=%d, R2=%.2f%%" % (totalDegree, 100 * R2)
    graph.setLegends(["", legend1])
    graph.setLegendPosition("upper left")
    print(
        "Degree=%d, R2=%.2f%%, Sparsity=%.2f%%"
        % (totalDegree, 100 * R2, 100 * sparsityRate)
    )
    stats=[totalDegree, 100 * R2, 100 * sparsityRate]
    return graph, stats

#convert sample dataframes into openturn sample objects
x_train = ot.Sample(sample_df_train.values.tolist())
x_test = ot.Sample(sample_df_test.values.tolist())

distribution = ot.FunctionalChaosAlgorithm.BuildDistribution(x_train)
inputDimension = distribution.getDimension()


#construct multivariate basis

#univariate orthogonal polynomial basis for each marginal.
coll = [
    ot.StandardDistributionPolynomialFactory(distribution.getMarginal(i))
    for i in range(inputDimension)
]
enumerateFunction = ot.LinearEnumerateFunction(inputDimension)

#combine into multivariateBasis
productBasis = ot.OrthogonalProductPolynomialFactory(coll, enumerateFunction)

marginalDistributionCollection = [
    distribution.getMarginal(i) for i in range(inputDimension)
]
multivariateBasis = ot.OrthogonalProductPolynomialFactory(
    marginalDistributionCollection
)

fig = pl.figure(figsize=(25, 4))

writer = pd.ExcelWriter(os.path.join(results_folder,f'{test_name}_{filename_prefix}_{sparse_or_full}_pce_max_degree_{maxDegree}.xlsx'))
for qoi in qois:
    r2_stats=[]
    y_train= [[i] for i in qoi_df_train[qoi].to_list()]
    y_test=ot.Sample([[i] for i in qoi_df_test[qoi].to_list()])

     
    ot.RandomGenerator.SetSeed(seed)
    #for each polynomial order (degree)
    for totalDegree in range(1, maxDegree + 1):
        #compute Polynomial Chaos expansion given input and output samples, polynomial order, distribution properties, either use full or sparce PCE
        result = ComputeLeastSquaresChaos(
            x_train, y_train, multivariateBasis, totalDegree, distribution,sparse_or_full
        )
        #output PCE model statistics via table and plot
        graph,r2_stats_by_degree = printChaosStats(
            multivariateBasis, result,x_test, y_test, totalDegree
        )
        r2_stats.append(r2_stats_by_degree)
        ax = fig.add_subplot(1, maxDegree, totalDegree)
        _ = ot.viewer.View(graph, figure=fig, axes=[ax])
        pl.suptitle("Metamodel validation")
    #save statistics
    r2_stats_df=pd.DataFrame(r2_stats,columns=['degree','R2','Sparsity'])
    r2_stats_df.to_excel(writer,sheet_name=f'qoi {qoi}',index=False)
writer.close()