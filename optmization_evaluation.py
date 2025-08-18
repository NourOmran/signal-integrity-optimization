import pandas as pd 
import numpy as np
import optuna
import joblib   


par = pd.read_csv("optimization preemphasis/optuna_results.csv")
pd1 = par.drop_duplicates() 


print("Number of unique trials:", pd1.shape)

pd2 =  pd.read_csv("optimization preemphasis/high_eyeSNR_data.csv")

pd2 = pd2.drop(columns=["Cable_Name"])
pd2= pd2.iloc[:59, :]
print( pd2)


parameters1 = pd1[["preW", "mainW", "postW", "dpre", "dpost", "Rdrv"]]
parameters2 = pd2[["preW", "mainW", "postW", "dpre", "dpost", "Rdrv"]]

common = pd.merge(parameters1, parameters2, how="inner")
print(f"Number of parameters1 found in parameters2: {len(common)}")