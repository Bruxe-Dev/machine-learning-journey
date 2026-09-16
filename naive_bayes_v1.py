import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sb 


df = pd.read_csv("data/spam.csv",encoding='latin-1')[['v1','v2']]

print(df.head())

df.columns = ['label','message']

print(df.head())
X = df['message']
Y = df['label']

print(Y.value_counts())