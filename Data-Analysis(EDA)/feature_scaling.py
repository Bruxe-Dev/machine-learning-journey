from seaborn import load_dataset
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler, StandardScaler

data = load_dataset('tips')
print(data.head())

# MinMaxScaler (Normalization)
num_feats = data[['total_bill','tip','size']]

scaler = MinMaxScaler()

num_feats = scaler.fit_transform(num_feats)

num_scaled_df = pd.DataFrame(data=num_feats,columns=num_feats.columns)
print(num_scaled.head())

#Standardisation

stScaler = StandardScaler()

st_num_feats = stScaler.fit_transform(num_feats)
std_numFeats_df = pd.DataFrame(st_num_feats,columns=st_num_feats.colums())