from seaborn import load_dataset
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler

data = load_dataset('tips')
print(data.head())

num_feats = data[['total_bill','tip','size']]

scaler = MinMaxScaler()

num_feats = scaler.fit_transform(num_feats)

num_scaled_df = pd.DataFrame(data=num_feats,columns=num_feats.columns)
print(num_scaled.head())