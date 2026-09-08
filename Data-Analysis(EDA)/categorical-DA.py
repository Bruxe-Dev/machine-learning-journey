import pandas as pd 
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import matplotlib.pyplot as plt 

data = sns.load_dataset('titanic')
print(data['class'].head())

# Mapping method( Going to use it on the Class feature)

map_dict = {
    'First':0,
    'Second':1,
    'Third':2
}

data['class'] = data['class'].map(map_dict)

#Using Ordinal Encoding (through skilearn.preprocessing)

encoder = OrdinalEncoder()

cats_feats = data[['alive','alone']]

cats_feats_encoded = encoder.fit_transform(cats_feats)

# Pandas Dummies (peforming one hot encoding using pandas)

dummies = pd.get_dummies(data['who'], drop_first=True)

data = pd.concat([data.drop('who',axis=1),dummies],axis=1 )

#One-Hot-Encoding

hotEncoder = OneHotEncoder()
townEncoded = hotEncoder.fit_transform(data[['embark_town']])