import pandas as pd 
import seaborn as sns
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