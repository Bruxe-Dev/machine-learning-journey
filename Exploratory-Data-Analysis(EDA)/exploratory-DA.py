import seaborn as sb 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd

data = sb.load_dataset("titanic")

print(data['class'].value_counts())

p_class = data[['pclass','class']]
print(p_class.head())

# print(data.describe())
# sb.histplot(data=data,x='age',bins=20,binrange=(0,80),shrink=0.85)
# plt.xticks(range(0,86,5))

# SHow passengers who were less than 20 years

less20 = data[data['age'] < 20]
print(less20.head())

# The categories usding the who column and represent it using a pie chart

categories = data['who']
categories.value_counts().plot(kind='pie')
plt.show()

# Missing values 
print(data.isnull().sum())

# most_common = data['embarked'].mode()[0]
# data['embarked'].fillna(most_common,inplace=True)

# print(data['embarked'].isnull().sum())

# scatter plotting
sb.scatterplot(data=data,x='age',y='fare',hue='class',style='survived')
plt.show()