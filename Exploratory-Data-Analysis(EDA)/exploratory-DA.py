import seaborn as sb 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd

data = sb.load_dataset("titanic")

print(data['survived'].value_counts())