import seaborn as sb 
import matplotlib.pyplot as plt 

data = sb.load_dataset("titanic")

print(data.describe().transpose())