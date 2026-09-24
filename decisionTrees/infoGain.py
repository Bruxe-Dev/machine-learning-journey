import numpy as np 

def entropy(y):
    classes,count = np.unique(y,return_counts=True)
    probabilities = count/len(y)

    return -np.sum(probabilities * np.log2(probabilities))

y = np.array(['yes','yes','yes','yes','yes','yes'])
print(entropy(y))