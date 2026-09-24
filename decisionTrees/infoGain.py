import numpy as np 

y = np.array([
    "yes", "No", "No", "No", "No", "No",
    "No", "No", "No", "No"
])

categories, count = np.unique(y,return_counts=True)

probabilities = count/len(y)

prob_w = probabilities * np.log2(probabilities)

entropy = -1 * np.sum(prob_w)

print(entropy)