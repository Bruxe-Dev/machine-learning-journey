import numpy as np 

y = np.array([
    "Yes", "Yes", "Yes", "Yes", "Yes", "Yes",
    "No", "No", "No", "No"
])

categories,count = np.unique(y,return_counts=True)
print(f"Categories: {categories}\n Count: {count}")

probabilities = count/len(y)

gini = 1 - np.sum(probabilities ** 2)

print(gini)