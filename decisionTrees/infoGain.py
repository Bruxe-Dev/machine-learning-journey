import numpy as np 

def entropy(y):
    classes,count = np.unique(y,return_counts=True)
    probabilities = count/len(y)

    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(parent,left,right):
    parent_entropy = entropy(parent)
    left_entropy = entropy(left)
    right_entropy = entropy(right)

    left_weight = len(left)/len(parent)
    right_weight = len(right)/len(parent)

    split_entropy = (
        left_weight * left_entropy
        + right_weight * right_entropy
    )

    gain = parent_entropy - split_entropy

    return gain

y = np.array(['yes','no','yes','no','yes','no'])
left = np.array(['yes','yes','yes','yes'])
right = np.array(['no','no','no','no','no',])

print(information_gain(y,left,right))