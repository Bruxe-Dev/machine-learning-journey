import numpy as np 

X = np.array([
    [2,  10],
    [4,  20],
    [6,  15],
    [8,  30],
    [10, 25],
    [12, 35]
])

y = np.array([
    "No",
    "No",
    "Yes",
    "Yes",
    "Yes",
    "Yes"
])

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


left_mask = X[:, 0] < 7
right_mask = ~left_mask

y_left = y[left_mask]
y_right = y[right_mask]

print(y_left)
print(y_right)

print(information_gain(y, y_left, y_right))