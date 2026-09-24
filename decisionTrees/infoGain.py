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

best_gain = -1
best_feature = None
best_threshold = None

for feature in range(X.shape[1]):

    values = np.sort(np.unique(X[:, feature]))
    thresholds = (values[:-1] + values[1:]) / 2

    for threshold in thresholds:

        left_mask = X[:, feature] < threshold
        right_mask = ~left_mask

        y_left = y[left_mask]
        y_right = y[right_mask]

        gain = information_gain(y, y_left, y_right)

        print(
            "Feature:", feature,
            "Threshold:", threshold,
            "Gain:", gain
        )