import numpy as np

# Cross validation

def cv_exp(X, y, k):
    num_examples = len(y)
    idx = np.random.choice(range(num_examples),num_examples,replace = False)
    x_parts = [None]*k
    y_parts = [None]*k
    part_size = num_examples // k
    for i in range(k):
        si = i * part_size
        if i == k-1:
            ei = num_examples
        else:
            ei = (i+1) * part_size
        x_parts[i] = X[idx][si:ei]
        y_parts[i] = np.array(y)[idx][si:ei]
    results = []
    # for n in number:
    val_scores = []
    for i in range(0, k):
        X_val, Y_val = x_parts[i], y_parts[i]
        Y_val = Y_val.astype("int64").flatten()
        X_train = np.concatenate(x_parts[:i] + x_parts[i+1:], axis=0)
        Y_train = np.concatenate(y_parts[:i] + y_parts[i+1:], axis=0)
        Y_train = Y_train.astype("int64").flatten()
        # model
        dt = DecisionTree(3)
        dt.fit(X_train, Y_train, verbose = False)
        val_scores.append(dt.accuracy(X_val, Y_val))
    results.append(np.mean(val_scores))
    return results[0]