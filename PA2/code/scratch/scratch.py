'''
from DataLoader import DataLoader
import numpy as np

### load and extract relevant quantities from training data.
train_path = "../data/pnp-train.txt"
loader = DataLoader()
X_train_raw, Y_train_raw = loader.load(train_path)
unique_chars = loader.unique_chars
unique_classes = loader.unique_classes
num_of_classes = loader.num_of_classes
num_of_train = loader.num_of_datapoints
num_of_features = len(unique_chars)

### map features to indexes
feature_to_index = {} # A dictionary that maps features f(Xi)s names to their corresponding indexes in the numpy array F.
for i, char in enumerate(unique_chars):
    feature_to_index[char] = i

### map classes to indexes
class_to_index = {} # A dictionary that maps class label to their corresponding indexes in the numpy array Y.
for i, cls in enumerate(unique_classes):
    class_to_index[cls] = i
    
### create a m x 1 classes numpy array Y, where m is the number of training examples.
Y = np.empty((num_of_train), dtype=np.int8)
for i in range(num_of_train):
    Y[i] = class_to_index[Y_train_raw[i]]
print(Y.shape)

### generate a f x m features numpy array F, where m is the number of training examples, f is the number of features for each training example Xi.
F = np.zeros((num_of_features, num_of_train))
for i in range(num_of_train):
    for char in X_train_raw[i]:
        F[feature_to_index[char], i] += 0.1

### initialize a f x y weights numpy array W with random values, where f is the number of features for each training example Xi, y is the number of classes.
W = np.random.rand(num_of_features, num_of_classes)

### implement the objective function 0.0
# S = W^T * F is a y x m select matrix in which we iterate through each column i = 1...m and select the element on the y-th row where y is the correct class label for input Xi.
# We sum all these selected m elements to get the first term in the objective.
first_term = 0.0
S = W.T @ F
for i in range(num_of_train):
    first_term += S[Y[i], i]

# ES = exp(S). I is a 1 * m intermediate numpy array containing the sum of each ith column in ES for i = 1...m. I contains the expression after the "log" in the second_term. 
second_term = 0.0
ES = np.exp(S)
I = np.sum(ES, axis=0)
second_term = np.sum(np.log(I))

# regularization_term = lambda * ||W||^2, where lambda is a hyperparameter to tune. Let's first set it to 0.1 and later tune this value.
lam = 1
regularization_term = lam * np.linalg.norm(W)

# obtaining the full expression for the objective function
objective = first_term - second_term - regularization_term
print(objective)

### implement the objective function 1.0
numerator = np.empty(num_of_train)
denominator = np.empty(num_of_train)
for i in range(num_of_train):
    numerator[i] = ES[Y[i], i]
    denominator[i] = np.sum(ES[:, i], axis=0)
objective = np.sum(np.log(numerator / denominator))
objective -= regularization_term
print(objective)

### implement the objective function 2.0
objective = 0
for i in range(num_of_train):
    numerator = np.exp(W[:, Y[i]].T @ F[:, i])
    #print(numerator)
    denominator = np.sum(np.exp(W.T @ F[:, i]), axis=0)
    #print(denominator)
    fraction = numerator / denominator
    objective += np.log(fraction)
objective -= regularization_term
print(objective)

### implement the objective function 3.0
objective = 0
for i in range(num_of_train):
    numerator = ES[Y[i], i]
    denominator = np.sum(ES[:, i], axis=0)
    fraction = numerator / denominator
    objective += np.log(fraction)
objective -= regularization_term
print(objective)


### compute the gradient (slow)
dLdW = np.empty((num_of_features, num_of_classes))
for y in range(num_of_classes):
    first_term = np.zeros(num_of_features)
    second_term = np.zeros(num_of_features)
    regularization_term = 2 * lam * W[:, y]
    for i in range(num_of_train):
        if Y[i] == y:
            first_term = first_term + F[:, i]
        prob = ES[y, i] / np.sum(ES[:, i], axis=0)
        second_term = second_term + prob * F[:, i]
    gradient = first_term - second_term - regularization_term;
    dLdW[:, y] = gradient
print(np.linalg.norm(dLdW))

### compute the gradient (fast)
dLdW = np.empty((num_of_features, num_of_classes))
# L is a m x y select matrix where each row i contains only one '1' on the jth column indicating the ith training example is of class j.
L = np.zeros((num_of_train, num_of_classes))
for i in range(num_of_train):
    L[i, Y[i]] = 1
first_term = F @ L

second_term = np.empty((num_of_features, num_of_classes))
denominator = np.sum(ES, axis=0)
fraction = ES / denominator
second_term = F @ fraction.T

regularization_term = 2 * lam * W

dLdW = first_term - second_term - regularization_term
print(np.linalg.norm(dLdW))


# implement gradient ascent
t = 0
alpha = 0.1 # learning rate
epsilon = 1 # convergence criterion
while True:
    ### compute the gradient (fast)
    dLdW = np.empty((num_of_features, num_of_classes))  
    # first_term
    L = np.zeros((num_of_train, num_of_classes)) # L is a m x y select matrix where each row i contains only one '1' on the jth column indicating the ith training example is of class j.
    for i in range(num_of_train):
        L[i, Y[i]] = 1
    first_term = F @ L
    #second_term
    ES = np.exp(W.T @ F)
    second_term = np.empty((num_of_features, num_of_classes))
    denominator = np.sum(ES, axis=0)
    fraction = ES / denominator
    second_term = F @ fraction.T
    # regularization_term
    regularization_term = 2 * lam * W
    # combine
    dLdW = first_term - second_term - regularization_term
 
    if t % 20 == 1:
        print(np.linalg.norm(W))
        ### implement the objective function 1.0
        numerator = np.empty(num_of_train)
        denominator = np.empty(num_of_train)
        ES = np.exp(W.T @ F)
        for i in range(num_of_train):
            numerator[i] = ES[Y[i], i]
            denominator[i] = np.sum(ES[:, i], axis=0)
        objective = np.sum(np.log(numerator / denominator))
        regularization_term = lam * np.linalg.norm(W)
        objective -= regularization_term
        print("objective:" + str(objective))
        
    # gradient ascent  
    t = t + 1
    old_W = W.copy()
    W = W + alpha / np.sqrt(t) * dLdW
        
    if np.linalg.norm(W - old_W) < epsilon:
        break
print(W)

### evaluation on the training set
Y_pred_train = np.argmax(W.T @ F, axis=0)
train_acc = float(np.sum(Y_pred_train == Y)) / num_of_train
print(train_acc) 

### evaluation on the validation set
val_path = "../data/pnp-validate.txt"
val_loader = DataLoader()
X_val, Y_val_raw = val_loader.load(val_path)
num_of_val = val_loader.num_of_datapoints

Y_val = np.empty((num_of_val), dtype=np.int8)
for i in range(num_of_val):
    Y_val[i] = class_to_index[Y_val_raw[i]]

F_val = np.zeros((num_of_features, num_of_val))
for i in range(num_of_val):
    for char in X_val[i]:
        if char in feature_to_index.keys():
            F_val[feature_to_index[char], i] += 0.1

Y_pred = np.argmax(W.T @ F_val, axis=0)
val_acc = float(np.sum(Y_pred == Y_val)) / num_of_val
print(val_acc)
'''
'''
import numpy as np
test_path = "../data/pnp-test.txt"
with open(test_path, 'r', encoding='iso-8859-1') as t:
    for line in t:
        record = line.strip('\n').split('\t')
        print(record)
        print(record[0])
        break
'''
'''
def get_n_grams(L, n):
    unique_n_grams = []
    for s in L:
        words = s.split()
        for i in range(len(words) - n + 1):
            n_gram = words[i:i+n]
            if n_gram not in unique_n_grams:
                unique_n_grams.append(n_gram)
    return sorted(unique_n_grams)

unique_words = get_n_grams(X_train, 1)
#unique_bigrams = get_n_grams(X_train, 2)
print(len(unique_words))
print(unique_words[0:100])
#print(len(unique_bigrams))
'''
'''
for i, feature in enumerate(unique_unigrams):
    print(feature)
    d[feature] = i
print(d)

for i in range(len(words) - n + 1):
    unigram = words[i:i+n]
    if unigram in d.keys():
        print(str(unigram) + "is in d\n")
'''    

class A(object):
    def __init__(self):
        self.n = 1
    def get_n(self):
        print(self.n)
        
class B(A):
    def __init__(self):
        super().__init__()
        self.n = 2

        
a = A()
a.get_n()
b = B()
b.get_n()


