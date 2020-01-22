import numpy as np

class UnigramModel(object):
    def __init__(self, train_path, val_path, test_path, lam=1.0):
        ### variable declarations
        self.X_train = []
        self.Y_train_raw = []
        self.X_val = []
        self.Y_val_raw = []
        self.X_test = []
        
        self.Y_train = None
        self.Y_val = None
        self.F_train = None
        self.F_val = None
        self.F_test = None
        self.W = None
        
        self.feature_to_index = {}
        self.class_to_index = {}
        self.num_of_train = 0
        self.num_of_val = 0
        self.num_of_test = 0
        self.num_of_features = 0
        self.num_of_classes = 0
        self.lam = lam
        
        ### initialization functions
        self.LoadData(train_path, val_path, test_path)
        self.getFeatureMap()
        self.getClassMap()
        self.generateYMatrices()
        self.generateFMatrices()
        self.initializeWMatrix()
        
    def LoadData(self, train_path, val_path, test_path):
        # load training data
        with open(train_path, 'r', encoding='iso-8859-1') as t:
            for line in t:
                record = line.strip('\n').split('\t')
                self.Y_train_raw.append(record[0])
                self.X_train.append(record[1]) 
        self.num_of_train = len(self.X_train)
 
        # load validation data
        with open(val_path, 'r', encoding='iso-8859-1') as t:
            for line in t:
                record = line.strip('\n').split('\t')
                self.Y_val_raw.append(record[0])
                self.X_val.append(record[1])
        self.num_of_val = len(self.X_val)
        
        with open(test_path, 'r', encoding='iso-8859-1') as t:
            for line in t:
                record = line.strip('\n').split('\t')
                self.X_test.append(record[0])
        self.num_of_test = len(self.X_test)
        
    def getFeatureMap(self):
        ''' 
            Map unigram features to indexes.
        '''
        unique_chars = sorted(list({l for word in self.X_train for l in word}))
        self.num_of_features = len(unique_chars)
        for i, char in enumerate(unique_chars):
            self.feature_to_index[char] = i
        
    def getClassMap(self):
        '''
            Map class labels to indexes.
        '''
        unique_classes = sorted(list(set(self.Y_train_raw)))
        self.num_of_classes = len(unique_classes)
        for i, cls in enumerate(unique_classes):
            self.class_to_index[cls] = i
            
    def generateYMatrices(self):
        '''
            Generate Y_train of shape (num_of_train,) of class labels for the training examples
            and      Y_val of shape (num_of_val,) of class labels for the validation examples.
        '''
        self.Y_train = np.empty((self.num_of_train), dtype=np.int8)
        self.Y_val = np.empty((self.num_of_val), dtype=np.int8)
        
        for i in range(self.num_of_train):
            self.Y_train[i] = self.class_to_index[self.Y_train_raw[i]]
        for i in range(self.num_of_val):
            self.Y_val[i] = self.class_to_index[self.Y_val_raw[i]]
    
    def generateFMatrices(self):
        '''
            Generate F_train of shape (num_of_features, num_of_train) of features for the training examples,
                    F_val of shape (num_of_features, num_of_val) of features for the validation examples,
            and      F_test of shape (num_of_features, num_of_val) of features for the test examples.
        '''
        self.F_train = np.zeros((self.num_of_features, self.num_of_train))
        self.F_val = np.zeros((self.num_of_features, self.num_of_val))
        self.F_test = np.zeros((self.num_of_features, self.num_of_test))
        
        for i in range(self.num_of_train):
            for char in self.X_train[i]:
                self.F_train[self.feature_to_index[char], i] += 0.1
        
        for i in range(self.num_of_val):
            for char in self.X_val[i]:
                if char in self.feature_to_index.keys():
                    self.F_val[self.feature_to_index[char], i] += 0.1
                    
        for i in range(self.num_of_test):
            for char in self.X_test[i]:
                if char in self.feature_to_index.keys():
                    self.F_test[self.feature_to_index[char], i] += 0.1
        
    def initializeWMatrix(self):
        '''
           Initialize W of shape (num_of_features, num_of_classes) containing the weights for each class label with random numbers.
        '''
        self.W = np.random.rand(self.num_of_features, self.num_of_classes)
        
    def getObjective(self, dataset="train"):
        '''
            Compute the objective function with:
                1. The current input feature matrix F, 
                2. The current input label matrix Y,
                3. The current weights matrix W, 
                4. Also adding L2 regularization.
            
            Parameter:
                dataset: A string. Either "train" or "val".
            
            Returns: 
                objective: A scalar.
        '''   
        if dataset == "train":
            Y = self.Y_train
            F = self.F_train
        
        else:
            Y = self.Y_val
            F = self.F_val
            
        W = self.W
        lam = self.lam
        m = Y.shape[0]
        ES = np.exp(W.T @ F)

        numerator = np.empty(m)
        denominator = np.empty(m)
        
        for i in range(m):
            numerator[i] = ES[Y[i], i]
            denominator[i] = np.sum(ES[:, i], axis=0)
        objective = np.sum(np.log(numerator / denominator))
        regularization_term = lam * np.linalg.norm(W)
        objective -= regularization_term
        
        return objective
        
    def computeGradient(self):
        '''
            Compute the gradient based on the training data for one iteration.
            
            Returns:
                dLdW: A num_of_features x num_of_classes numpy array.
        '''
        lam = self.lam
        W = self.W
        F = self.F_train
        Y = self.Y_train
        m = self.num_of_train
        f = self.num_of_features
        c = self.num_of_classes
        dLdW = np.empty((f, c))
        ES = np.exp(W.T @ F)

        L = np.zeros((m, c)) # L is a m x c select matrix where each row i contains only one '1' on the jth column indicating the ith training example is of class j.
        for i in range(m):
            L[i, Y[i]] = 1
        first_term = F @ L
        
        second_term = np.empty((f, c))
        denominator = np.sum(ES, axis=0)
        fraction = ES / denominator
        second_term = F @ fraction.T

        regularization_term = 2 * lam * W

        dLdW = first_term - second_term - regularization_term
        return dLdW

    def train(self, alpha=0.1, epsilon=1.0):
        '''
            Use gradient ascent to train the weights to maximize the objective function.
            
            Parameters:
                alpha: The learning rate.
                epsilon: The tolerance for the difference between the L2 norms of old_W and new_W as the stopping criterion.
        '''
        t = 0
        while True:
            t = t + 1
            dLdW = self.computeGradient()  
            old_W = self.W.copy()
            self.W = self.W + alpha / np.sqrt(t) * dLdW
            if t % 100 == 1:
                print(self.getObjective())
            if np.linalg.norm(self.W - old_W) < epsilon:
                break
    
    def predict(self, dataset="test"):
        '''
            Predict class labels based on the model.
            
            Parameter:
                dataset: A string. Either "train", "val", or "test".
                
            Returns:
                Y_pred: A (num_of_datapoints,) numpy array of predicted labels.               
        '''
        if dataset == "train":
            F = self.F_train
        elif dataset == "val":
            F = self.F_val
        else:
            F = self.F_test
        
        W = self.W
        Y_pred = np.argmax(W.T @ F, axis=0)
        return Y_pred
        
    def evaluate(self, dataset="val"):
        '''
            Evaluate the model in terms of accuracy on the dataset.
            
            Parameter:
                dataset: A string. Either "train" or "val"
                
            Returns:
                acc: A scalar. The predicting accuracy of the model.
        '''
        if dataset == "train":
            Y = self.Y_train
            m = self.num_of_train
        else:
            Y = self.Y_val
            m = self.num_of_val
            
        Y_pred = self.predict(dataset=dataset)
        acc = float(np.sum(Y_pred == Y)) / m
        return acc
        
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    