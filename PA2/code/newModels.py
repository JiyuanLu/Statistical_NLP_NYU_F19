import numpy as np
 
class UnigramModel(object):
    def __init__(self, train_path, val_path, test_path):
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
        self.index_to_class = {}
        self.num_of_train = 0
        self.num_of_val = 0
        self.num_of_test = 0
        self.num_of_features = 0
        self.num_of_classes = 0
        self.feature_weight = 0.1
        self.n = 1
        self.lam = 1.0
        
        ### initialization functions
        self.LoadData(train_path, val_path, test_path)
        self.getFeatureMap()
        self.getClassMap()
        self.generateYMatrices()
        self.generateFMatrices()
        self.initializeWMatrix()
        
        self.customized_features = None
        
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
        
        # load test data
        with open(test_path, 'r', encoding='iso-8859-1') as t:
            for line in t:
                record = line.strip('\n').split('\t')
                self.X_test.append(record[1])
        self.num_of_test = len(self.X_test)
        
    def getFeatureMap(self):
        ''' 
            Map unigram features to indexes.
        '''
        #unique_chars = sorted(list({l for word in self.X_train for l in word}))
        self.n = 1
        n = self.n
        unique_chars = self.getUniqueNGrams(n)
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
            self.index_to_class[i] = cls
            
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

    def train(self, alpha=0.1, epsilon=0.01):
        '''
            Use gradient ascent to train the weights to maximize the objective function.
            
            Parameters:
                alpha: The learning rate.
                epsilon: The tolerance for the difference between the L2 norms of old_W and new_W as the stopping criterion.
        
            Returns:
                objectives: A list. The objective functions every 100 iterations.
                train_accs: A list.The accuracy on the training set every 100 iterations.
                val_accs: A list. The accuracy on the validation set every 100 iterations.  
        '''
        t = 0
        objectives = []
        train_accs = []
        val_accs = []
        while True:
            t = t + 1
            dLdW = self.computeGradient()  
            old_W = self.W.copy()
            self.W = self.W + alpha / np.sqrt(t) * dLdW
            if t % 100 == 1:
                objective = self.getObjective()
                print(objective)
                objectives.append(objective)
                train_acc = self.evaluate(dataset="train")
                train_accs.append(train_acc)
                val_acc = self.evaluate(dataset="val")
                val_accs.append(val_acc)
            if np.linalg.norm(self.W - old_W) < epsilon:
                break
        return objectives, train_accs, val_accs
        
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
      
    def produceTestOutput(self):
        m = self.num_of_test
        Y_pred = self.predict()
        
        Y = np.empty(m, dtype=object).reshape(-1, 1)
        for i in range(m):
            Y[i] = self.index_to_class[int(Y_pred[i])]
            
        X = np.asarray(self.X_test, dtype=object).reshape(-1, 1)
        Z = np.hstack((Y, X))
        
        np.savetxt("../data/output.txt", Z, fmt="%s", delimiter='\t')
        
    def getUniqueNGrams(self, n):
        '''
            Get unique character n-grams from the training data.
            
            Parameter:
                n: An integer indicating the order of n-gram.
                
            Returns:
                unique_NGrams: A list of unique character n-grams.
        '''
        unique_NGrams = []
        for s in self.X_train:
            for i in range(len(s) - n + 1):
                n_gram = s[i:i+n]
                if n_gram not in unique_NGrams:
                    unique_NGrams.append(n_gram)
        return sorted(unique_NGrams)

    def getUniqueWordUnigrams(self):
        '''
            Get unique word unigrams from the training data.
       
            Returns: 
                unique_unigrams: A list of unique word unigrams.
        '''
        unique_unigrams = []
        for s in self.X_train:
            words = s.split()
            for unigram in words:
                if unigram not in unique_unigrams:
                    unique_unigrams.append(unigram)
        return sorted(unique_unigrams)
        
    def getFirstWordUnigrams(self):
        '''
            Get unique first word unigrams from the training data.
            
            Returns:
                unique_unigrams: A list of unique first word unigrams.
        '''
        unique_unigrams = []
        for s in self.X_train:
            unigram = s.split()[0]
            if unigram not in unique_unigrams:
                unique_unigrams.append(unigram)
        return sorted(unique_unigrams)
        
    def getLastWordUnigrams(self):
        '''
            Get unique last word unigrams from the training data.
            
            Returns:
                unique_unigrams: A list of unique last word unigrams.
        '''
        unique_unigrams = []
        for s in self.X_train:
            unigram = s.split()[-1]
            if unigram not in unique_unigrams:
                unique_unigrams.append(unigram)
        return sorted(unique_unigrams)
            
class BigramModel(UnigramModel):
    def __init__(self, train_path, val_path, test_path):
        super().__init__(train_path, val_path, test_path)
        self.n = 2
        
    def getFeatureMap(self):
        n = self.n
        unique_bigrams = self.getUniqueNGrams(n)
        self.num_of_features = len(unique_bigrams)
        for i, gram in enumerate(unique_bigrams):
            self.feature_to_index[gram] = i
        
    def generateFMatrices(self):
        '''
            Generate F_train of shape (num_of_features, num_of_train) of features for the training examples,
                    F_val of shape (num_of_features, num_of_val) of features for the validation examples,
            and      F_test of shape (num_of_features, num_of_val) of features for the test examples.
        '''
        self.F_train = np.zeros((self.num_of_features, self.num_of_train))
        self.F_val = np.zeros((self.num_of_features, self.num_of_val))
        self.F_test = np.zeros((self.num_of_features, self.num_of_test))
        
        n = self.n
        for i in range(self.num_of_train):
            x = self.X_train[i]
            for j in range(len(x) - n + 1):
                gram = x[j:j+n]
                self.F_train[self.feature_to_index[gram], i] += 0.1
        
        for i in range(self.num_of_val):
            x = self.X_val[i]
            for j in range(len(x) - n + 1):
                gram = x[j:j+n]
                if gram in self.feature_to_index.keys():
                    self.F_val[self.feature_to_index[gram], i] += 0.1
                    
        for i in range(self.num_of_test):
            x = self.X_test[i]
            for j in range(len(x) - n + 1):
                gram = x[j:j+n]
                if gram in self.feature_to_index.keys():
                    self.F_test[self.feature_to_index[gram], i] += 0.1
        
class TrigramModel(UnigramModel):
    def __init__(self, train_path, val_path, test_path, lam=1.0):
        super().__init__(train_path, val_path, test_path, lam=1.0)
        self.n = 3
        
    def getFeatureMap(self):
        n = self.n
        unique_trigrams = self.getUniqueNGrams(n)
        self.num_of_features = len(unique_trigrams)
        for i, gram in enumerate(unique_trigrams):
            self.feature_to_index[gram] = i
        
    def generateFMatrices(self):
        '''
            Generate F_train of shape (num_of_features, num_of_train) of features for the training examples,
                    F_val of shape (num_of_features, num_of_val) of features for the validation examples,
            and      F_test of shape (num_of_features, num_of_val) of features for the test examples.
        '''
        self.F_train = np.zeros((self.num_of_features, self.num_of_train))
        self.F_val = np.zeros((self.num_of_features, self.num_of_val))
        self.F_test = np.zeros((self.num_of_features, self.num_of_test))
        n = self.n
        
        for i in range(self.num_of_train):
            x = self.X_train[i]
            for j in range(len(x) - n + 1):
                gram = x[j:j+n]
                self.F_train[self.feature_to_index[gram], i] += 0.1
        
        for i in range(self.num_of_val):
            x = self.X_val[i]
            for j in range(len(x) - n + 1):
                gram = x[j:j+n]
                if gram in self.feature_to_index.keys():
                    self.F_val[self.feature_to_index[gram], i] += 0.1
                    
        for i in range(self.num_of_test):
            x = self.X_test[i]
            for j in range(len(x) - n + 1):
                gram = x[j:j+n]
                if gram in self.feature_to_index.keys():
                    self.F_test[self.feature_to_index[gram], i] += 0.1

class CustomizedUnigramModel(UnigramModel):
    def __init__(self, train_path, val_path, test_path):
        super().__init__(train_path, val_path, test_path)

    def getFeatureMap(self):       
        unique_chars = self.getUniqueNGrams(1)
        first_word_unigrams = self.getFirstWordUnigrams()
        last_word_unigrams = self.getLastWordUnigrams()
        unique_ngram_features = sorted(list(set(unique_chars + first_word_unigrams + last_word_unigrams)))
        self.num_of_features = len(unique_ngram_features)
        print(self.num_of_features)
        for i, feature in enumerate(unique_ngram_features):
            self.feature_to_index[feature] = i

        customized_features = ['CONTAIN_NUMBERS', 'NOW=1', 'NOW=2', 'NOW=3', 'NOW=4', 'NOW>=5', 'CONTAIN_INC', 'CONTAIN_LTD', 'CONTAIN_CORP', 'CONTAIN_CO', 'CONTAIN_PLC', 
                               'CONTAIN_TRUST', 'CONTAIN_CORPORAT', 'CONTAIN_GEL', 'CONTAIN_CREAM', 'CONTAIN_LOTION', 'CONTAIN_CAPLET', 'CONTAIN_COUGH', 'CONTAIN_DAY', 
                               'CONTAIN_NIGHT', 'CONTAIN_HOUR', 'CONTAIN_SPRAY', 'CONTAIN_LIQUID', 'CONTAIN_COLD', 'CONTAIN_POWDER', 'CONTAIN_SOLUTION', 'CONTAIN_MEDICINE',
                               'CONTAIN_STRENGTH', 'CONTAIN_COMPOUND', 'LENGTH<10', 'LENGTH<20', 'LENGTH<30', 'LENGTH>=30', 'START_WITH_UPPER' ]
                               
        self.num_of_features += len(customized_features)
        for i, feature in enumerate(customized_features):
            self.feature_to_index[customized_features[i]] = i + len(unique_chars)
    
    def generateFMatrices(self):
        '''
            Generate F_train of shape (num_of_features, num_of_train) of features for the training examples,
                    F_val of shape (num_of_features, num_of_val) of features for the validation examples,
            and      F_test of shape (num_of_features, num_of_val) of features for the test examples.
        '''
        self.F_train = np.zeros((self.num_of_features, self.num_of_train))
        self.F_val = np.zeros((self.num_of_features, self.num_of_val))
        self.F_test = np.zeros((self.num_of_features, self.num_of_test))
        
        self.addNGramFeatures("train")
        self.addNGramFeatures("val")
        self.addNGramFeatures("test")
        self.addCustomizedFeatures("train")
        self.addCustomizedFeatures("val")
        self.addCustomizedFeatures("test")
        
    def addNGramFeatures(self, dataset):
        n = self.n
        feature_weight = self.feature_weight
        if dataset == "train":
            F = self.F_train
            m = self.num_of_train
            X = self.X_train
        elif dataset == "val":
            F = self.F_val
            m = self.num_of_val
            X = self.X_val
        else:
            F = self.F_test
            m = self.num_of_test
            X = self.X_test
         
        
        for i in range(m):
            for k in range(1, n+1): # k is the order of n-gram
                x = X[i]
                for j in range(len(x) - k + 1):
                    gram = x[j:j+k]
                    if dataset == "train":
                        F[self.feature_to_index[gram], i] += feature_weight
                    else:
                        if gram in self.feature_to_index.keys():
                            F[self.feature_to_index[gram], i] += feature_weight
        
        if dataset == "train":
            self.F_train = F
        elif dataset == "val":
            self.F_val = F
        else:
            self.F_test = F
    
    def addCustomizedFeatures(self, dataset):
        n = self.n
        feature_weight = self.feature_weight
        if dataset == "train":
            F = self.F_train
            m = self.num_of_train
            X = self.X_train
        elif dataset == "val":
            F = self.F_val
            m = self.num_of_val
            X = self.X_val
        else:
            F = self.F_test
            m = self.num_of_test
            X = self.X_test
        
        for i in range(m):
            x = X[i]
            if any(c.isdigit() for c in x):
                F[self.feature_to_index['CONTAIN_NUMBERS'], i] += (feature_weight)
            if len(x.split()) == 1:
                F[self.feature_to_index['NOW=1'], i] += (feature_weight)
            elif len(x.split()) == 2:
                F[self.feature_to_index['NOW=2'], i] += (feature_weight)
            elif len(x.split()) == 3:
                F[self.feature_to_index['NOW=3'], i] += (feature_weight)
            elif len(x.split()) == 4:
                F[self.feature_to_index['NOW=4'], i] += (feature_weight)
            else:
                F[self.feature_to_index['NOW>=5'], i] += (feature_weight)
            if "INC" in x.upper():
                F[self.feature_to_index['CONTAIN_INC'], i] += (feature_weight)
            elif "LTD" in x.upper():
                F[self.feature_to_index['CONTAIN_LTD'], i] += (feature_weight)
            elif "CORP" in x.upper():
                F[self.feature_to_index['CONTAIN_CORP'], i] += (feature_weight)
            elif "CO" in x.upper():
                F[self.feature_to_index['CONTAIN_CO'], i] += (feature_weight)
            elif "PLC" in x.upper():
                F[self.feature_to_index['CONTAIN_PLC'], i] += (feature_weight)
            elif "TRUST" in x.upper():
                F[self.feature_to_index['CONTAIN_TRUST'], i] += (feature_weight)
            elif "CORPORAT" in x.upper():
                F[self.feature_to_index['CONTAIN_CORPORAT'], i] += (feature_weight)
            elif "GEL" in x.upper():
                F[self.feature_to_index['CONTAIN_GEL'], i] += (feature_weight)
            elif "CREAM" in x.upper():
                F[self.feature_to_index['CONTAIN_CREAM'], i] += (feature_weight)   
            elif "LOTION" in x.upper():
                F[self.feature_to_index['CONTAIN_LOTION'], i] += (feature_weight)                
            elif "CAPLET" in x.upper():
                F[self.feature_to_index['CONTAIN_CAPLET'], i] += (feature_weight)              
            elif "COUGH" in x.upper():
                F[self.feature_to_index['CONTAIN_COUGH'], i] += (feature_weight)               
            elif "DAY" in x.upper():
                F[self.feature_to_index['CONTAIN_DAY'], i] += (feature_weight)               
            elif "NIGHT" in x.upper():
                F[self.feature_to_index['CONTAIN_NIGHT'], i] += (feature_weight)                
            elif "HOUR" in x.upper():
                F[self.feature_to_index['CONTAIN_HOUR'], i] += (feature_weight) 
            elif "SPRAY" in x.upper():
                F[self.feature_to_index['CONTAIN_SPRAY'], i] += (feature_weight)                 
            elif "LIQUID" in x.upper():
                F[self.feature_to_index['CONTAIN_LIQUID'], i] += (feature_weight)                 
            elif "COLD" in x.upper():
                F[self.feature_to_index['CONTAIN_COLD'], i] += (feature_weight)                
            elif "POWDER" in x.upper():
                F[self.feature_to_index['CONTAIN_POWDER'], i] += (feature_weight)     
            elif "SOLUTION" in x.upper():
                F[self.feature_to_index['CONTAIN_SOLUTION'], i] += (feature_weight)                
            elif "MEDICINE" in x.upper():
                F[self.feature_to_index['CONTAIN_MEDICINE'], i] += (feature_weight)                
            elif "STRENGTH" in x.upper():
                F[self.feature_to_index['CONTAIN_STRENGTH'], i] += (feature_weight)
            elif "COMPOUND" in x.upper():
                F[self.feature_to_index['CONTAIN_COMPOUND'], i] += (feature_weight) 
                      
            if len(x) < 10:
                F[self.feature_to_index['LENGTH<10'], i] += (feature_weight)
            elif len(x) < 20:
                F[self.feature_to_index['LENGTH<20'], i] += (feature_weight)
            elif len(x) < 30:
                F[self.feature_to_index['LENGTH<30'], i] += (feature_weight)
            else:
                F[self.feature_to_index['LENGTH>=30'], i] += (feature_weight)
                    
            for word in x.split():
                if word[0].isupper():
                    F[self.feature_to_index['START_WITH_UPPER'], i] += feature_weight
                    
        if dataset == "train":
            self.F_train = F
        elif dataset == "val":
            self.F_val = F
        else:
            self.F_test = F  
                
        
            
class CustomizedBigramModel(CustomizedUnigramModel):
    def __init__(self, train_path, val_path, test_path):
        super().__init__(train_path, val_path, test_path)
        self.n = 2
        
    def getFeatureMap(self):
        ''' 
            Map Bigram features to indexes.
        '''
        unique_chars = self.getUniqueNGrams(1)
        unique_bigrams = self.getUniqueNGrams(2)
        first_word_unigrams = self.getFirstWordUnigrams()
        last_word_unigrams = self.getLastWordUnigrams()
        unique_ngram_features = sorted(list(set(unique_chars + unique_bigrams + first_word_unigrams + last_word_unigrams))) 
        self.num_of_features = len(unique_ngram_features)
        print(self.num_of_features)
        for i, feature in enumerate(unique_ngram_features):
            self.feature_to_index[feature] = i
            
        customized_features = ['CONTAIN_NUMBERS', 'NOW=1', 'NOW=2', 'NOW=3', 'NOW=4', 'NOW>=5', 'CONTAIN_INC', 'CONTAIN_LTD', 'CONTAIN_CORP', 'CONTAIN_CO', 'CONTAIN_PLC', 
                               'CONTAIN_TRUST', 'CONTAIN_CORPORAT', 'CONTAIN_GEL', 'CONTAIN_CREAM', 'CONTAIN_LOTION', 'CONTAIN_CAPLET', 'CONTAIN_COUGH', 'CONTAIN_DAY', 
                               'CONTAIN_NIGHT', 'CONTAIN_HOUR', 'CONTAIN_SPRAY', 'CONTAIN_LIQUID', 'CONTAIN_COLD', 'CONTAIN_POWDER', 'CONTAIN_SOLUTION', 'CONTAIN_MEDICINE',
                               'CONTAIN_STRENGTH', 'CONTAIN_COMPOUND', 'LENGTH<10', 'LENGTH<20', 'LENGTH<30', 'LENGTH>=30', 'START_WITH_UPPER' ]
                               
        self.num_of_features += len(customized_features)
        for i, feature in enumerate(customized_features):
            self.feature_to_index[customized_features[i]] = i + len(unique_ngram_features)

        
        '''
        unique_word_unigrams = self.getUniqueWordUnigrams()
        self.num_of_features += len(unique_word_unigrams)
        for i, feature in enumerate(unique_word_unigrams):
            self.feature_to_index[feature] = i + len(unique_ngram_features) + len(customized_features)
        '''
        
        '''
            Did not add character bigram features!
        '''
        
class CustomizedTrigramModel(CustomizedBigramModel):
    def __init__(self, train_path, val_path, test_path):
        super().__init__(train_path, val_path, test_path)
        self.n = 3
        
    def getFeatureMap(self):
        ''' 
            Map Bigram features to indexes.
        '''
        '''
        unique_chars = self.getUniqueNGrams(1)
        unique_bigrams = self.getUniqueNGrams(2)
        unique_trigrams = self.getUniqueNGrams(3)
        first_word_unigrams = self.getFirstWordUnigrams()
        last_word_unigrams = self.getLastWordUnigrams()
        unique_ngram_features = sorted(list(set(unique_chars + unique_bigrams + unique_trigrams + first_word_unigrams + last_word_unigrams))) 
        self.num_of_features = len(unique_ngram_features)
        print(self.num_of_features)
        for i, feature in enumerate(unique_ngram_features):
            self.feature_to_index[feature] = i
            
        customized_features = ['CONTAIN_NUMBERS', 'NOW=1', 'NOW=2', 'NOW=3', 'NOW=4', 'NOW>=5', 'CONTAIN_INC', 'CONTAIN_LTD', 'CONTAIN_CORP', 'CONTAIN_CO', 'CONTAIN_PLC', 
                               'CONTAIN_TRUST', 'CONTAIN_CORPORAT', 'CONTAIN_GEL', 'CONTAIN_CREAM', 'CONTAIN_LOTION', 'CONTAIN_CAPLET', 'CONTAIN_COUGH', 'CONTAIN_DAY', 
                               'CONTAIN_NIGHT', 'CONTAIN_HOUR', 'CONTAIN_SPRAY', 'CONTAIN_LIQUID', 'CONTAIN_COLD', 'CONTAIN_POWDER', 'CONTAIN_SOLUTION', 'CONTAIN_MEDICINE',
                               'CONTAIN_STRENGTH', 'CONTAIN_COMPOUND', 'LENGTH<10', 'LENGTH<20', 'LENGTH<30', 'LENGTH>=30', 'START_WITH_UPPER']
                               
        self.num_of_features += len(customized_features)
        for i, feature in enumerate(customized_features):
            self.feature_to_index[customized_features[i]] = i + len(unique_ngram_features)
        '''
        unique_chars = self.getUniqueNGrams(1)
        unique_bigrams = self.getUniqueNGrams(2)
        unique_trigrams = self.getUniqueNGrams(3)
        unique_ngram_features = unique_chars + unique_bigrams + unique_trigrams
        self.num_of_features = len(unique_ngram_features)
        print(self.num_of_features)
        for i, feature in enumerate(unique_ngram_features):
            self.feature_to_index[feature] = i
            
        customized_features = ['CONTAIN_NUMBERS', 'NOW=1', 'NOW=2', 'NOW=3', 'NOW=4', 'NOW>=5', 'CONTAIN_INC', 'CONTAIN_LTD', 'CONTAIN_CORP', 'CONTAIN_CO', 'CONTAIN_PLC', 
                               'CONTAIN_TRUST', 'CONTAIN_CORPORAT', 'CONTAIN_GEL', 'CONTAIN_CREAM', 'CONTAIN_LOTION', 'CONTAIN_CAPLET', 'CONTAIN_COUGH', 'CONTAIN_DAY', 
                               'CONTAIN_NIGHT', 'CONTAIN_HOUR', 'CONTAIN_SPRAY', 'CONTAIN_LIQUID', 'CONTAIN_COLD', 'CONTAIN_POWDER', 'CONTAIN_SOLUTION', 'CONTAIN_MEDICINE',
                               'CONTAIN_STRENGTH', 'CONTAIN_COMPOUND', 'LENGTH<10', 'LENGTH<20', 'LENGTH<30', 'LENGTH>=30', 'START_WITH_UPPER']
                               
        self.num_of_features += len(customized_features)
        for i, feature in enumerate(customized_features):
            self.feature_to_index[customized_features[i]] = i + len(unique_ngram_features)
        
        '''
            Did not add character trigram features!
        '''
        
        
        
        
        
        
        
        
        
    