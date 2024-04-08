import numpy as np
import pandas as pd
import re


class LogisticRegression:
    def __init__(self, n: int):
        """
        >>> lr = LogisticRegression(5)
        >>> lr.W
        array([0., 0., 0., 0., 0.])
        >>> lr.B
        array([0.])
        """ 
        self.W = np.zeros(n)
        self.B = np.zeros(1)

    def sigmoid(self, X: np.array) -> np.array:
        return 1.0 / (1.0 + np.exp(-X))

    def step(self, X: np.array) -> np.array:
        return np.where(X >= 0.5, 1, 0)

    def train(self, X, y, epochs: int = 1, learning_rate: float = 0.1, verbose: bool=False):
        for epoch in range(epochs):
            # Forward propagation.
            Z = np.dot(X, self.W.T) + self.B
            A = self.sigmoid(Z)
            # output = self.step(A)
            output = A

            # Calculate error. 
            error = y - output

            # Backpropagation.
            dW = np.dot(X.T, error) / len(X)
            dB = np.sum(error) / len(X)
            self.W += learning_rate * dW
            self.B += learning_rate * dB
            
            if verbose:
                if epoch % 1 == 0:
                    error = np.mean(error)
                    print(f"Epoch {epoch}, Loss: {error}")


    def predict(self, X: np.array) -> np.array: 
        """
        >>> V = create_vocabulary('tweets-doctest.csv')
        >>> X, y = read_labeled_data('tweets-doctest.csv', V)
        >>> lr = LogisticRegression(len(V))
        >>> lr.train(X, y, 10, 0.001)
        >>> lr.predict(X)
        array([0, 1])
        """
        # Z = np.dot(X, self.W.T)
        Z = np.dot(X, self.W.T) + self.B
        A = self.sigmoid(Z)
        output = self.step(A) 
        return output

    def evaluate(self, X, y):
        # Calculate predictions. 
        predictions = self.predict(X)
        TP, TN, FP, FN = 0, 0, 0, 0
        for i in range(len(y)):
            if predictions[i] == 1 and y[i] == 1:
                TP += 1
            elif predictions[i] == 0 and y[i] == 0:
                TN += 1
            elif predictions[i] == 1 and y[i] == 0:
                FP += 1
            else:
                FN += 1
        precision = TP / (TP + FP) if TP + FP != 0 else 0
        recall = TP / (TP + FN) if TP + FN != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN != 0 else 0
        return precision, recall, f1, accuracy


def tokenize(s: str) -> list[str]:
    """
    Splits string into tokens (words).
    >>> tokenize("I love star wars.")
    ['i', 'love', 'star', 'wars']
    >>> tokenize("Yo BRO! This is some AI shit!")
    ['yo', 'bro', 'this', 'is', 'some', 'ai', 'shit']
    """
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(s.lower())


def create_vocabulary(file: str) -> dict[str, int]:
    """
    Reads the given file and generates vocabulary mapping
    from word to word id. 
    >>> V = create_vocabulary('tweets-doctest.csv')
    >>> V
    {'have': 0, 'a': 1, 'nice': 2, 'day': 3, 'love': 4, 'you': 5, 'all': 6, 'bad': 7, 'damn': 8}
    """
    vocabulary: dict[str, int]={}
    word_id: int = 0 
    data = pd.read_csv(file)
    for _, row in data.iterrows():
        text = row['text']
        for token in tokenize(text):
            if token not in vocabulary:
                vocabulary[token] = word_id
                word_id += 1
    return vocabulary


def read_labeled_data(file: str, vocabulary: dict[str, int], verbose:bool=False) -> tuple[np.array, np.array]:
    """
    >>> V = create_vocabulary('tweets-doctest.csv') 
    >>> X, y = read_labeled_data('tweets-doctest.csv', V) 
    >>> y
    array([0., 1.])
    >>> X
    array([[1., 1., 1., 1., 1., 1., 1., 0., 0.],
           [1., 1., 0., 1., 0., 1., 1., 1., 1.]])
    """ 
    n = len(vocabulary)
    labels = []
    X = []
    data = pd.read_csv(file) 
    for index, row in data.iterrows():
        text, label = row['text'], float(row['generated'])
        x = np.zeros(n)
        for word in tokenize(text):
            x[vocabulary[word]] += 1
        X.append(x)
        labels.append(float(label))
        if verbose:
            if index % 1000 == 0:
                print(f'Read row {index}')
    X, y = np.stack(X), np.array(labels)
    return X, y


if __name__ == '__main__':
    learning_rate = 0.001
    epochs = 10
    print()
    print(f"epochs = {epochs}, learning_rate = {learning_rate}")
    print() 

    # Create vocabulary.
    print(f'Creating vocabulary..')
    V = create_vocabulary('AI_Human.csv')
    print(f'Created vocabulary!')
    print()
    
    print(f'Reading labeled data..')
    X, y = read_labeled_data('AI_Human.csv', V)
    print(f'Read labeled data!')
    print()
    
    # Shuffle data. 
    permu = np.random.permutation(len(X)) 
    X, y = X[permu], y[permu]

    # Split for training and testing data.
    X_train, y_train = X[:30000], y[:30000]
    X_test, y_test = X[30000:], y[30000:]

    # Create LogisticRegression model.
    lr = LogisticRegression(len(V))
    # Train on training data. 
    print(f'Start training NeuralNetwork..')
    lr.train(X_train, y_train, epochs, learning_rate, True)
    print(f'Training finished!')
    print()

    # Evaluate on testing data. 
    precision, recall, f1, accuracy = lr.evaluate(X_test, y_test)
    print(f"precision = {round(precision * 100, 2)}%")
    print(f"recall = {round(recall * 100, 2)}%")
    print(f"F1 = {round(f1 * 100, 2)}%")
    print(f"accuracy = {round(accuracy * 100, 2)}%")
    print()
