import numpy as np
import pandas as pd
import re


class LogisticRegression:
    def __init__(self, n: int) -> None:
        """
        n: Length of Vocabulary.
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

    def train(self, X, y, epochs: int = 10, learning_rate: float = 0.001, verbose: bool=False) -> None:
        for epoch in range(epochs):
            # Forward propagation.
            Z = np.dot(X, self.W.T) + self.B
            A = self.sigmoid(Z)

            # Calculate error. 
            error = y - A

            # Backpropagation.
            dW = np.dot(X.T, error) / len(X)
            dB = np.sum(error) / len(X)
            
            # Update weights.
            self.W += learning_rate * dW
            self.B += learning_rate * dB
            
            if verbose:
                if epoch % 1 == 0:
                    error = np.mean(error)
                    print(f"Epoch {epoch}, Loss: {error}")

    def predict(self, X: np.array) -> np.array: 
        Z = np.dot(X, self.W.T) + self.B
        A = self.sigmoid(Z)
        output = self.step(A) 
        return output

    def evaluate(self, X: np.array, y: np.array) -> tuple[float, float, float, float]:
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

    def save(self) -> None:
        with open('model.npy', 'wb') as f:
            np.save(f, self.W)
            np.save(f, self.B)

    def load(self) -> None:
        with open('model.py', 'rb') as f:
            self.W = np.load(f)
            self.B = np.load(f)


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