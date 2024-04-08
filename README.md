# detect-ai-written-text
Detect AI generated text, logistic regression neural network , written in raw python (just numpy).

## About the dataset
Around 60.000 essays are available in this dataset, both created by AI and written by Human.
https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text/data

##### Results
Trained on 15.000 AI Generated and 15.0000 Human written essays.
Here are some results with 15.000 AI Generated and 15.000 Human essays that the model never has seen bevor.

#### Parameters:
- epochs: 1
- learning_rate: 0.1

```shell
precision = 67.79%
recall = 94.71%
F1 = 79.02%
accuracy = 72.27%
```
--------------------------------
#### Parameters:
- epochs: 5
- learning_rate: 0.001

```shell
precision = 73.14%
recall = 90.89%
F1 = 81.05%
accuracy = 76.66%
```
--------------------------------
#### Parameters:
- epochs: 10
- learning_rate: 0.001

```shell
precision = 78.94%
recall = 87.84%
F1 = 83.15%
accuracy = 80.35%
```
--------------------------------
#### Parameters:
- epochs: 20
- learning_rate: 0.001

```shell
precision = 82.53%
recall = 86.53%
F1 = 84.48%
accuracy = 82.55%
```

## How to use

### Train the model.
```python
from logistic-regression import LogisticRegression, read_labeled_data, create_vocabulary

# Create vocabulary.
V = create_vocabulary('AI_Human.csv')
# Read labeled data.
X, y = read_labeled_data('AI_Human.csv', V)

# Shuffle data. 
permu = np.random.permutation(len(X)) 
X, y = X[permu], y[permu]

# Split for training and testing data.
X_train, y_train = X[:30000], y[:30000]
X_test, y_test = X[30000:], y[30000:]

# Create LogisticRegression model.
lr = LogisticRegression(len(V))

# Train on training data. 
lr.train(X_train, y_train, True)

# Evaluate on testing data. 
precision, recall, f1, accuracy = lr.evaluate(X_test, y_test)
print(f"precision = {round(precision * 100, 2)}%")
print(f"recall = {round(recall * 100, 2)}%")
print(f"F1 = {round(f1 * 100, 2)}%")
print(f"accuracy = {round(accuracy * 100, 2)}%")
```

## Save weights and bias.
To save trained weights and bias just write:
```python
lr.save()
```
## Load weights and bias.
To save trained weights and bias just write:
```python
lr.load()
```

