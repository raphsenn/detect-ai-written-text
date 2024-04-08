# detect-ai-written-text
Detect AI generated text, logistic regression neural network , written in raw python (just numpy).

## About the dataset
Around 60.000 essays are available in this dataset, both created by AI and written by Human.

##### Results
Trained on 15.000 AI Generated and 15.0000 Human written essays.
Here are some results with more 15.000 AI Generated and 15.000 Human essays that the model never has seen bevor.
--------------------------------
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


