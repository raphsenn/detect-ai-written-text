import numpy as np
from logisticregression import LogisticRegression, read_labeled_data, create_vocabulary


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
    # lr.save()
    print(f'Training finished!')
    print()

    # Evaluate on testing data. 
    precision, recall, f1, accuracy = lr.evaluate(X_test, y_test)
    print(f"precision = {round(precision * 100, 2)}%")
    print(f"recall = {round(recall * 100, 2)}%")
    print(f"F1 = {round(f1 * 100, 2)}%")
    print(f"accuracy = {round(accuracy * 100, 2)}%")
    print()
