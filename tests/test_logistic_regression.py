#!/usr/bin/env python3

from logisticregression import LogisticRegression
import numpy as np


def test_logistic_regression_1():
    X = np.array([[0], [1]])
    y = np.array([1, 0])
    not_ = LogisticRegression(1)
    not_.train(X, y, epochs=1000, learning_rate=0.1)
    assert np.array_equal(not_.predict(X), y)


def test_logistic_regression_2():
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y = np.array([0, 1, 1, 1])
    or_gate= LogisticRegression(2)
    or_gate.train(X, y, epochs=1000, learning_rate=0.1)
    assert np.array_equal(or_gate.predict(X), y)


def test_logistic_regression_3():
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y = np.array([0, 0, 0, 1])
    and_gate= LogisticRegression(2)
    and_gate.train(X, y, epochs=1000, learning_rate=0.1)
    assert np.array_equal(and_gate.predict(X), y)