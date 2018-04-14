#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 01:37:56 2018

@author: himanshu
"""

import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import random
import collections
import pickle


filename = "Assignment_2_data.txt"
fhandle = open (filename, "r")

message_type = []
word_data = []
all_words = set ()

table = str.maketrans('', '', string.punctuation)
stop_words = set(stopwords.words('english'))
porter = PorterStemmer()

for line in fhandle:
    if line.startswith ("ham"):
        message_type.append (0)
        line = line[4:]
    else:
        message_type.append (1)
        line = line[5:]
    words = word_tokenize (line)

    # Remove punc
    words = [w.translate (table) for w in words]
    words = [word for word in words if word != '']

    # remove remaining tokens that are not alphabetic
    # words = [word for word in words if word.isalpha()]

    # Remove stop words
    words = [w.lower () for w in words if not w in stop_words]

    # Stemming
    words = [porter.stem(word) for word in words]

    word_data.append (set (words))
    all_words = all_words.union (set (words))

# print ("message_type:", message_type)
# print ("word_data:", word_data)
all_words = list (all_words)
# print ("all_words:", all_words)

# counter = collections.Counter(all_words)
# all_words = counter.most_common (2000)



X = []
y = message_type
for i in range (num_examples):
    v = []
    for j in range (num_features):
        if all_words[j] in word_data[i]:
            v.append (1)
        else:
            v.append (0)
    X.append (v)
print ("completed")


# print (X[0])
# print (all_words)

X = np.asarray (X)
y = np.asarray (y)
'''

f = open("input.pickle", "rb")
X = pickle.load(f)
f.close()

f = open("output.pickle", "rb")
y = pickle.load(f)
f.close()
'''
print ("X-shape:", X.shape)
print ("y-shape:", y.shape)

num_examples = X.shape[0]
num_features = X.shape[1]

print ("Number of examples:", num_examples)
print ("Number of features:", num_features)

test_size = 0.2

len_test = (int )(test_size * num_examples)
len_train = num_examples - len_test

# print (type (X), X[:10])
# print (type (y), y[:10])

# Train test split
l1 = random.sample(range(0,num_examples), len_train)
l2 =[]
for i in  range(num_examples):
    l2.append(i)
l3 = (list(set(l2) - set(l1)))

X_train = np.zeros (shape = (len_train, num_features), dtype = float, order = 'C')
X_test = np.zeros (shape = (len_test, num_features), dtype = float, order = 'C')
y_train = np.zeros (shape = len_train, dtype = float)
y_test = np.zeros (shape = len_test, dtype = float)

for i in range (len_train):
    X_train[i] = X[l1[i]]
    y_train[i] = y[l1[i]]

for i in range (len_test):
    X_test[i] = X[l3[i]]
    y_test[i] = y[l3[i]]

print ("X_train-shape:", X_train.shape)
print ("y_train-shape:", y_train.shape)
print ("X_test-shape:", X_test.shape)
print ("y_test-shape:", y_test.shape)

m = 1
n0 = X_train.shape[1]
n1 = 100
n2 = 50
n3 = 1
alpha = 0.1

def tan_h (z):
    return (np.exp (z) - np.exp (-z)) / (np.exp (z) + np.exp (-z))

def calculate_da_logistic (y, a):
    return -(y / a) + ((1 - y) / (1 - a))

def tan_h_grad (z):
    return 1 - np.square (tan_h (z))

def calculate_loss_logistic (y, a):
    return -(1 / (2 * y.shape[1])) * np.sum (y * np.log (a) + (1 - y) * np.log (1 - a))

def calculate_loss_squared (y, a):
    return (1 / 2 * y.shape[1]) * np.sum (np.square (y - a))

def calculate_da_squared (y, a):
    return 2 * (a - y)

W1 = np.random.randn (n1, n0) * 0.1
b1 = np.zeros ((n1, 1))

W2 = np.random.randn (n2, n1) * 0.1
b2 = np.zeros ((n2, 1))

W3 = np.random.randn (n3, n2) * 0.1
b3 = np.zeros ((n3, 1))

"""
W1  = np.random.uniform(low =-1, high =1, size=(n1,n0))
W2  = np.random.uniform(low =-1, high =1, size=(n2,n1))
W3  = np.random.uniform(low =-1, high =1, size=(1,n2))
b1 = np.random.uniform (low = -1, high=1, size=(n1, 1))
b2 = np.random.uniform (low = -1, high=1, size=(n2, 1))
b3 = np.random.uniform (low = -1, high=1, size=(n3, 1))
"""

epochs = 20
train_errors = []
test_errors = []
for i in range (epochs):
    j = 0
    a = None
    train_error = 0
    test_error = 0
    while j < X_train.shape[0]:
        A0 = X_train[j].reshape ((n0, m))

        y_hat = y_train[j]

        Z1 = np.matmul (W1, A0) + b1

        #print (Z1.shape)

        Z2 = np.matmul (W2, tan_h (Z1)) + b2

        Z3 = np.matmul (W3, tan_h (Z2)) + b3

        a = tan_h (Z3)

        da = 2 * (a[0][0] - y_train[j])

        # print ("a:", a)
        # print ("Z3:", Z3)
        # print ("da:", da)
        #print ("da =", da)

        dZ3 = tan_h_grad (Z3) * da
        dW3 = np.matmul (dZ3, np.transpose (sigmoid (Z2)))
        db3 = (1 / m) * np.sum (dZ3, axis = 1, keepdims = True)

        dZ2 = np.multiply (tan_h_grad (Z2), np.matmul (np.transpose (W3), dZ3))
        dW2 = np.matmul (dZ2, np.transpose (sigmoid (Z1)))
        db2 = (1 / m) * np.sum (dZ2, axis = 1, keepdims = True)

        dZ1 = np.multiply (tan_h_grad (Z1), np.matmul (np.transpose (W2), dZ2))
        dW1 = np.matmul (dZ1, np.transpose (A0))
        db1 = (1 / m) * np.sum (dZ1, axis = 1, keepdims = True)

        W3 = W3 - alpha * dW3
        b3 = b3 - alpha * db3

        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2

        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1

        j += m

    j = 0
    while j < X_train.shape[0]:
        Z1 = np.matmul (W1, X_train[j].reshape (n0, m)) + b1
        Z2 = np.matmul (W2, tan_h (Z1)) + b2
        Z3 = np.matmul (W3, tan_h (Z2)) + b3
        a_ = tan_h (Z3)
        train_error += (a_[0][0] - y_train[j]) ** 2
        j += m

    j = 0
    while j < X_test.shape[0]:
        Z1 = np.matmul (W1, X_test[j].reshape(n0, m)) + b1
        Z2 = np.matmul (W2, tan_h (Z1)) + b2
        Z3 = np.matmul (W3, tan_h (Z2)) + b3
        a_ = tan_h (Z3)
        test_error += (a_[0][0] - y_test[j]) ** 2
        j += m

    train_errors.append (train_error / (2 * X_train.shape[0]))
    test_errors.append (test_error / (2 * X_test.shape[0]))


    print ("Epoch:", i, " ==> Train Loss:", train_errors[i], " And Test Loss:", test_errors[i])

    random.shuffle (X_train)

    i += 1

plt.plot (range (len (train_errors)), train_errors, c = "r")
plt.plot (range (len (test_errors)), test_errors, c = "b")
plt.xlabel ("Epochs")
plt.ylabel ("Mean Square Error")
plt.title ("Mean Square Error v/s Epochs")
plt.show ()

thresolds = np.linspace (0.1, 0.9, 9)
minimum_error = None
min_thresold = None
for thresold in thresolds:
    j = 0
    test_error = 0
    while j < X_test.shape[0]:
        Z1 = np.matmul (W1, X_test[j].reshape(n0, m)) + b1
        Z2 = np.matmul (W2, tan_h (Z1)) + b2
        Z3 = np.matmul (W3, tan_h (Z2)) + b3
        a_ = tan_h (Z3)
        test_error += int ((a_[0][0] < thresold) ^ (y_test[j] < thresold))
        # test_error += (a_[0][0] - y_test[j]) ** 2
        j += m
    print ("Thresold:", thresold, " => Error:", test_error)
    if minimum_error is None:
        minimum_error = test_error
        min_thresold = thresold
    elif test_error < minimum_error:
        minimum_error = test_error
        min_thresold = thresold
print ("We find mimimum error on thresold = {min_thresold}, which is {minimum_error}".format (min_thresold = min_thresold, minimum_error = minimum_error))
