%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import time
import random

## Load the training set
train_data = np.load('train_data.npy')
train_labels = np.load('train_labels.npy')

## Load the testing set
test_data = np.load('test_data.npy')
test_labels = np.load('test_labels.npy')

## Define a function that displays a digit given its vector representation
def show_digit(x):
    plt.axis('off')
    plt.imshow(x.reshape((28,28)), cmap=plt.cm.gray)
    plt.show()
    return

## Define a function that takes an index into a particular data set ("train" or "test") and displays that image.
def vis_image(index, dataset="train"):
    if(dataset=="train"):
        show_digit(train_data[index,])
    else:
        show_digit(test_data[index,])
    return
## Takes two datas and returns the squared distance
def squared_dist(x,y):
    return np.sum(np.square(x-y))

## Takes a vector x and returns the index of its nearest neighbor in train_data
def find_NN(x):
    # Compute distances from x to every row in train_data
    distances = [squared_dist(x,train_data[i,]) for i in range(len(train_labels))]
    # Get the index of the smallest distance
    return np.argmin(distances)

## Takes a vector x and returns the class of its nearest neighbor in train_data
def NN_classifier(x):
    index = find_NN(x)
    return train_labels[index]


answer = 0
our_guess = 0
for i in range(5):
    newnum = random.randint(1,1000)
    vis_image(newnum, "test")
    our_guess = (our_guess*10)+NN_classifier(test_data[newnum,])
    answer = (answer*10)+test_labels[newnum,]
entry = input("Enter the captcha:")
if(entry == answer):
    print("Yay!!",entry,"is the correct answer")
else:
    print("Oops! the answer is not",entry,". It is",answer)
print("We guessed",our_guess)
