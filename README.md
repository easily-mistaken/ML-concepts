Machine learning concepts

prediction, classification, and calculating their error rate 

1. MNIST files on Handwritten digits

This includes files sim.py, genMNIST.py. It contains a simple python code that shows you a few pictures and asks you to guess what digits they are. It tells you what the answer is as well as what the machine predicted. I have used 1-NN prediction here. This file takes it's data from test data, test label, train label and train data files. These are data from MNIST.

2. what-floor.py

It uses the floor and weight data of girls living in Girls Hostel 3 and tries to predict the floor of new entry on the basis of weight. This uses Generative modelling in 1-D. The dataset comes from GH3.txt file. 

Generative modelling:
The optimal prediction is said to be the one with highest πj(Pj(x)).

3. Winery files

These include wineuni.py, winebi.py, winemulti.py. The dataset to this is in wine.data.txt and a general overview of the data in wine.names.txt. We use generative modelling using univariate gaussian, bivariate gaussian and multivariate gaussian in wineuni.py, winebi.py and winemulti.py.

Gaussian:

