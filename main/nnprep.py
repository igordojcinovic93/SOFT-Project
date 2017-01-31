'''
Modul za obucavanje neuronske mreze i sve prevashodne
pripreme nad podacima potrebne za obucavanje

Sa velikim ugledom na materijal ponudjen na repozitorijumu predmeta

Igor Dojcinovic
'''

#potrebne biblioteke

from sklearn.datasets import fetch_mldata
from keras.layers.core import Activation, Dense
from keras.models import  Sequential
from keras.optimizers import  SGD
import numpy as np

#globalne promenljive
global train_out
global test_out
global train_data
global test_data
global prept_model
train_data = []
train_out = []
test_out = []
test_data = []
prept_model = []
#pomocne funkcije

def get_test_data():
    if(test_data == []):
        prep_data()
    return test_data

def get_model():
    global prept_model
    if(prept_model == []):
        prept_model = model_prep()
    return prept_model

def to_categorical(labels, n):
    retVal = np.zeros((len(labels), n), dtype = 'int')
    ll = np.array(list(enumerate(labels)))
    retVal[ll[:,0],ll[:,1]] = 1
    return retVal


#preuzimanje i razdvajanje MNIST data-set-a
def prep_data():
    mnist = fetch_mldata('MNIST original')
    data = mnist.data / 255.0
    labels = mnist.target.astype('int')
    print("Successfully acuiered MNIST")

#deljenje dataset-a na betch za treniranje i za testiranje
    train_rank = 5000
    test_rank = 100

    train_subset = np.random.choice(data.shape[0], train_rank)
    test_subset = np.random.choice(data.shape[0], test_rank)
    global train_data
    train_data = data[train_subset]
    global test_data
    test_data = data[test_subset]
    train_labels = labels[train_subset]
    test_labels = labels[test_subset]
    print("Successfully accuiered subsets")


#konverzija subsetova u pogodan oblik
    global train_out, test_out
    train_out = to_categorical(train_labels, 10)
    test_out = to_categorical(test_labels, 10)
    print("Successfully transformed labels")

#funkcija za pripremu modela za obucavanje
def model_prep():
    model = Sequential()
    model.add(Dense(70, input_dim=784))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))

    sgd = SGD(lr=0.1, decay=0.001, momentum=0.7)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model

#funkcija za obucavanje
def train_nn():
    train(prept_model, train_data, train_out)

def train(model, tr_data, tr_out):
    training = model.fit(tr_data, tr_out, nb_epoch=500, batch_size=400, verbose=0)
    print training.history['loss'][-1]

#funkcija za validaciju
def validate(model, data, data_out):
    scores = model.evaluate(data, data_out, verbose=1)
    print 'test', scores

#glava funkcija
def prep():
    prep_data()
    print("Data prept")
    global prept_model
    prept_model = model_prep()
    print("Model prept")
    train_nn()
    print("Neural network trained")


def validation():
    validate(prept_model, test_data, test_out)
    validate(prept_model, train_data, train_out)
