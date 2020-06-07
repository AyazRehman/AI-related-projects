# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 23:00:51 2019

@author: YourAverageSciencePal
"""
import sys
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray
import pickle
import random
'''
Depending on your choice of library you have to install that library using pip
'''


'''
Read chapter on neural network from the book. Most of the derivatives,formulas 
are already there.
Before starting this assignment. Familarize yourselves with np.dot(),
What is meant by a*b in 2 numpy arrays.
What is difference between np.matmul and a*b and np.dot.
Numpy already has vectorized functions for addition and subtraction and even for division
For transpose just do a.T where a is a numpy array 
Also search how to call a static method in a class.
If there is some error. You will get error in shapes dimensions not matched
because a*b !=b*a in matrices
'''

class NeuralNetwork():
    @staticmethod
    #note the self argument is missing i.e. why you have to search how to use static methods/functions)

    def cross_entropy_loss(y_pred, y_true):
        '''implement cross_entropy loss error function here
        Hint: Numpy has a sum function already
        Numpy has also a log function
        Remember loss is a number so if y_pred and y_true are arrays you have to sum them in the end
        after calculating -[y_true*log(y_pred)]'''
        return -(y_true * np.log(y_pred)).sum()

    @staticmethod
    def accuracy(y_pred, y_true):
        '''function to calculate accuracy of the two lists/arrays
        Accuracy = (number of same elements at same position in both arrays)/total length of any array
        Ex-> y_pred = np.array([1,2,3]) y_true=np.array([1,2,4]) Accuracy = 2/3*100 (2 Matches and 1 Mismatch)'''
        acc = 0
        for i in range(len(y_pred)):
            if np.argmax(y_pred[i]) == np.argmax(y_true[i]):
                acc += 1
        acc = acc/len(y_pred)
        acc = acc * 100
        return acc

    
    @staticmethod
    def softmax(x):
        '''Implement the softmax function using numpy here
        Hint: Numpy sum has a parameter axis to sum across row or column. You have to use that
        Use keepdims=True for broadcasting
        You guys should have a pretty good idea what the size of returned value is.
        '''
        expx = np.exp(x)
        return expx / expx.sum(axis=1, keepdims=True)
    
    @staticmethod
    def sigmoid(x):
        '''Implement the sigmoid function using numpy here
        Sigmoid function is 1/(1+e^(-x))
        Numpy even has a exp function search for it.Eh?
        '''
        return 1 / (1 + np.exp(-x))
    
    def __init__(self):
        '''Creates a Feed-Forward Neural Network.
        "nodes_per_layer" is a list containing number of nodes in each layer (including input layer)
        "num_layers" is the number of layers in your network 
        "input_shape" is the shape of the image you are feeding to the network
        "output_shape" is the number of probabilities you are expecting from your network'''

        self.num_layers = 3 # includes input layer
        self.nodes_per_layer = [784,30,10] 
        self.input_shape = 784
        self.output_shape = 10
        self.__init_weights(self.nodes_per_layer)
        self.history= None

    def __init_weights(self, nodes_per_layer):
        '''Initializes all weights and biases between -1 and 1 using numpy'''
        self.weights_ = []
        self.biases_ = []
        for i,_ in enumerate(nodes_per_layer):
            if i == 0:
                # skip input layer, it does not have weights/bias
                continue
            W_h = np.random.normal(size=(nodes_per_layer[i-1], nodes_per_layer[i]))
            b_h = np.zeros(shape=(1,nodes_per_layer[i]))
            self.weights_.append(W_h)
            self.biases_.append(b_h)
    
    def fit(self, Xs, Ys, epochs, lr=1e-3):
        '''Trains the model on the given dataset for "epoch" number of itterations with step size="lr". 
        Returns list containing loss for each epoch.'''
        history = []
        for epoch in range(epochs):
            num_samples = Xs.shape[0]
            for i in range(num_samples):
                sample_input = Xs[i,:].reshape((1,self.input_shape))
                sample_target = Ys[i,:].reshape((1,self.output_shape))
                
                activations = self.forward_pass(sample_input)
                deltas = self.backward_pass(sample_target, activations)

                layer_inputs = [sample_input] + activations[:-1]
                self.weight_update(deltas, layer_inputs, lr)
            
            current_acc,current_loss,preds  = self.evaluate(Xs, Ys)
            print("----------------------------------")
            print("\nEpoch no: ",epoch)
            print("\nLoss: " ,100-current_acc )
            print("\nAccuracy: ",current_acc)
            print("\nnumber of corrctly identifed images: ",int(current_acc*len(preds)/100),"/" ,len(preds))
            print("----------------------------------")


            history.append(current_loss)
        return history
    
    
    
    def forward_pass(self, input_data):
        '''Executes the feed forward algorithm.
        "input_data" is the input to the network in row-major form
        Returns "activations", which is a list of all layer outputs (excluding input layer of course)
        What is activation?
        In neural network you have inputs(x) and weights(w).
        What is first layer? It is your input right?
        A linear neuron is this: y = w.T*x+b =>T is the transpose operator 
        A sigmoid neuron activation is y = sigmoid(w1.T*x+b1) for 1st hidden layer 
        Now for the last hidden layer the activation y = sigmoid(w2.T*y+b2).
        '''

        activations=[]
        l1=self.sigmoid(np.matmul(input_data,self.weights_[0])+self.biases_[0])
        activations.append(l1)
        l2=self.sigmoid(np.matmul(l1,self.weights_[1])+self.biases_[1])
        activations.append(l2)
        return activations
    
    def backward_pass(self, targets, layer_activations):
        '''Executes the backpropogation algorithm.
        "targets" is the ground truth/labels
        "layer_activations" are the return value of the forward pass step
        Returns "deltas", which is a list containing weight update values for all layers (excluding the input layer of course)
        You need to work on the paper to develop a generalized formulae before implementing this.
        Chain rule and derivatives are the pre-requisite for this part.
        '''
        deltas=[]
        bp1_s1=layer_activations[1]-targets
        bp1_s2=(layer_activations[1])*(1-layer_activations[1])
        bp1_s3=np.multiply(bp1_s2,bp1_s1)
        
        bp2_s1=np.transpose(np.matmul(self.weights_[1],np.transpose(bp1_s3)))
        bp2_s2=layer_activations[0]*(1-layer_activations[0])
        bp2_s3=np.multiply(bp2_s2,bp2_s1)
        
        deltas.append(bp2_s3)
        deltas.append(bp1_s3)
        return deltas
            
    def weight_update(self, deltas, layer_inputs, lr):
        '''Executes the gradient descent algorithm.
        "deltas" is return value of the backward pass step
        "layer_inputs" is a list containing the inputs for all layers (including the input layer)
        "lr" is the learning rate
        You just have to implement the simple weight update equation. 
        
        '''
        self.weights_[1]=self.weights_[1]-np.transpose(np.matmul(np.transpose(deltas[1]),layer_inputs[1]))*lr
        self.biases_[1]=self.biases_[1]-lr*(deltas[1])
        self.biases_[0]=self.biases_[0]-lr*(deltas[0])
        self.weights_[0]=self.weights_[0]-np.transpose(np.matmul(np.transpose(deltas[0]),layer_inputs[0]))*lr

        
    def predict(self, Xs):
        '''Returns the model predictions (output of the last layer) for the given "Xs".'''
        predictions = []
        num_samples = Xs.shape[0]
        for i in range(num_samples):
            sample = Xs[i,:].reshape((1,self.input_shape))
            sample_prediction = self.forward_pass(sample)[-1]
            predictions.append(sample_prediction.reshape((self.output_shape,)))
        return np.array(predictions)
    
    def evaluate(self, Xs, Ys):
        '''Returns appropriate metrics for the task, calculated on the dataset passed to this method.'''
        pred = self.predict(Xs)
        acc = self.accuracy(pred, Ys) 
        loss = self.cross_entropy_loss(pred, Ys)
        return acc,loss,pred
    def give_images(self,listDirImages,mode):
        '''Returns the images and labels from the listDirImages list after reading
        Hint: Use os.listdir(),os.getcwd() functions to get list of all directories
        in the provided folder. Similarly os.getcwd() returns you the current working
        directory. 
        For image reading use any library of your choice. Commonly used are opencv,pillow but
        you have to install them using pip
        "images" is list of numpy array of images 
        labels is a list of labels you read 
        '''
        if(mode=='training'):
            images =np.zeros((60000,784))
        elif(mode=="testing"):
            images=np.zeros((10000,784))
        labels = []
        i=0
        main_dir = os.getcwd() + '/' + listDirImages
        for sub_dir in os.listdir(main_dir):
            if "." not in sub_dir:
                filesInDir = os.listdir(main_dir + "/" + sub_dir)
                for file in filesInDir:
                    data = Image.open(main_dir + "/" + sub_dir + "/" + file,'r')
                    data=(np.asarray(data))
                    data=data.flatten()
                    data=(data-127.5/127.5)
                    data=(data - np.mean(data)) / np.std(data)
                    images[i]=data
                    labels.append(int(sub_dir))
                    i=i+1

        labels=self.generate_labels(labels,mode)
        z = list(zip(images,labels))
        random.shuffle(z)
        images, labels = zip(*z)
        images=np.array(images)
        labels=np.array(labels)
        return images,labels
    def generate_labels(self,labels,mode):
        '''Returns your labels into one hot encoding array
        labels is a list of labels [0,1,2,3,4,1,3,3,4,1........]
        Ex-> If label is 1 then one hot encoding should be [0,1,0,0,0,0,0,0,0,0]
        Ex-> If label is 9 then one hot encoding shoudl be [0,0,0,0,0,0,0,0,0,1]
        Hint: Use sklearn one hot-encoder to convert your labels into one hot encoding array
        "onehotlabels" is a numpy array of labels. In the end just do np.array(onehotlabels).
        '''
        if(mode=='training'):
            onehotlabels=np.zeros((60000,10))
        elif(mode=="testing"):
            onehotlabels=np.zeros((10000,10))
        labels=np.array(labels)
        for i,label in enumerate(labels):
            onehotlabels[i][label] = 1
        return (onehotlabels)
    def save_weights(self,fileName):
        '''save the weights of your neural network into a file
        Hint: Search python functions for file saving as a .txt'''
        data = {'weights': self.weights_, 'biases': self.biases_}
        file = open(fileName, 'wb')
        pickle.dump(data, file)
        file.close()
    def reassign_weights(self,fileName):
        '''assign the saved weights from the fileName to the network
        Hint: Search python functions for file reading
        '''
        file = open(fileName, 'rb')
        data = pickle.load(file)
        file.close()
        self.weights_ = data['weights']
        self.biases_ = data['biases']
    def savePlot(self,history,para):
        '''function to plot the execution time versus learning rate plot
        You can edit the parameters pass to the savePlot function'''
       
        plt.plot(history,para)
        plt.gca().set(xlabel='Learning rate', ylabel='Execution time', title = '')
        plt.show()
        print("plotted")

    

def train(dir_name,lrate):

        print("Training mode: ")
        nn = NeuralNetwork()
        print("\n-------------READING IMAGES--------------")
        images, labels = nn.give_images(dir_name,"training")
        print("\n---------------IMAGES LOADED-----------------\n")
        print('FIT function called: ')
        start_time = time.time()
        history = nn.fit(images,labels, epochs=2, lr=lrate)
        total = (time.time() - start_time) 
        print('Time spent on training:', total, 'secs')
        nn.save_weights("weights")
        "--------------------Program END-------------------"

def test(dir_name,fileName):
        print("Testing mode : ")
        nn = NeuralNetwork()
        print("\n-------------READING IMAGES--------------")
        images, labels = nn.give_images(dir_name,"testing")
        print("\n---------------IMAGES LOADED-----------------\n")
        nn.reassign_weights(fileName)
        print('\nTesting the model:  \n')
        start_time = time.time()
        accuracy, loss,preds = nn.evaluate(images, labels)
        total = (time.time() - start_time)/60
        print('Time spent on testing:', total, 'secs')
        print("\n///-------------Results-----------/////\n")
        print("Accuracy:",accuracy)

def train_plot(dir_name,l1,l2,l3):

        nn = NeuralNetwork()
        print("Plotting mode: ")
        print("\n-------------READING IMAGES--------------")
        images, labels = nn.give_images(dir_name,"testing")
        print("\n---------------IMAGES LOADED-----------------\n")
        print('First learning rate: ',l1,"\n")        
        start_time = time.time()
        history = nn.fit(images,labels, epochs=2, lr=l1)
        total1 = (time.time() - start_time) 
        print('Time spent on training:', total1, 'secs')
        nn.save_weights("weights")
        "--------------------Program END for learning rate 1-------------------"
        print('Second learning rate: ',l2,"\n")  
        nn = NeuralNetwork()      
        start_time = time.time()
        history = nn.fit(images,labels, epochs=2, lr=l2)
        total2 = (time.time() - start_time) 
        print('Time spent on training:', total2, 'secs')
        nn.save_weights("weights")
        "--------------------Program END for learning rate 1-------------------"
        print('Third learning rate: ',l3,"\n")        
        start_time = time.time()
        history = nn.fit(images,labels, epochs=2, lr=l3)
        total3 = (time.time() - start_time) 
        print('Time spent on training:', total3, 'secs')
        nn.save_weights("weights")
        "--------------------Program END for learning rate 1-------------------"
        nn.savePlot([total1,total2,total3],[l1,l2,l3])

def main():
    dir_name = sys.argv[2]
    
    if sys.argv[1] == 'plot':
        train_plot(dname,float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5]))

    elif sys.argv[1] == 'train':
        lrate = float(sys.argv[3])
        train(dir_name,lrate)
    elif sys.argv[1] == 'test':
        test(dir_name,'netWeights.txt')
    
main()

