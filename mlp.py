import numpy as np
from scipy.special import expit as sigmoid
from sklearn.utils.extmath import softmax
from constants import *
import loader

def delta(a, y):
    return (a-y)


def derivative(z, fn):
    if fn==SIGMOID:
        f=sigmoid
    else:
        f=softmax
    return f(z)*(1-f(z))


class MLP(object):
    def __init__(self,layers,random_state=None):
        np.random.seed(random_state)
        self.num_layers=len(layers)
        self.layers=layers
        self.initialize_weights()

    def initialize_weights(self):
        self.biases=[np.random.randn(y,1) for y in self.layers[1:]]
        self.weights=[np.random.randn(y,x)/np.sqrt(x) for x,y in zip(self.layers[:-1],self.layers[1:])]

    def fit(self,training_data,l1=0.0,l2=0.0,epochs=500,eta=0.001,minibatches=1,regularizaiton=L2):
        self.l1=l1
        self.l2=l2
        n=len(training_data)
        for epoch in range(epochs):
            np.random.shuffle(training_data)
            mini_batches=[training_data[k:k+100] for k in range(0,n,minibatches)]
            for mini_batch in mini_batches:
                self.batch_update(mini_batch,eta,len(training_data),regularizaiton)

    def batch_update(self, mini_batch, eta, n, regularizaiton):
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b,delta_nabla_w=self.back_propogation(x,y)
            nabla_b=[nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w=[nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
        self.biases=[b-(eta/len(mini_batch))*nb for b,nb in zip(self.biases,nabla_b)]
        if regularizaiton==L2:
            self.weights=[(1-eta*(self.l2/n))*w -(eta/len(mini_batch))*nw for w,nw in zip(self.weights,nabla_w)]
        elif regularizaiton==L1:
            self.weights=[w-eta*self.l1*np.sign(w)/n-(eta/len(mini_batch))*nw for w,nw in zip(self.weights,nabla_w)]

    def back_propogation(self, x, y,fn=SIGMOID):
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape)for w in self.weights]
        activation =x
        activations=[x]
        zs=[]
        for b,w in zip(self.biases,self.weights):
            z=np.dot(w,activation)+b
            zs.append(z)
            if fn==SIGMOID:
                activation=sigmoid(z)
            else:
                activation=softmax(z)
            activations.append(activation)
        dell=delta(activations[-1],y)
        nabla_b[-1]=dell
        nabla_w[-1]=np.dot(dell,activations[-2].transpose())
        for l in range(self.num_layers-2,0,-1):
            dell=np.dot(self.weights[l+1].transpose(),dell)*derivative(zs[l],fn)
            nabla_b[-l]=dell
            nabla_w[-l]=np.dot(dell,activations[-l-1].transpose())
        return (nabla_b,nabla_w)

    def cross_entropy_loss(self,a,y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    def log_likelihood_loss(self,a,y):
        return -np.dot(y,softmax(a).transpose())

if __name__ == '__main__':
    images,labels=loader.load_mnist("./mnist/")
    mlp=MLP([300,10])
    mlp.fit(images)


