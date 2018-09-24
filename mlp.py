import numpy as np
from scipy.special import expit as sigmoid
from constants import *
import loader

def delta(a, y):
    y=y[:,np.newaxis]
    return (a-y)


def derivative(z, fn):
    if fn==SIGMOID:
        f=sigmoid
    else:
        f=softmax
    return np.multiply(f(z),(1-f(z)))


def softmax(z):
    np.exp(z,z)
    sum=np.sum(z)
    z/=sum
    return z


class MLP(object):
    def __init__(self,layers,random_state=None):
        np.random.seed(random_state)
        self.num_layers=len(layers)
        self.layers=layers
        self.initialize_weights()

    def initialize_weights(self):
        self.biases=[np.random.randn(y,1) for y in self.layers[1:]]
        self.weights=[np.random.randn(y,x)/np.sqrt(x) for x,y in zip(self.layers[:-1],self.layers[1:])]

    def fit(self,training_data,l1=0.0,l2=0.0,epochs=20,eta=0.001,minibatches=50,batchsize=50,regularizaiton=L2):
        self.l1=l1
        self.l2=l2
        n=len(training_data)
        for epoch in range(epochs):
            # if(epoch%10==0):
            print("epoch:%d"%(epoch))
            np.random.shuffle(training_data)
            mini_batches=[training_data[k:k+batchsize] for k in range(0,n,n//minibatches)]
            for mini_batch in mini_batches:
                self.batch_update(mini_batch,eta,len(training_data),regularizaiton)

    def batch_update(self, mini_batch, eta, n, regularizaiton):
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]

        for data in mini_batch:
            x = data[:-10]
            y = data[-10:]
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
        activation =x[:, np.newaxis]
        activations=[np.mat(x)]
        zs=[]
        # print("forward calculate")
        for b,w in zip(self.biases,self.weights):
            z=np.dot(w,activation)+b
            zs.append(z)
            if len(activations)<self.num_layers-1:
                activation=sigmoid(z)
            else:
                activation=softmax(z)
            activations.append(activation)
        # print("forward end, backward begin")
        dell=np.multiply(delta(activations[-1],y),derivative(zs[-1],fn=SOFTMAX))
        nabla_b[-1]=dell
        nabla_w[-1]=np.dot(dell,activations[-2].transpose())
        for l in range(self.num_layers-3,-1,-1):
            dell=np.dot(self.weights[l+1].T,np.multiply(derivative(zs[l+1],fn),dell))
            nabla_b[l]=dell
            nabla_w[l]=np.dot(dell,activations[l])
        return (nabla_b,nabla_w)

    def predict(self,testdata):
        images=testdata[:,:-1]
        labels=testdata[:,-1]
        n=testdata.shape[0]
        total_score=0
        test=images
        for i in range(self.num_layers-2):
            test=sigmoid(np.dot(self.weights[i],test.T)+self.biases[i])
        temp=np.dot(self.weights[-1],test)+self.biases[-1]
        for i in range(n):
            temp[:,i]=softmax(temp[:,i])
        test=temp
        for i in range(n):
            result=np.argmax(test[:,i] )
            total_score+=(result==labels[i])
        print("corrent rate: %.5f "%(100*total_score/n),"%")


    def cross_entropy_loss(self,a,y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    def log_likelihood_loss(self,a,y):
        return -np.dot(y,softmax(a).transpose())


def one_hot(y):
    y_one_hot = np.zeros((len(y), 10))
    for i, label in enumerate(y):
        y_one_hot[i, label] = 1
    return y_one_hot

if __name__ == '__main__':
    images,labels=loader.load_mnist("mnist/")
    onehotlabels=one_hot(labels)
    trainingData=np.hstack((images,onehotlabels))
    mlp=MLP([784,300,10])
    mlp.fit(trainingData)
    timg,tlab=loader.load_mnist("mnist/",kind="t10k")
    tlab=tlab[:,np.newaxis]
    print(len(timg),len(tlab))
    testData=np.hstack((timg,tlab))
    print(testData.shape)
    test=testData[:10000]
    mlp.predict(test)


