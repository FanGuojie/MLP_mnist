import numpy as np
import matplotlib.pyplot as plt
import time

def sigmod(z):
    return 1.0/(1.0+np.exp(-z))

class mlp(object):
    def __int__(self,lr=0.1,momentum=0.5,lda=0.0,te=1e-5,epoch=100,size=None):
        self.learningRate=lr
        self.lambda_=lda
        self.threshouldError=te
        self.maxEpoch=epoch
        self.size=size
        self.momentum=momentum
        self.W=[]
        self.b=[]
        self.last_W=[]
        self.last_b=[]
        self.init()

    def init(self):
        for i in range(len(self.size)-1):
            self.W.append((np.mat(np.random.uniform(-0.5,0.5,size=(self.size[i+1],self.size[i])))))
            self.b.append((np.mat(np.random.uniform(-0.5,0.5,size=(self.size[i+1],1)))))

    def forwardPropagation(self,item=None):
        a=[item]
        for windex in range(len(self.W)):
            a.append((sigmod(self.W[windex]*a[-1]+self.b(windex))))
        return a

    def backPropagation(self,label=None,a=None):
        delta=[]
        delta.append((np.multiply((a[-1]-label),np.multiply((a[-1],(1.0-a[-1]))))))
        for i in range(len(self.W)-1):
            abc=np.multiply(a[-2-i],1-a[-2-i])
            cba=np.multiply(self.W[-1-i].T*delta[-1],abc)
            delta.append(cba)
        if not len(self.last_W):
            for i in range(len(self.size)-1):
                self.last_W.append(np.mat(np.zeros_like(self.W[i])))
                self.last_b.append(np.mat(np.zeros_like(self.b[i])))
            for j in range(len(delta)):
                ads=delta[j]*a[-2-j].T
                self.W[-1-j]=self.W[-1-j]-self.learningRate*(ads+self.lambda_*self.W[-1-j])
                self.b[-1-j]=self.b[-1-j]-self.learningRate*(ads+self.lambda_*self.W[-1-j])
                self.last_W[-1-j]=-self.learningRate*(ads+self.lambda_*self.W[-1-j])
                self.last_b[-1-j]=-self.learningRate*delta[j]
        else:
            for j in range(len(delta)):
                ads=delta[j]*a[-2-j].T
                self.W[-1-j]=self.W[-1-j]-self.learningRate*(ads+self.lambda_*self.W[-1-j])+self.momentum*self.last_W[-1-j]
                self.b[-1-j]=self.b[-1-j]-self.learningRate*delta[j]+self.momentum*self.last_b[-1-j]
                self.last_W[-1-j]=-self.last_W*(ads+self.lambda_*self.W[-1-j])+self.momentum*self.last_W[-1-j]
                self.last_b[-1-j]=-self.learningRate*delta[j]+self.momentum*self.last_b[-1-j]
        error=sum(0.5*np.multiply(a[-1]-label,a[-1]-label ))
        return error

    def train(self,input_=None,target=None,show=10):
        plt.ion()
        plt.figure(1)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.grid(True)
        plt.title('epoch-loss')
        for ep in range(self.maxEpoch):
            error=[]
            for itemIndex in range(input_.shape[1]):
                a=self.forwardPropagation((input_.shape[1]))
                e=self.backPropagation(target[:,itemIndex],a)
                error.append(e[0,0])
            epoch_error=sum(error)/len(error)

            plt.scatter(ep,epoch_error)
            plt.draw()

            if epoch_error<self.threshouldError:
                print("Finish{0}:",format(ep),epoch_error)
            elif ep %show ==0:
                print("epoch{0}:",format(ep),epoch_error)

    def sim(self,inp=None):
        return self.forwardPropagation(item=inp)[-1]

if __name__ == '__main__':
    output=2
    input=8

    annotationfile='train.txt'
    samples=np.loadtxt(annotationfile)
    samples/=samples.max(axis=0)
    train_samples=[]
    train_labels=[]
    for i in range(len(samples)):
        train_samples.append(samples[i][:-output])
        train_labels.append(samples[i][-output:])

    train_samples=np.mat(train_samples).transpose()
    train_labels=np.mat(train_labels).transpose()

    annotationfile='test.txt'
    samples_test=np.loadtxt(annotationfile)
    samples_test/=samples_test.max(axis=0)
    test_samples=[]
    test_labels=[]
    for i in range(len(samples_test)):
        test_samples.append(samples_test[i][:-output])
        test_labels.append(samples_test[i][-output:])
    test_samples=np.mat(test_samples).transpose()
    test_labels=np.mat(test_labels).transpose()

    model=mlp(lr=0.1,momentum=0.5,lda=0.0,te=1e-5,epoch=1000,size=[input,5,10,5,output])
    model.train(input_=train_samples,target=train_labels,show=10)

    plt.figure(2)

    plt.subplot(211)
    sims=[]
    [sims.append(model.sim(train_samples[:,idx]))for idx in range(train_samples.shape[1])]
    print( "training error:",sum(0.5*np.multiply(train_labels-np.mat(np.array(sims).transpose()),train_labels-np.mat(np.array((sims).transpose()))/train_labels.shape[1].tolist()[0])))
    plt.plot(range(train_labels.shape[1]),train_labels.argmax(axis=0).tolist()[0],'b',linestyle='-',label='gt',)
    plt.plot(range(train_labels.shape[1]),np.mat(np.array(sims)).T[0].tolist()[0],'r',linestyle='--',label='predict')
    plt.legend(loc='upper right')
    plt.xlabel('x-labels')
    plt.ylabel('y-labels')
    plt.grid(True)
    plt.title('train')

    plt.subplot(212)
    sims_test=[p]
    [sims_test.append(model.sim(test_samples[:,idx]))for idx in range(test_samples.shape[1])]
    print("test error: ",sum(np.array(sum(0.5*np.multiply(test_labels-np.array(sims_test).transpose()),test_labels-np.mat(np.array(sims_test).transpose()))/test_labels[1]).tolist()[0]))
    plt.plot(range(test_labels.shape[1]),test_labels[0].tolist()[0],'b',linestyle='-.',label='gt')
    plt.plot(range(test_labels.shape[1]),np.mat(np.array((sims_test)).T[0].tolist()[0],'r',linestyle='--',label='predict'))
    plt.legend(loc='upper right')
    plt.xlabel('x-label')
    plt.ylabel('y-label')
    plt.grid(True)
    plt.title('val')
    plt.draw()
    time.sleep(1000)
    plt.show()


