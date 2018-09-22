import os,struct
import numpy as np
def one_hot(y):
    y_one_hot = np.zeros((len(y), 10))
    for i, label in enumerate(y):
        y_one_hot[i, label] = 1
    return y_one_hot

def load_mnist(path,kind='train'):
    labels_path=os.path.join('data',path,'%s-labels-idx1-ubyte' % kind)
    images_path=os.path.join('data',path,'%s-images-idx3-ubyte' % kind)
    with open(labels_path,'rb') as lpath:
        magic,n=struct.unpack('>II',lpath.read(8))
        # print(magic,n)
        labels=np.fromfile(lpath,dtype=np.uint8)
        # print(labels.shape)
    onehotlabels=one_hot(labels)
    with open(images_path,'rb')as imgpath:
        magic,num,rows,cols=struct.unpack('>IIII',imgpath.read(16))
        images=np.fromfile(imgpath,dtype=np.int8).reshape(len(labels),784)
        images=images/255.0#归一化
        # print(np.max(images))
    return  images,onehotlabels
if __name__ == '__main__':
    load_mnist("./mnist/")