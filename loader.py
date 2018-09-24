import os,struct
import numpy as np

def load_mnist(path,kind='train'):
    labels_path=os.path.join('data',path,'%s-labels-idx1-ubyte' % kind)
    images_path=os.path.join('data',path,'%s-images-idx3-ubyte' % kind)
    with open(labels_path,'rb') as lpath:
        magic,n=struct.unpack('>II',lpath.read(8))
        # print(magic,n)
        labels=np.fromfile(lpath,dtype=np.uint8)
        # print(labels.shape)

    with open(images_path,'rb')as imgpath:
        magic,num,rows,cols=struct.unpack('>IIII',imgpath.read(16))
        images=np.fromfile(imgpath,dtype=np.int8).reshape(len(labels),784)
        images=images/255.0#归一化
        # print(np.max(images))
    return  images,labels
if __name__ == '__main__':
    load_mnist("./mnist/")
