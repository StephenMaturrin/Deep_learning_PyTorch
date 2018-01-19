from __future__ import print_function
from __future__ import division
from torch.autograd import Variable
import  functions
import gzip # pour décompresser les données
import numpy # pour pouvoir utiliser des matrices
import matplotlib.pyplot as plt # pour l'affichage
import torch,torch.utils.data
import pickle  # pour désérialiser les données
# fonction qui va afficher l'image située à l'index index
import  variables as var


if __name__ == '__main__':
    # nombre d'image lues à chaque fois dans la base d'apprentissage (laisser à 1 sauf pour la question optionnelle sur les minibatchs)
    TRAIN_BATCH_SIZE = 1


    with gzip.open('mnist.pkl.gz','rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()

    # print(p)
    # on charge les données de la base MNIST
    # data = pickle.load(gzip.open('mnist.pkl.gz'))
    # images de la base d'apprentissage [torch.FloatTensor of size 63000x784]
    train_data = torch.Tensor(data[0][0])


    # labels de la base d'apprentissage [torch.FloatTensor of size 63000x784]
    train_data_label = torch.Tensor(data[0][1])


    # images de la base de test [torch.FloatTensor of size 7000x784]
    test_data = torch.Tensor(data[1][0])



    # labels de la base de test [torch.FloatTensor of size 7000x10]
    test_data_label = torch.Tensor(data[1][1])

    # on crée la base de données d'apprentissage (pour torch)
    train_dataset = torch.utils.data.TensorDataset(train_data,train_data_label)
    # on crée la base de données de test (pour torch)
    test_dataset = torch.utils.data.TensorDataset(test_data,test_data_label)
    # on crée le lecteur de la base de données d'apprentissage (pour torch)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    # on crée le lecteur de la base de données de test (pour torch)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    # 10 fois


sigmoid_v = numpy.vectorize(functions.sigmoid)
DFsigmoid_v = numpy.vectorize(functions.DFsigmoid)



dtype = torch.FloatTensor
N, D_in, H, D_out = var.N_IMAGES_TRAIN,var.N_FEATURES, 300, var.N_CLASSES

x = Variable(train_data , requires_grad=False)
y = Variable(train_data_label, requires_grad=False)
w1 = Variable(torch.randn(D_in, H).uniform_(-0.1,0.1).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).uniform_(-0.1,0.1).type(dtype), requires_grad=True)
learning_rate = 5e-6
for t in range(100):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    loss = (y_pred - y).pow(2).sum()
    print(t, loss.data[0])
    loss.backward()

    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data
    w1.grad.data.zero_()
    w2.grad.data.zero_()


accurrancy= 0

x = Variable(test_data , requires_grad=False)
y = Variable(test_data_label, requires_grad=False)



y_pred = x.mm(w1).clamp(min=0).mm(w2)

for i in range(var.N_IMAGES_TEST):
    d = y_pred[i,:]
    valuesx, indicesx = torch.max(d, 0)
    indices2 = numpy.argmax(test_data_label[i, :])
    indices1 =  indicesx.data.numpy()[0]
    print("predicted %f label %f" % (indices1,indices2  ))
    if (indices1==indices2):
        accurrancy += 1

print("Valeurs bien predit: %d " % (accurrancy))
print("Valeurs mal predit:  %d " % (var.N_IMAGES_TEST))
print("Taux de reussite:    %f" % (accurrancy/var.N_IMAGES_TEST*100))
print("Taux d'erreur:       %f" %  (100-(accurrancy/var.N_IMAGES_TEST*100)))