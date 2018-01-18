
from __future__ import print_function
from __future__ import division
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




X = torch.Tensor(1, var.N_FEATURES + 1)  # R 1*785
W_Entry_Layer = torch.rand(var.N_FEATURES+1,30).uniform_(-1,1) # R 785*30
W_Hidden_Layer = torch.rand(30+1,var.N_CLASSES).uniform_(-1,1) # R 30*10
Y = torch.Tensor(1, var.N_CLASSES)      # R 1*10
y1 = torch.Tensor(1, 31)
y2 = torch.Tensor(1, 10)

bias = torch.ones(1)

label = torch.Tensor(1, var.N_CLASSES)

prediction = torch.Tensor(1,var.N_CLASSES)

deltaLabel = torch.Tensor(1,var.N_CLASSES)

aux = torch.Tensor(var.N_FEATURES+1, 1)

deltaW  = torch.Tensor(var.N_FEATURES+1,var.N_CLASSES)



sigmoid_v = numpy.vectorize(functions.sigmoid)
DFsigmoid_v = numpy.vectorize(functions.DFsigmoid)

for i in range(var.N_IMAGES_TRAIN):
    X = torch.cat((bias,train_data[i, :]), 0)
    X= X [numpy.newaxis]
    label =   train_data_label [i, :]
    Z2 = numpy.dot(X,W_Entry_Layer) # R 1*30
    a2 = sigmoid_v(Z2)   # R 1*30
    a2 = numpy.concatenate((bias,a2[0, :]),0)  # R 31*1
    Z3= numpy.dot(a2,W_Hidden_Layer)# R 1*10
    delta3 = numpy.add(label,Z3*-1)
    delta3 = delta3
    aux = DFsigmoid_v(Z2)
    delta2 = numpy.dot(delta3,numpy.transpose((W_Hidden_Layer)))
    delta2 = delta2 [numpy.newaxis] # 1D -> D2 array
    delta2 = numpy.multiply(delta2[0,1:31],aux)
    delta_W_Entry_Layer = var.EPSILON * numpy.dot(numpy.transpose((X)),delta2)
    W_Entry_Layer = numpy.add(delta_W_Entry_Layer,W_Entry_Layer)
    delta3 = delta3 [numpy.newaxis]
    delta_W_Hidden_Layer =  var.EPSILON * numpy.dot(numpy.transpose((Z2)),delta3)
    W_Hidden_Layer [1:,:] = numpy.add(delta_W_Hidden_Layer,W_Hidden_Layer[1,:])

    accurrancy= 0
    for i in range(var.N_IMAGES_TEST):
        X = torch.cat((bias, test_data[i, :]), 0)
        X = X[numpy.newaxis]
        label = test_data_label[i, :]
        Z2 = numpy.dot(X, W_Entry_Layer)  # R 1*30
        a2 = sigmoid_v(Z2)  # R 1*30
        a2 = numpy.concatenate((bias, a2[0, :]), 0)  # R 31*1
        Z3 = numpy.dot(a2, W_Hidden_Layer)  # R 1*10
        print(Z3)
        print("predicted %f label %f" % (numpy.argmax(Z3), numpy.argmax(label)))
        if (numpy.argmax(Z3) == numpy.argmax(label)):
            accurrancy += 1




    print("Valeurs bien predit: %d " % (accurrancy))
    print("Valeurs mal predit:  %d " % (var.N_IMAGES_TEST))
    print("Taux de reussite:    %f" % (accurrancy/var.N_IMAGES_TEST*100))
    print("Taux d'erreur:       %f" %  (100-(accurrancy/var.N_IMAGES_TEST*100)))

