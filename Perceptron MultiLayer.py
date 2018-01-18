
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
W_Hidden_Layer = torch.rand(var.N_FEATURES+1,30) # R 785*30
W_Exit_Layer = torch.rand(30,var.N_CLASSES) # R 30*10
Y = torch.Tensor(1, var.N_CLASSES)      # R 1*10

bias = torch.ones(1)

label = torch.Tensor(1, var.N_CLASSES)

prediction = torch.Tensor(1,var.N_CLASSES)

deltaLabel = torch.Tensor(1,var.N_CLASSES)

aux = torch.Tensor(var.N_FEATURES+1, 1)

deltaW  = torch.Tensor(var.N_FEATURES+1,var.N_CLASSES)

y1 = np.zeros((1, 1))

sigmoid_v = numpy.vectorize(functions.sigmoid)

# for i in range(var.N_IMAGES_TRAIN):
X = torch.cat((bias,train_data[1, :]), 0)
label =   train_data_label [1, :]
y1 = sigmoid_v(numpy.dot(X, W_Hidden_Layer)/785)

y1 = y1 [numpy.newaxis]

print(y1[1,:])

deltaLabel = numpy.add(label,prediction*-1)
X = X [numpy.newaxis] # 1D -> D2 array
deltaLabel = deltaLabel [numpy.newaxis]
aux = var.EPSILON * numpy.transpose(X)
deltaW =  numpy.dot(aux, deltaLabel)
# W = numpy.add(W,deltaW)
accurrancy= 0
# for i in range(var.N_IMAGES_TEST):
#     X = torch.cat((bias, train_data[i, :]), 0)
#     label = train_data_label[i, :]
#     prediction = numpy.dot(X, W) / (var.N_FEATURES+1)
#  # print("predicted %f label %f" % (numpy.argmax(prediction),numpy.argmax(label)))
#
#
#     if(numpy.argmax(prediction)==numpy.argmax(label)):
#         accurrancy+=1
#
# print("Valeurs bien predit: %d " % (accurrancy))
# print("Valeurs mal predit:  %d " % (var.N_IMAGES_TEST))
# print("Taux de reussite:    %f" % (accurrancy/var.N_IMAGES_TEST*100))
# print("Taux d'erreur:       %f" %  (100-(accurrancy/var.N_IMAGES_TEST*100)))

