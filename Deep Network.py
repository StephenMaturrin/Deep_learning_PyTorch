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

dtype = torch.FloatTensor

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
W_Entry_Layer = torch.rand(var.N_FEATURES+1,400).uniform_(-0.1,0.1) # R 785*30
W_Hidden_Layer = torch.rand(400+1,var.N_CLASSES).uniform_(-1,1) # R 30*10
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
    #########################################################################
    X = torch.cat((bias,train_data[i, :]), 0)
    X= X [numpy.newaxis]
    label =   train_data_label [i, :]
    Z2 = numpy.dot(X,W_Entry_Layer) #R 1*30 = R 1*785 . R 785*30
    a2 = sigmoid_v(Z2)   # R 1*31
    a2 = numpy.concatenate((bias,a2[0,:]),0)  #R 1*31 a2[0, :]
    a2 = a2[numpy.newaxis]
    a2 = torch.from_numpy(a2)
    Z3= numpy.dot(a2,W_Hidden_Layer)# R 1*10 = R 1*31 R 31*10
    Z3 = torch.from_numpy(Z3)
    ################################################################################
    delta2 = numpy.add(label,Z3*-1)  #R 1*10
    aux = DFsigmoid_v(Z2)
    aux = torch.from_numpy(aux)  #R 1*30
    delta1 = numpy.dot(delta2,numpy.transpose((W_Hidden_Layer[1:,:]))) # R 1* 30 = R 1*10 . R 10*31
    delta1 = torch.from_numpy(delta1) # R 1 * 31
    # delta2 = delta2 [numpy.newaxis] # 1D -> D2 array
    delta1= numpy.multiply(delta1,aux) #delta2[0,1:31]
    delta_W_Entry_Layer = var.EPSILON * numpy.dot(numpy.transpose((X)),delta1)
    delta_W_Entry_Layer = torch.from_numpy(delta_W_Entry_Layer)
    W_Entry_Layer = numpy.add(delta_W_Entry_Layer,W_Entry_Layer)
    Z2 = torch.from_numpy(Z2)
    delta_W_Hidden_Layer =  var.EPSILON * numpy.dot(numpy.transpose(Z2),delta2)    #R 1*30 #R 1*10
    delta_W_Hidden_Layer = torch.from_numpy(delta_W_Hidden_Layer)
    W_Hidden_Layer[1:,:] = numpy.add(delta_W_Hidden_Layer,W_Hidden_Layer[1:,:])#  W_Hidden_Layer[1,:]



accurrancy= 0
for i in range(var.N_IMAGES_TEST):
    X = torch.cat((bias, test_data[i, :]), 0)
    label = test_data_label[i, :]

    print("predicted %f label %f" % (numpy.argmax(Z3), numpy.argmax(label)))
    if (numpy.argmax(Z3) == numpy.argmax(label)):
        accurrancy += 1




print("Valeurs bien predit: %d " % (accurrancy))
print("Valeurs mal predit:  %d " % (var.N_IMAGES_TEST))
print("Taux de reussite:    %f" % (accurrancy/var.N_IMAGES_TEST*100))
print("Taux d'erreur:       %f" %  (100-(accurrancy/var.N_IMAGES_TEST*100)))
    #

# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs, and wrap them in Variables.
# Setting requires_grad=False indicates that we do not need to compute gradients
# with respect to these Variables during the backward pass.
x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

# Create random Tensors for weights, and wrap them in Variables.
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Variables during the backward pass.
w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y using operations on Variables; these
    # are exactly the same operations we used to compute the forward pass using
    # Tensors, but we do not need to keep references to intermediate values since
    # we are not implementing the backward pass by hand.
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # Compute and print loss using operations on Variables.
    # Now loss is a Variable of shape (1,) and loss.data is a Tensor of shape
    # (1,); loss.data[0] is a scalar value holding the loss.
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.data[0])

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Variables with requires_grad=True.
    # After this call w1.grad and w2.grad will be Variables holding the gradient
    # of the loss with respect to w1 and w2 respectively.
    loss.backward()

    # Update weights using gradient descent; w1.data and w2.data are Tensors,
    # w1.grad and w2.grad are Variables and w1.grad.data and w2.grad.data are
    # Tensors.
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    # Manually zero the gradients after updating weights
    w1.grad.data.zero_()
    w2.grad.data.zero_()