import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
from torch.autograd import grad
import os
import numpy as np
import random
import torch.nn.init as init
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import host_subplot

import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt

plt.style.use( 'seaborn-whitegrid' )
import re



RANDOMIZATION = [ 0 ,0,0,0,0  ,0.05,0.05,0.05,0.05,0.05, 0.1, 0.1,0.1,0.1,0.1, 0.15, 0.15,0.15,0.15,0.15, 0.2 ,0.2 ,0.2 ,0.2 ,0.2 ,0.25,0.25,0.25,0.25,0.25, 0.3, 0.3, 0.3, 0.3, 0.3,0.35,0.35,0.35,0.35,0.35, 0.4,0.4,0.4,0.4,0.4, 0.45,0.45,0.45,0.45,0.45, 0.5, 0.5, 0.5, 0.5, 0.5,0.6,0.6,0.6,0.6,0.6,0.7,0.7,0.7,0.7,0.7,0.8,0.8,0.8,0.8,0.8,0.9,0.9,0.9,0.9,0.9,1.0,1.0,1.0,1.0,1.0]


BATCH_SIZE=200 # for fair loss values during training, take batch sizes that divide 10.000
LEARNING_RATE=0.03
TRAINING_EPOCHS=10000 # 150
# 0=weights first layer, 1=bias first layer, 2=weights second layer, 3= bias second layer, 4=weights last layer, 5=bias last layer
LAYERS_TO_COMPUTE=(2,4,6,8)
NO_OF_LAYERS=np.array(LAYERS_TO_COMPUTE).size


epsilon=1e-3 # stopping criteria for Jacobian
layer_id = 8 # layer for stopping criteria with small Jacobian

dir_path = "flatness"


all_results=np.zeros([len(RANDOMIZATION),NO_OF_LAYERS,6])

# loading and transforming the image
transform=transforms.Compose([transforms.ToTensor()])
train_data = torchvision.datasets.MNIST('.', train=True, transform=transform, download=True)
test_data = torchvision.datasets.MNIST('.', train=False, transform=transform, download=True)
print(train_data)



for seed in range(len(RANDOMIZATION)):

    # getting training data in an array
    reading_generator = data.DataLoader( train_data, batch_size=60000, shuffle=False )
    for b, l in reading_generator:
        train_X_tensor = b
        train_y_tensor = l
    train_X = []
    train_y = []
    for i in range( 60000 ):
        label_selector=train_y_tensor[i].flatten().data.numpy().tolist()
        if label_selector== 0 or label_selector== 1 or label_selector==2 or label_selector==3:

            train_X.append( train_X_tensor[i].flatten().data.numpy() )
            train_y.append( label_selector)

    # getting testing data in an array
    reading_generator = data.DataLoader( test_data, batch_size=10000, shuffle=False )
    for b, l in reading_generator:
        test_X_tensor = b
        test_y_tensor = l
    test_X = []
    test_y = []
    for i in range( 10000 ):
        label_selector = test_y_tensor[i].flatten().data.numpy().tolist()
        if label_selector == 0 or label_selector == 1 or label_selector==2 or label_selector==3:
            test_X.append( test_X_tensor[i].flatten().data.numpy() )
            test_y.append(label_selector )


    # Randomizing of network labels,
    train_dataset_size = len(train_y)
    test_dataset_size = len(test_y)
    labels_to_change = np.random.choice( train_dataset_size, int( train_dataset_size * RANDOMIZATION[seed] ), replace=False )
    for i in labels_to_change:
        add = np.random.randint( 3 ) + 1
        train_y[i] = (train_y[i] + add) % 4

    labels_to_change_test = np.random.choice( test_dataset_size, int( test_dataset_size * RANDOMIZATION[seed] ), replace=False )
    for i in labels_to_change_test:
        add = np.random.randint( 3 ) + 1
        test_y[i] = (test_y[i] + add) % 4

    # Putting it back into a pytorch Dataset
    train_y_randomized_Tensor = torch.LongTensor( train_y )
    train_x_selected_Tensor = torch.Tensor(train_X)
    train_data_randomized = data.TensorDataset( train_x_selected_Tensor, train_y_randomized_Tensor )

    test_y_randomized_Tensor = torch.LongTensor( test_y )
    test_x_selected_Tensor = torch.Tensor(test_X)
    test_data_randomized = data.TensorDataset( test_x_selected_Tensor, test_y_randomized_Tensor )



    # Network specification

    class MnistFCNet(nn.Module):
        def __init__(self):
            torch.manual_seed( 100*seed+20 )
            torch.cuda.manual_seed_all( 100*seed+20 )
            np.random.seed( 100*seed+20 )
            random.seed( 100* seed+20 )

            super(MnistFCNet, self).__init__()
            torch.backends.cudnn.deterministic=True

            self.fc1 = nn.Linear(784, 100)
            init.xavier_normal_(self.fc1.weight.data)
            init.zeros_(self.fc1.bias.data)
            self.fc2 = nn.Linear(100, 100)
            init.xavier_normal_(self.fc2.weight.data)
            init.zeros_(self.fc2.bias.data)
            self.fc3 = nn.Linear(100, 100)
            init.xavier_normal_(self.fc3.weight.data)
            init.zeros_(self.fc3.bias.data)
            self.fc4 = nn.Linear( 100, 100 )
            init.xavier_normal_( self.fc4.weight.data )
            init.zeros_( self.fc4.bias.data )
            self.fc5 = nn.Linear( 100, 10 )
            init.xavier_normal_( self.fc5.weight.data )
            init.zeros_( self.fc5.bias.data )


        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = self.fc5(x)
            return x



    loss = nn.CrossEntropyLoss()


    # Training
    training_generator = data.DataLoader( train_data_randomized, batch_size=BATCH_SIZE, shuffle=True )
    testing_generator = data.DataLoader( test_data_randomized, batch_size=BATCH_SIZE, shuffle=False )


    net = MnistFCNet().cuda()

    learning_rate=LEARNING_RATE

    update_rule = optim.SGD( net.parameters(), lr=learning_rate )


    epoch = 0
    train = True
    while train:

        if epoch %500 == 0 and epoch >0:
            learning_rate*=0.5

        if epoch % 10==0:
            print(learning_rate)

        # Training

        epoch_loss = 0

        grads = None

        for local_batch, local_labels in training_generator:
            batch = []
            for e in local_batch:
                batch.append( e.flatten().data.numpy() )
            update_rule.zero_grad()  # zero the gradient buffers
            output = net( torch.cuda.FloatTensor( np.array( batch ) ) )
            l = loss( output, torch.cuda.LongTensor(np.array( local_labels ) ) )
            l.backward()
            update_rule.step()  # Does the update
            epoch_loss += l.data.cpu().numpy().tolist()


            i=0
            for p in net.parameters():
                if i==layer_id:
                    if grads is None:
                        grads = p.grad.data.cpu().numpy()
                    else:
                        grads += p.grad.data.cpu().numpy()
                i += 1



        all_grads_small = True
        for g in grads.flatten():
            all_grads_small = all_grads_small and (abs(g) <= epsilon )
        if all_grads_small:
            train = False

        if epoch %10 == 0:
            print( "Epoch:", epoch + 1, ", loss:", 1.0*epoch_loss * BATCH_SIZE / train_dataset_size )


        # Validation

        with torch.set_grad_enabled( False ):
            epoch_loss = 0
            for local_batch, local_labels in testing_generator:
                batch = []
                for e in local_batch:
                    batch.append( e.flatten().data.numpy() )
                output = net( torch.cuda.FloatTensor( np.array( batch ) ) )
                l = loss( output, torch.cuda.LongTensor( np.array( local_labels ) ) )
                epoch_loss += l.data.cpu().numpy().tolist()
                val_loss = 1.0*epoch_loss * np.min([BATCH_SIZE,test_dataset_size]) / test_dataset_size


            correct = 0
            for local_batch, local_labels in training_generator:
                batch = []
                for e in local_batch:
                    batch.append( e.flatten().data.numpy() )
                output = net( torch.cuda.FloatTensor( np.array( batch ) ) )
                _, predicted = torch.max( output.data, 1 )
                ground_truth = torch.cuda.LongTensor(local_labels.cuda())

                correct += (predicted == ground_truth).sum().item()

            if epoch % 10 ==0:
                print( "Validation loss", val_loss )

                print( correct )
                print( 'Accuracy of the network on train: %d %%' % (
                    100.0 * correct / train_dataset_size) )


            epoch +=1

            #if epoch == 80:
            #    torch.save( net.state_dict(), "{}/modelsRandom/models_at_80/seed_{}_batchsize_{}.txt".format(dir_path, seed, BATCH_SIZE ) )
            #if epoch == 120:
            #    torch.save( net.state_dict(), "{}/modelsRandom/models_at_120/seed_{}_batchsize_{}.txt".format(dir_path, seed, BATCH_SIZE ) )

    #torch.save( net.state_dict(), "{}/modelsRandom/seed_{}_batchsize_{}.txt".format(dir_path, seed, BATCH_SIZE ) )


    print("Collecting measures")

    ## test loss
    test_output = net(torch.cuda.FloatTensor(np.array(test_X)))
    test_loss = loss( test_output, torch.cuda.LongTensor(test_y ) )
    test_loss_overall = np.sum( test_loss.data.cpu().numpy() )
    print( "Test loss calculated: {}".format(test_loss_overall ) )


    ## train loss
    train_output = net(torch.cuda.FloatTensor(train_X))
    train_loss = loss( train_output, torch.cuda.LongTensor( train_y ) )
    train_loss_overall = np.sum( train_loss.data.cpu().numpy() )
    print( train_loss_overall )
    print( "Train loss calculated: {}".format(train_loss_overall) )

    generalization_error = test_loss_overall - train_loss_overall
    print("Generalization error calculated: {}".format(generalization_error))


    ## calculate Hessian of each layer weights

    train_loss = loss( train_output, torch.cuda.LongTensor( train_y  ) )

    l=0
    for layer_id in LAYERS_TO_COMPUTE:
        j=0

        for p in net.parameters():

            if layer_id == j:
                layer = p
            j += 1

        shape= layer.shape

        layer_jacobian = grad( train_loss, layer, create_graph=True, retain_graph=True )
        layer_jacobian_out= layer_jacobian[0]
        print(layer_jacobian_out.shape)

        c = 1
        values = []
        trace=0.0
        for neuron in range(shape[1]):
            neuron_weights=layer[:,neuron]


            hessian = []


            for n_grd in layer_jacobian_out:
                drv2 = grad( n_grd[neuron], layer, retain_graph=True )
                hessian.append( drv2[0][:,neuron].data.cpu().numpy().flatten() )


            if c%10==0:
                print( "Hessian number {} of size {}x{} calculated".format(c, hessian[0].shape,hessian[0].shape) )


            trace += np.trace( hessian )

            ## calculate w^THw
            neuron_weights_left = np.reshape(neuron_weights.detach().cpu().numpy(),(1,shape[0]))
            result= neuron_weights_left.dot(np.array(hessian)).dot(np.transpose(neuron_weights_left))[0][0]
            values.append(result)
            c += 1



        sqrd_norm = 0.0
        for n in layer.data.cpu().numpy():
            for w in n:
                sqrd_norm += w ** 2
        print( "squared euclidian norm is calculated", sqrd_norm )

        ## calculate the largest eigenvalue
        #max_eignv = LA.eigvalsh( hessian, eigvals=(hessian[0].shape[0]-1,hessian[0].shape[0]-1) )[-1]
        #print( "largest eigenvalue is", max_eignv )


        ## Calculate flatness measure for layer
        # rho^max
        flatness_max = np.max(np.array(values))
        print( "flatness_max is", flatness_max )

        #rho^l_sigma
        flatness_sigma = np.sum(values)
        print( "flatness_sigma is", flatness_sigma )

        # kappa_trace
        flatness_trace = trace * sqrd_norm
        print( "flatness_trace is", flatness_trace )


        all_results[seed][l][0]=test_loss_overall
        all_results[seed][l][1]=train_loss_overall
        all_results[seed][l][2]= sqrd_norm
        all_results[seed][l][3]=flatness_max
        all_results[seed][l][4]=flatness_sigma
        all_results[seed][l][5]=flatness_trace

        l+=1

        del hessian
        del layer_jacobian


        np.save(dir_path+"/results/all_results",all_results)

    del test_output
    del train_output
    del net

    torch.cuda.empty_cache()



    # Plotting



    results_folder = "results/"

    results = np.load( results_folder + "/all_results.npy" )

    selection = np.zeros( [16, 4, 6, 2] )

    for i in range( 16 ):
        mean = np.mean( results[5 * i:5 * i + 5, :, :], axis=0 )
        std_dev = np.std( results[5 * i:5 * i + 5, :, :], axis=0 )

        selection[i, :, :, 0] = mean
        selection[i, :, :, 1] = std_dev

    # print(selection)

    flatness_trace_sum = selection[:, 0, 5, 0]
    flatness_trace_sum_std = 1.0 / 4 * selection[:, 0, 5, 0]

    for layer in range( 1, 4 ):
        flatness_trace_sum += selection[:, layer, 5, 0]
        flatness_trace_sum_std += 1.0 / 4.0 * selection[:, layer, 5, 1]

    generalization_errors = selection[:, 0, 0, 0] - selection[:, 0, 1, 0]
    generalization_errors_std = np.zeros( [16] )

    for i in range( 16 ):
        std_dev = np.std( results[5 * i:5 * i + 5, 0, 0] - results[5 * i:5 * i + 5, 0, 1], axis=0 )

        generalization_errors_std[i] = std_dev

    squared_norm_0 = selection[:, 0, 2, 0]
    flatness_max_0 = selection[:, 0, 3, 0]
    flatness_sigma_0 = selection[:, 0, 4, 0]
    flatness_trace_0 = selection[:, 0, 5, 0]

    squared_norm_1 = selection[:, 1, 2, 0]
    flatness_max_1 = selection[:, 1, 3, 0]
    flatness_sigma_1 = selection[:, 1, 4, 0]
    flatness_trace_1 = selection[:, 1, 5, 0]

    squared_norm_2 = selection[:, 2, 2, 0]
    flatness_max_2 = selection[:, 2, 3, 0]
    flatness_sigma_2 = selection[:, 2, 4, 0]
    flatness_trace_2 = selection[:, 2, 5, 0]

    squared_norm_3 = selection[:, 3, 2, 0]
    flatness_max_3 = selection[:, 3, 3, 0]
    flatness_sigma_3 = selection[:, 3, 4, 0]
    flatness_trace_3 = selection[:, 3, 5, 0]

    squared_norm_std_0 = selection[:, 0, 2, 1]
    flatness_max_std_0 = selection[:, 0, 3, 1]
    flatness_sigma_std_0 = selection[:, 0, 4, 1]
    flatness_trace_std_0 = selection[:, 0, 5, 1]

    squared_norm_std_1 = selection[:, 1, 2, 1]
    flatness_max_std_1 = selection[:, 1, 3, 1]
    flatness_sigma_std_1 = selection[:, 1, 4, 1]
    flatness_trace_std_1 = selection[:, 1, 5, 1]

    squared_norm_std_2 = selection[:, 2, 2, 1]
    flatness_max_std_2 = selection[:, 2, 3, 1]
    flatness_sigma_std_2 = selection[:, 2, 4, 1]
    flatness_trace_std_2 = selection[:, 2, 5, 1]

    squared_norm_std_3 = selection[:, 3, 2, 1]
    flatness_max_std_3 = selection[:, 3, 3, 1]
    flatness_sigma_std_3 = selection[:, 3, 4, 1]
    flatness_trace_std_3 = selection[:, 3, 5, 1]

    # plt.scatter(flatness_trace, generalization_errors)

    x_values = np.array( [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] )

    plt.figure( figsize=(5 * 1.6, 5) )
    mpl.rcParams.update( {'font.size': 12} )

    host = host_subplot( 111, axes_class=AA.Axes )
    plt.subplots_adjust( right=0.75 )

    par1 = host.twinx()
    par2 = host.twinx()
    par3 = host.twinx()

    offset = 40
    new_fixed_axis = par2.get_grid_helper().new_fixed_axis


    par2.axis["right"] = new_fixed_axis( loc="right",
                                         axes=par2,
                                         offset=(offset, 0) )

    par3.axis["right"] = new_fixed_axis( loc="right",
                                         axes=par3,
                                         offset=(2 * offset, 0) )

    par1.axis["right"].toggle( all=True )

    par2.axis["right"].toggle( all=True )

    par3.axis["right"].toggle( all=True )

    #################
    ### FOR TRACIAL MEASURE ###
    #################

    plt.xlim( [-0.007, 0.9] )
    plt.ylim( [0, 1300] )

    plt.xlabel( "Randomization" )
    plt.ylabel( "Tracial Flatness Layer 1" )
    par1.set_ylabel( "Tracial Flatness Layer 2" )
    par2.set_ylabel( "Tracial Flatness Layer 3" )
    par3.set_ylabel( "Generalization Gap x 10**4" )

    color = plt.cm.viridis( np.linspace( 0, 1, 4 ) )

    p1 = plt.errorbar( x_values, flatness_trace_0, yerr=flatness_trace_std_0, label="Layer 1", fmt='.-', color=color[1],
                       ecolor='lightskyblue', elinewidth=2, capsize=2 )
    p2 = par1.errorbar( x_values, flatness_trace_1, yerr=flatness_trace_std_1, label="Layer 2", fmt='.--',
                        color=color[2], ecolor='palegreen', elinewidth=2, capsize=2 )
    p3 = par2.errorbar( x_values, flatness_trace_2, yerr=flatness_trace_std_2, label="Layer 3", fmt='.:',
                        color=color[3], ecolor='#F4F27F', elinewidth=2, capsize=2 )
    p4 = par3.errorbar( x_values, generalization_errors, yerr=generalization_errors_std, label="Generalization gap",
                        fmt=':', linewidth=3, color=color[0], ecolor='mediumpurple', elinewidth=2, capsize=2 )

    par1.set_ylim( 0, 200 )
    par2.set_ylim( 0, 16 )
    par3.set_ylim( 0, 25 )

    #################
    ### FOR SIGMA ###
    #################

    # plt.xlabel("Randomization")
    # plt.ylabel("Sigma Flatness Layer 1")
    # par1.set_ylabel("Sigma Flatness Layer 2")
    # par2.set_ylabel("Sigma Flatness Layer 3")
    # par3.set_ylabel("Generalization Gap x 10**4")

    # color=plt.cm.viridis(np.linspace(0,1,4))

    # p1=plt.errorbar(x_values, flatness_sigma_0, yerr=flatness_sigma_std_0, label="Layer 1",fmt='.-', color=color[1], ecolor='lightskyblue', elinewidth=2, capsize=2)
    # p2=par1.errorbar(x_values, flatness_sigma_1, yerr=flatness_sigma_std_1, label="Layer 2",fmt='.--', color=color[2], ecolor='palegreen', elinewidth=2, capsize=2)
    # p3=par2.errorbar(x_values, flatness_sigma_2, yerr=flatness_sigma_std_2, label="Layer 3",fmt='.:', color=color[3], ecolor='#F4F27F', elinewidth=2, capsize=2)
    # p4=par3.errorbar(x_values, generalization_errors, yerr=generalization_errors_std, label="Generalization gap",fmt=':', linewidth=3, color=color[0], ecolor='mediumpurple',elinewidth=2, capsize=2)

    # plt.ylim([0, 0.1])
    # par1.set_ylim(0, 0.1)
    # par2.set_ylim(0, 0.05)
    # par3.set_ylim(0, 25)

    # For weight
    # par1.set_ylim(0, 300)
    # par2.set_ylim(0, 100)

    plt.legend( frameon=True, ncol=3, fancybox=True, shadow=True )

    host.axis["left"].label.set_color( color[1] )
    par1.axis["right"].label.set_color( color[2] )
    par2.axis["right"].label.set_color( color[3] )
    par3.axis["right"].label.set_color( color[0] )

    plt.grid( False )

    mpl.rcParams['pgf.rcfonts'] = False

    plt.draw()
    plt.show()

    plt.savefig( results_folder + '/plot', format='png' )
    plt.savefig( results_folder + '/plot', format='eps' )
