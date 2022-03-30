import torch
import numpy as np
from dynamics_model import FC
import sys
sys.path.append('../')

def convert(prefix, model_prefix, num_layer, input_dim, hidden_dim, output_dim):
    model = FC(num_layer=num_layer, input_dim = input_dim, hidden_dim = hidden_dim, output_dim = output_dim)
    
    PATH = '../model/' + prefix + '/'+model_prefix+'.pth'
    model.load_state_dict(torch.load(PATH))
    weights = []
    biases = []

    # for i in range(model.num_layer):
    #     for p in model.fc[i].parameters():
            
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights.append(param.detach().numpy())
        if 'bias' in name:
            biases.append(param.detach().numpy())
    writeNNet(weights, biases, '../nnet/'+prefix+'/'+model_prefix+'.nnet')

def writeNNet(weights,biases,fileName):
    '''
    Write network data to the .nnet file format. This format is used by NeuralVerification.jl

    Args:
        weights (list): Weight matrices in the network order 
        biases (list): Bias vectors in the network order
        fileName (str): File where the network will be written
    '''
    
    #Open the file we wish to write
    with open(fileName,'w') as f2:

        #####################
        # First, we write the header lines:
        # The first line written is just a line of text
        # The second line gives the four values:
        #     Number of fully connected layers in the network
        #     Number of inputs to the network
        #     Number of outputs from the network
        #     Maximum size of any hidden layer
        # The third line gives the sizes of each layer, including the input and output layers
        # The fourth line gives an outdated flag, so this can be ignored
        # The fifth line specifies the minimum values each input can take
        # The sixth line specifies the maximum values each input can take
        #     Inputs passed to the network are truncated to be between this range
        # The seventh line gives the mean value of each input and of all outputs
        # The eighth line gives the range of each input and of all outputs
        #     These two lines are used to map raw inputs to the 0 mean, unit range of the inputs and outputs
        #     used during training
        # The ninth line begins the network weights and biases
        ####################
        f2.write("// Neural Network File Format by Kyle Julian, Stanford 2016\n")

        #Extract the necessary information and write the header information
        numLayers = len(weights)
        inputSize = weights[0].shape[1]
        outputSize = len(biases[-1])
        maxLayerSize = inputSize
        
        # Find maximum size of any hidden layer
        for b in biases:
            if len(b)>maxLayerSize :
                maxLayerSize = len(b)

        # Write data to header 
        f2.write("%d,%d,%d,%d,\n" % (numLayers,inputSize,outputSize,maxLayerSize) )
        f2.write("%d," % inputSize )
        for b in biases:
            f2.write("%d," % len(b) )
        f2.write("\n")
        f2.write("0,\n") #Unused Flag

        f2.write("0\n")
        f2.write("0\n")
        f2.write("0\n")
        f2.write("0\n")

        ##################
        # Write weights and biases of neural network
        # First, the weights from the input layer to the first hidden layer are written
        # Then, the biases of the first hidden layer are written
        # The pattern is repeated by next writing the weights from the first hidden layer to the second hidden layer,
        # followed by the biases of the second hidden layer.
        ##################
        for w,b in zip(weights,biases):
            for i in range(w.shape[0]):
                for j in range(w.shape[1]):
                    f2.write("%.5e," % w[i,j]) #Five digits written. More can be used, but that requires more more space.
                f2.write("\n")
                
            for i in range(len(b)):
                f2.write("%.5e,\n" % b[i]) #Five digits written. More can be used, but that requires more more space.


if __name__ == "__main__":
    convert("unicycle-FC3-100", "epoch_900", 3, 6, 100, 4)