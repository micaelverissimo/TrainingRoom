from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import *
import keras.callbacks as callbacks


class Compact_Keras(object):
    '''
    version -- v0.0
    Compact Keras is a class that you can create Keras models faster and more easy with less code lines.
    
    Arguments:
    
    input_size -- the size of data (features) that will be use for contruct the neural network model.
    
    Keyword arguments:
    
    hidden_layers -- number of hidden layer to be used in your model (default is 1).
    numbers_of_neurons -- numbers of neurons to be used to construct the model (default is 5).
    epoch_numbers -- number of training epochs (default is 50).
    batch_size -- number of events to be use in echa epoch (mini-batch) (default is 5).
    callbacks -- callbacks to use in traning see https://keras.io/callbacks/ (default is None).
    number_neurons_output_layer -- number of neurons to be use in output layer
    loss_function -- function to be minimized in training (default is Binary Crossentropy).
    optimizer -- optimizer to be used in traning (default is Adam).
    output_layer_activation -- the function to be use in the output layer (default is hiperbolic tangent[tanh])
    
    
    *~ it's possible use other Keras optimezers and Keras loss_functions more information on:
        https://keras.io/optimizers/
        https://keras.io/losses/  ~*

    
                        Compact Keras - Compact for those tho see giant for those who uses:
    *=============================================================================================================*
    ||                                   Classification Exemple:                                                 ||
    || Create a sintetic input and sintetic target                                                               ||
    || >>> A = np.vstack((np.ones((100,2)),-1*np.ones((100,2)))                                                  ||
    || >>> sintetic_target = np.hstack((np.ones((100)),-1*np.ones((100)))                                        ||
    || >>> sparse_trgt = np_utils.to_categorical(sintetic_target) # make targets maximally sparse                ||     
    || Create the Compact Keras objet                                                                            ||
    || >>> compact_keras = Compact_Keras(A.shape[1])                                                             ||
    || >>> train_model = compact_keras.train(A[:150,:],sparse_trgt[:150,:],                                      ||
    ||                                       validation_data=(A[150:,:],sparse_trgt[150:,:]))                    ||
    ||                                                                                                           ||
    ||                                                                                                           ||
    *=============================================================================================================*
    
    '''
    
    def __init__(self, input_size, n_hidden_layers=1, number_of_neurons=5, number_neurons_output_layer=2,
                 epoch_numbers=100, batch_size=5, callbacks=None,
                 loss_function='binary_crossentropy',optimizer='adam',output_layer_activation='tanh'):
        
        
        self.n_hidden_layers = n_hidden_layers
        self.number_of_neurons = number_of_neurons
        self.epoch_numbers = epoch_numbers
        self.batch_size = batch_size
        self.callbacks = callbacks
        self.number_neurons_output_layer = number_neurons_output_layer
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.output_layer_activation = output_layer_activation
        
        # the constructor create and compile a keras model
        model = Sequential()
        model.add(Dense(input_size,
                        kernel_initializer='identity',
                        trainable=False,
                        input_dim=input_size))
        model.add(Activation('linear'))
        
        for ilayer in range(self.n_hidden_layers):
            model.add(Dense(self.number_of_neurons, kernel_initializer='uniform',
                            input_dim=input_size))
            model.add(Activation('tanh'))
            if ilayer != 0:
                model.add(Dense(self.number_of_neurons, kernel_initializer='uniform',
                                input_dim=self.number_of_neurons))
                model.add(Activation('tanh'))
            
        model.add(Dense(self.number_neurons_output_layer,kernel_initializer='uniform'))
        model.add(Activation(self.output_layer_activation)) 
        model.compile(loss=self.loss_function, optimizer=self.optimizer,
                      metrics=['accuracy'])
        self.model = model
    
    def train(self,input_data,data_target,validation_data,verbose=0,shuffle=True):
    
        self.verbose = verbose
        self.shuffle = shuffle
        
        trn_desc = self.model.fit(input_data, data_target, 
                             epochs=self.epoch_numbers, 
                             batch_size=self.batch_size,
                             callbacks=self.callbacks, 
                             verbose=verbose,
                             validation_data=validation_data,
                             shuffle=shuffle)
        
        return trn_desc
