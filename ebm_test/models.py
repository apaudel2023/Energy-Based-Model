from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from keras.activations import relu, sigmoid, softplus, linear, tanh, softmax, elu

def ConvNet(in_chan=1, out_chan=64, nh=8):
    model = Sequential()
    model.add(Conv2D(nh * 4, kernel_size=(3, 3), input_shape=(28, 28, in_chan), bias=True, padding='valid'))
    model.add(Activation(relu))
    model.add(Conv2D(nh * 8, kernel_size=(3, 3), bias=True, padding='valid'))
    model.add(Activation(relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(nh * 8, kernel_size=(3, 3), bias=True, padding='valid'))
    model.add(Activation(relu))
    model.add(Conv2D(nh * 16, kernel_size=(3, 3), bias=True, padding='valid'))
    model.add(Activation(relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(out_chan, kernel_size=(4, 4), bias=True, padding='valid'))

    return model

def FCNet(in_dim, out_dim, l_hidden=(50,), activation='sigmoid', out_activation='linear'):
    model = Sequential()
    l_neurons = tuple(l_hidden) + (out_dim,)
    activation = (activation,) * len(l_hidden) + (out_activation,)
    
    prev_dim = in_dim
    for n_hidden, act in zip(l_neurons, activation):
        model.add(Dense(n_hidden, input_dim=prev_dim))
        if act == 'relu':
            model.add(Activation(relu))
        elif act == 'sigmoid':
            model.add(Activation(sigmoid))
        elif act == 'softplus':
            model.add(Activation(softplus))
        elif act == 'linear':
            pass  # No activation needed for linear
        elif act == 'tanh':
            model.add(Activation(tanh))
        elif act == 'leakyrelu':
            model.add(Activation(elu))  # LeakyReLU is not directly available in Keras, elu is a close alternative
        elif act == 'softmax':
            model.add(Activation(softmax))
        else:
            raise ValueError(f'Unexpected activation: {act}')
        
        prev_dim = n_hidden

    return model

# Example usage:
# conv_model = conv_net()
# fc_model = fc_net(784, 10, l_hidden=(128, 64), activation='relu', out_activation='softmax')
