from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, Dropout,
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)

def simple_rnn_model(input_dim, output_dim=29, show_summary=True):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    if show_summary:
            print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29, show_summary=True):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # COMPLETED: Add batch normalization 
    bn_rnn = BatchNormalization()(simp_rnn)
    # COMPLETED: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    if show_summary:
        print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29,show_summary=True):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # COMPLETED: Add batch normalization
    bn_rnn = BatchNormalization()(simp_rnn)
    # COMPLETED: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    if show_summary:
        print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29,show_summary=True):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    first_layer = True
    # COMPLETED: Add recurrent layers, each with batch normalization
    for layer in range(recur_layers):
        if first_layer:
            rnn = GRU(output_dim, return_sequences=True, implementation=2)(input_data)
            first_layer = False
        else:
            rnn = GRU(output_dim, return_sequences=True, implementation=2)(bn)
            
        bn = BatchNormalization()(rnn)
        
    # COMPLETED: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    if show_summary:
        print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29,show_summary=True):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # COMPLETED: Add bidirectional recurrent layer
    gru = Bidirectional(GRU(128, activation='relu',return_sequences=True))(input_data)
    # COMPLETED: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim,activation='softmax'))(gru)  
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    if show_summary:
        print(model.summary())
    return model


def model5(spectrogram=True):
    """ Build a deep network for speech 
    """
    """ Build a deep network for speech 
    """
    if spectrogram:
        input_dim = 161
    else:  # MFCC
        input_dim = 13
    conv_layers= 3
    bidirectional_layers = 4
    time_distributed_layers = 4
    output_dim = 29
    first_layer = True
    
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    # COMPLETED: Specify the layers in your network
    for layer in range(conv_layers):
        if first_layer:
            x = Conv1D(filters=350,kernel_size=11, strides=1, padding='same', activation='relu')(input_data)
            first_layer = False
        else:
            x = Conv1D(350,11 , strides=1, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = Dense(350,activation='sigmoid')(x)


    for layer in range(bidirectional_layers):
        x = Bidirectional(LSTM(200,return_sequences=True))(x)
        x = BatchNormalization()(x)

    for layer in range(time_distributed_layers-1):  
        x = TimeDistributed(Dense(100*2**(time_distributed_layers-layer-2)))(x)
        x = Dropout(0.2)(x)
        x = Dense(350,activation='softmax')(x)
      

    x = TimeDistributed(Dense(output_dim))(x)
    # COMPLETED: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(x)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # COMPLETED: Specify model.output_length
    model.output_length = lambda x: x
    (model.summary())
    return model

def final_model(spectrogram=True):
    """ Build a deep network for speech 
    """
    if spectrogram:
        input_dim = 161
    else:  # MFCC
        input_dim = 13
    
    bidirectional_layers = 3
    output_dim = 29
    
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    # COMPLETED: Specify the layers in your network
    x = Conv1D(filters=100,kernel_size=11, strides=1, padding='same', activation='relu')(input_data)
    x = BatchNormalization()(x)
    
    for layer in range(bidirectional_layers):
        x = Bidirectional(GRU(100, activation='relu',return_sequences=True))(x)
        x = BatchNormalization()(x)

    x = TimeDistributed(Dense(output_dim))(x)
    # COMPLETED: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(x)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # COMPLETED: Specify model.output_length
    model.output_length = lambda x: x
    (model.summary())
    return model