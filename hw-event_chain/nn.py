#! /usr/bin/env python
#coding=utf-8
from __future__ import division

import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    Dense,
    TimeDistributed,
    Dropout,
    Flatten,
    Activation,
    RepeatVector,
    Permute,
    AveragePooling1D,
    MaxPooling1D,
    GRU,
    Conv1D,
    Multiply,
    Concatenate,
    Add,
)
from tensorflow.keras.models import Model, Sequential

# TODO start：设置合适的参数

EMBED_SIZE = 128      # 词向量维度
HIDDEN_SIZE = 128     # LSTM隐藏层大小
NUM_LAYERS = 2        # LSTM层数（>1 时自动堆叠，前层 return_sequences=True）
DROPOUT = 0.2         # LSTM层输出 dropout
MAX_LEN = 100         # 序列最大长度
BATCH_SIZE = 32       # 批大小
EPOCHS = 10           # 训练轮数
RANDOM_SEED = 42      # 随机种子  传统好吧

# TODO end

nb_filter=250
filter_length=3

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

def get_embedding_input_output(part_name,vocab_size):
    main_input = Input(shape=(MAX_LEN,), dtype='int32', name=part_name+'_input')
        
    x = Embedding(output_dim=EMBED_SIZE, input_dim=vocab_size, input_length=MAX_LEN)(main_input)
        
    return main_input,x


def lstm_train(X_train,y_train,vocab_size):
    
    X_train = sequence.pad_sequences(X_train, maxlen=MAX_LEN)
    y_train = np.asarray(y_train, dtype='float32')
           
    main_input = Input(shape=(MAX_LEN,), dtype='int32')
           
    x = Embedding(output_dim=EMBED_SIZE, input_dim=vocab_size, input_length=MAX_LEN)(main_input)
    
    # 堆叠多层 LSTM，前层需 return_sequences=True
    h = x
    for i in range(max(NUM_LAYERS - 1, 0)):
        h = LSTM(HIDDEN_SIZE, return_sequences=True, dropout=DROPOUT)(h)
    lstm_out = LSTM(HIDDEN_SIZE, dropout=DROPOUT)(h)
    
    main_loss = Dense(1, activation='sigmoid', name='main_output')(lstm_out)
    
    model = Model(inputs=main_input, outputs=main_loss)
    
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
        
    return model


def cnn_train(X_train,y_train,vocab_size):
    
    X_train = sequence.pad_sequences(X_train, maxlen=MAX_LEN)
    y_train = np.asarray(y_train, dtype='float32')
           
    print('Build model...')
    model = Sequential()
    model.add(Embedding(vocab_size, EMBED_SIZE, input_length=MAX_LEN))
    
    model.add(Dropout(0.25))
    
    # we add a 1D convolution, which will learn nb_filter
    # word group filters of size filter_length:
    model.add(Conv1D(filters=nb_filter,
                     kernel_size=filter_length,
                     padding='valid',
                     activation='relu',
                     strides=1))
    # we use standard max pooling (halving the output of the previous layer):
    model.add(MaxPooling1D(pool_size=2))
    
    # We flatten the output of the conv layer,
    # so that we can add a vanilla dense layer:
    model.add(Flatten())
    
    # We add a vanilla hidden layer:
    model.add(Dense(HIDDEN_SIZE))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    
    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
        
    return model

def cnn_combine_train(X_train_list,y_train,vocab_size):
    N=len(X_train_list)
    
    X_train_list = [sequence.pad_sequences(x_train, maxlen=MAX_LEN) for x_train in X_train_list]
    y_train = np.asarray(y_train, dtype='float32')
    
    input_list=[]
    out_list=[]
    for i in range(N):
        input,out=get_embedding_input_output('f%d' %i,vocab_size)
        input_list.append(input)
        out_list.append(out)
            
    x = Concatenate()(out_list)
    
    x = Dropout(0.25)(x)
        
    # we add a Convolution1D, which will learn nb_filter
    # word group filters of size filter_length:
    x = Conv1D(filters=nb_filter,
               kernel_size=filter_length,
               padding='valid',
               activation='relu',
               strides=1)(x)
                            
    # we use standard max pooling (halving the output of the previous layer):
    x = MaxPooling1D(pool_size=2)(x)
    
    # We flatten the output of the conv layer,
    # so that we can add a vanilla dense layer:
    x = Flatten()(x)
    
    # We add a vanilla hidden layer:
    x = Dense(HIDDEN_SIZE)(x)
    x = Dropout(0.25)(x)
    x = Activation('relu')(x)
    
    # We project onto a single unit output layer, and squash it with a sigmoid:
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    
    model = Model(inputs=input_list, outputs=x)
    
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    model.fit(X_train_list, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
    
    return model

def lstm_combine_train(X_train_list,y_train,vocab_size):
    N=len(X_train_list)
    
    X_train_list = [sequence.pad_sequences(x_train, maxlen=MAX_LEN) for x_train in X_train_list]
    y_train = np.asarray(y_train, dtype='float32')
    
    input_list=[]
    out_list=[]
    for i in range(N):
        input,out=get_embedding_input_output('f%d' %i,vocab_size)
        input_list.append(input)
        out_list.append(out)
            
    x = Concatenate()(out_list)
    
    x = LSTM(HIDDEN_SIZE)(x)
    
    # We project onto a single unit output layer, and squash it with a sigmoid:
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    
    model = Model(inputs=input_list, outputs=x)
    
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    model.fit(X_train_list, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
    
    return model

def lstm_attention_combine_train(X_train_list,y_train,vocab_size):
    N=len(X_train_list)
    
    X_train_list = [sequence.pad_sequences(x_train, maxlen=MAX_LEN) for x_train in X_train_list]
    y_train = np.asarray(y_train, dtype='float32')
    
    input_list=[]
    out_list=[]
    for i in range(N):
        input,out=get_embedding_input_output('f%d' %i,vocab_size)
        input_list.append(input)
        out_list.append(out)
            
    x = Concatenate()(out_list)
    
    lstm_out = LSTM(HIDDEN_SIZE, return_sequences=True)(x)
    
    x = lstm_out
    for i in range(10):
        att = TimeDistributed(Dense(1))(x)
        att = Flatten()(att)
        att = Activation(activation="softmax")(att)
        att = RepeatVector(HIDDEN_SIZE)(att)
        att = Permute((2,1))(att)
        x = att

    mer = Multiply()([att, lstm_out])
    mer = Multiply()([mer, out_list[-1]])
    hid = AveragePooling1D(pool_size=2)(mer)
    hid = Flatten()(hid)
    
    #hid = merge([hid,out_list[-1]], mode='concat')
        
    main_loss = Dense(1, activation='sigmoid', name='main_output')(hid)
    
    model = Model(inputs=input_list, outputs=main_loss)
    
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    model.fit(X_train_list, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
    
    return model


def lstm_memory_train(X_train_list,y_train,vocab_size):
    N=len(X_train_list)
    
    X_train_list = [sequence.pad_sequences(x_train, maxlen=MAX_LEN) for x_train in X_train_list]
    y_train = np.asarray(y_train, dtype='float32')
    
    input_list=[]
    out_list=[]
    for i in range(N):
        input,out=get_embedding_input_output('f%d' %i,vocab_size)
        input_list.append(input)
        out_list.append(out)
            
    x = Concatenate()(out_list)
    
    lstm_out = LSTM(HIDDEN_SIZE, return_sequences=True)(x)
    
    lstm_share=GRU(HIDDEN_SIZE, return_sequences=True)
    
    x = lstm_out
    for i in range(2):
        att = TimeDistributed(Dense(1))(x)
        att = Flatten()(att)
        att = Activation(activation="softmax")(att)
        att = RepeatVector(HIDDEN_SIZE)(att)
        att = Permute((2,1))(att)
        
        mer = Multiply()([att, lstm_out])
        mer = Multiply()([mer, out_list[-1]])
        
        z = Add()([lstm_out,mer])
        z = lstm_share(z)
        x = z

    hid = AveragePooling1D(pool_size=2)(x)
    hid = Flatten()(hid)
    
    #hid = merge([hid,out_list[-1]], mode='concat')
        
    main_loss = Dense(1, activation='sigmoid', name='main_output')(hid)
    
    model = Model(inputs=input_list, outputs=main_loss)
    
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    model.fit(X_train_list, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
    
    return model
