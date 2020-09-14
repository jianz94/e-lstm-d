import os
import keras.backend as K
from keras.layers import Dense, Flatten, Input, LSTM, CuDNNLSTM, Reshape, Dropout, Add, Permute
from keras.models import Sequential, Model
from keras.regularizers import l2
from keras.layers import Layer, TimeDistributed, Lambda
from keras.optimizers import Adam, SGD, Adadelta
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score
from utils import *

flags = tf.app.flags
FLAGS = flags.FLAGS
# flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
# flags.DEFINE_string('GPU', '-1', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
# flags.DEFINE_float('weight_decay', 0.01, 'Initial learning rate.')


class e_lstm_d():

    def __init__(self, num_nodes, historical_len, encoder_units, lstm_units, decoder_units, name=None):
       
        self.historical_len = historical_len
        self.num_nodes = num_nodes
        self.encoder_units = encoder_units
        self.stacked_lstm_units = lstm_units
        self.decoder_units = decoder_units
        self.model = None
        self.loss = build_refined_loss(FLAGS.BETA)

        if name:
            self.name = name
        else:
            self.name = 'e_lstm_d'

        self._build()
    
    def _build(self):
        self.encoder = self._build_encoder()
        self.stacked_lstm = self._build_stack_lstm()
        self.decoder = self._build_decoder()

        x = Input(shape=(self.historical_len, self.num_nodes, self.num_nodes))
        h = TimeDistributed(self.encoder)(x)
        h = Reshape((self.historical_len, -1))(h)
        h = Lambda(lambda x: K.sum(x, axis=1))(h)
        h = Reshape((self.num_nodes, -1))(h)
        h = self.stacked_lstm(h)
        y = self.decoder(h)

        self.model = Model(inputs=x, outputs=y)
     
    def _build_encoder(self):
        model = Sequential()
        for i in range(len(self.encoder_units)):
            if i == 0:
                model.add(Dense(self.encoder_units[i], input_shape=(self.historical_len, self.num_nodes, self.num_nodes),
                                activation='relu', kernel_regularizer=l2(FLAGS.weight_decay)))
            else:
                model.add(Dense(self.encoder_units[i], activation='relu', kernel_regularizer=l2(FLAGS.weight_decay)))
        return model
    
    def _build_decoder(self):
        model = Sequential()
        for i in range(len(self.decoder_units)):
            if i == len(self.decoder_units) - 1:
                if i == 0:
                    model.add(Dense(self.decoder_units[i], input_shape=(self.num_nodes, self.stacked_lstm_units[-1]), activation='sigmoid', kernel_regularizer=l2(FLAGS.weight_decay)))
                else:
                    model.add(Dense(self.decoder_units[i], activation='sigmoid', kernel_regularizer=l2(FLAGS.weight_decay)))
            else:
                if i == 0:
                    model.add(Dense(self.decoder_units[i], input_shape=(self.num_nodes, self.stacked_lstm_units[-1]), activation='relu', kernel_regularizer=l2(FLAGS.weight_decay)))
                else:
                    model.add(Dense(self.decoder_units[i], activation='relu', kernel_regularizer=l2(FLAGS.weight_decay)))
        return model

    def _build_stack_lstm(self):
        model = Sequential()
        if FLAGS.GPU == '-1':
            _lstm = LSTM
        else:
            _lstm = CuDNNLSTM
        for i in range(len(self.stacked_lstm_units)):
            if i == 0:
                model.add(_lstm(units=self.stacked_lstm_units[i], input_shape=(self.num_nodes, self.encoder_units[-1]), return_sequences=True))
            else:
                model.add(_lstm(units=self.stacked_lstm_units[i], return_sequences=True))
        return model
    
    def train(self, x, y):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
        session = tf.Session(config=config)
        K.set_session(session)
        self.model.compile(optimizer=Adam(lr=FLAGS.learning_rate), loss=self.loss)
        history = self.model.fit(x, y, batch_size=FLAGS.batch_size, epochs=FLAGS.num_epochs, verbose=1)
        return history

    def evaluate(self, x, y):
        y_preds = self.model.predict(x, batch_size=32)
        template = np.ones((self.num_nodes, self.num_nodes)) - np.identity(self.num_nodes)
        aucs, err_rates = [], []
        for i in range(y_preds.shape[0]):
            y_pred = y_preds[i] * template
            aucs.append(get_auc(np.reshape(y_pred, (-1, )), np.reshape(y[i], (-1, ))))
            err_rates.append(get_err_rate(y_pred, y[i]))
        return aucs, err_rates

    def predict(self, x):
        return self.model.predict(x, batch_size=32)

    def save_weights(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save_weights(path+self.name+'.h5')

    def load_weights(self, weightFile):
        if not os.path.exists(weightFile):
            raise FileNotFoundError
        else:
            self.model.load_weights(weightFile)
    

if __name__ == "__main__":
    model = e_lstm_d(num_nodes=274, historical_len=10, encoder_units=[128], lstm_units=[256, 256], decoder_units=[274])
    print(model.model.summary())
