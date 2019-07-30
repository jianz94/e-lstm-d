#coding -*- utf-8 -*-
import tensorflow as tf
from utils import edge_wise_loss, build_reconstruction_loss
from tensorflow.contrib.keras.api.keras.layers import GRU, Dense, Reshape, Add, concatenate, Activation, BatchNormalization

flags = tf.app.flags
FLAGS = flags.FLAGS


class DDNE():

    def __init__(self, placeholders, input_shape, **kwargs):

        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        self.input_s = placeholders['input_s']
        self.input_t = placeholders['input_t']
        self.weight = placeholders['weight']
        self.input_shape = input_shape
        self.output_dim = self.input_shape[-1]
        self.output_s = []
        self.output_t = []
        self.diff = None
        self.historical_len = 2
        self.loss = 0
        self.layers = []
        self.gradients = 0
        self.placeholders = placeholders
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
        self.opt_op = None
        # self.trainable= placeholders['in_train_mode']

        self.build()


    def build(self):

        gru = GRU(units=self.input_shape[-1], return_sequences=False, name='gru')
        bn = BatchNormalization()
        decoder_1 = Dense(units=128, activation='sigmoid', name='decoder-1')
        decoder_2 = Dense(units=self.output_dim, activation='sigmoid',name='decoder-2')
        self.layers.append(gru)
        self.layers.append(decoder_1)
        self.layers.append(decoder_2)

        activations = {}
        for i in range(self.historical_len):
            s_input = self.input_s[i]
            t_input = self.input_t[i]
            if i == 0:
                # hs = bn(s_input)
                hs = gru(s_input)
                activations['s-r-{}-o'.format(i)] = hs
                hs = Reshape((1, -1))(hs)
                activations['s-r-{}'.format(i)] = hs

                # ht = bn(t_input)
                ht = gru(t_input)
                activations['t-r-{}-o'.format(i)] = ht
                ht = Reshape((1, -1))(ht)
                activations['t-r-{}'.format(i)] = ht

            else:
                # hs = bn(s_input)
                hs = gru(Add()([activations['s-r-{}-o'.format(i - 1)], s_input]))
                activations['s-r-{}-o'.format(i)] = hs
                hs = Reshape((1, -1))(hs)
                activations['s-r-{}'.format(i)] = hs

                # ht = bn(t_input)
                ht = gru(Add()([activations['t-r-{}-o'.format(i - 1)], t_input]))
                activations['s-r-{}-o'.format(i)] = ht
                ht = Reshape((1, -1))(ht)
                activations['t-r-{}'.format(i)] = ht

        for i in range(self.historical_len):
            idx = self.historical_len - 1 - i
            s_input = self.input_s[idx]
            t_input = self.input_t[idx]
            if i == 0:
                # hs = bn(s_input)
                hs = gru(s_input)
                activations['s-l-{}-o'.format(i)] = hs
                hs = Reshape((1, -1))(hs)
                activations['s-l-{}'.format(i)] = hs

                # ht = bn(t_input)
                ht = gru(t_input)
                activations['t-l-{}-o'.format(i)] = ht
                ht = Reshape((1, -1))(ht)
                activations['t-l-{}'.format(i)] = ht

            else:
                # hs = bn(s_input)
                hs = gru(Add()([activations['s-l-{}-o'.format(i - 1)], s_input]))
                activations['s-l-{}-o'.format(i)] = hs
                hs = Reshape((1, -1))(hs)
                activations['s-l-{}'.format(i)] = hs

                # ht = bn(t_input)
                ht = gru(Add()([activations['t-r-{}-o'.format(i - 1)], t_input]))
                activations['s-l-{}-o'.format(i)] = ht
                ht = Reshape((1, -1))(ht)
                activations['t-l-{}'.format(i)] = ht

        s_h_r = concatenate([activations['s-r-{}'.format(i)] for i in range(self.historical_len)], axis=1)
        s_h_l = concatenate([activations['s-l-{}'.format(i)] for i in range(self.historical_len)], axis=1)
        t_h_r = concatenate([activations['t-r-{}'.format(i)] for i in range(self.historical_len)], axis=1)
        t_h_l = concatenate([activations['t-l-{}'.format(i)] for i in range(self.historical_len)], axis=1)

        s_c = concatenate([s_h_r, s_h_l], axis=1)
        t_c = concatenate([t_h_r, t_h_l], axis=1)
        s_c = Activation('relu')(Reshape((-1,))(s_c))
        t_c = Activation('relu')(Reshape((-1,))(t_c))

        # diff = tf.square(tf.subtract(s_c, t_c))
        # tmp = diff.get_shape().as_list()[-1]
        # self.weight = RepeatVector(tmp)(self.weight)
        # self.weight = tf.reshape(self.weight, [-1, tmp])
        # self.diff = tf.multiply(diff, self.weight)

        self.output_s = decoder_2(decoder_1(s_c))
        self.output_t = decoder_2(decoder_1(t_c))

        self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.vars = {var.name: var for var in self.variables}

        self._loss()
        self._gradients()

        self.opt_op = self.optimizer.minimize(self.loss)

    def _loss(self):
        for layer in self.layers:
            for weight in layer.get_weights():
                self.loss += FLAGS.weight_decay*tf.nn.l2_loss(weight)
        reconstruction_loss = build_reconstruction_loss(10)

        self.loss += reconstruction_loss(self.placeholders['true_s'], self.output_s) + \
                     reconstruction_loss(self.placeholders['true_t'], self.output_t)
                     # 0*edge_wise_loss(self.diff)


    def _gradients(self):
        self.gradients = tf.gradients(self.loss, self.input_s)

    def predict(self, x):
        pass

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)