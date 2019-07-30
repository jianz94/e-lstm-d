from model import e_lstm_d
import os
import tensorflow as tf
import numpy as np
from utils import *


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('GPU', '0', 'train model on which GPU devide. -1 for CPU')
flags.DEFINE_string('dataset', 'contact', 'the dataset used for training and testing')
flags.DEFINE_integer('historical_len', 10, 'number of historial snapshots each sample')
flags.DEFINE_string('encoder', None, 'encoder structure parameters')
flags.DEFINE_string('lstm', None, 'stacked lstm structure parameters')
flags.DEFINE_string('decoder', None, 'decoder structure parameters')
flags.DEFINE_integer('num_epochs', 800, 'Number of training epochs.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for regularization item')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate. ')
flags.DEFINE_float('BETA', 2., 'Beta.')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.GPU


data = load_data('data/{}.npy'.format(FLAGS.dataset))
model = e_lstm_d(num_nodes=274, historical_len=FLAGS.historical_len, encoder_units=[int(x) for x in FLAGS.encoder[1:-1].split(',')], 
                 lstm_units=[int(x) for x in FLAGS.lstm[1:-1].split(',')], 
                 decoder_units=[int(x) for x in FLAGS.decoder[1:-1].split(',')],
                 name=FLAGS.dataset)

trainX = np.array([data[k: FLAGS.historical_len+k] for k in range(240)], dtype=np.float32)
trainY = np.array(data[FLAGS.historical_len: 240+FLAGS.historical_len], dtype=np.float32)
testX = np.array([data[240+k:240+FLAGS.historical_len+k] for k in range(80)], dtype=np.float32)
testY = np.array(data[240+FLAGS.historical_len:320+FLAGS.historical_len], dtype=np.float32)

history = model.train(trainX, trainY)
loss = history.history['loss']
# np.save('loss.npy', np.array(loss))
aucs, err_rates = model.evaluate(testX, testY)
# model.save_weights('tmp/')
print(np.average(aucs), np.average(err_rates))







