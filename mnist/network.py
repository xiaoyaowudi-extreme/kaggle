import tensorflow as tf

INPUT_NODE   = 784
OUTPUT_NODE  = 10

IMAGE_SIZE   = 28
NUM_CHANNELS = 1
NUM_LABELS   = 10

CONV1_DEEP   = 32
CONV1_SIZE   = 5

CONV2_DEEP   = 64
CONV2_SIZE   = 5

FC_SIZE      = 512

class mnist(object):
    def pooling(name, ksize, stride, input):
        with tf.variable_scope(name):
            pool = tf.nn.max_pool(input, ksize = [1, ksize, ksize, 1], strides = [1, stride, stride, 1], padding = 'SAME')
        return pool
    def conv2d(name, kernel, input, out_depth, stride = 1):
        in_depth  = input.shape[-1]
        with tf.variable_scope(name):
            conv_weights = tf.get_variable(
                'weights', [kernel, kernel, in_depth, out_depth],
                initializer = tf.truncated_normal_initializer(stddev = 0.1))
            conv_biases  = tf.get_variable(
                'bias', [out_depth],
                initializer = tf.constant_initializer(0.0))
            
            conv         = tf.nn.conv2d(input, conv_weights, strides = [1, stride, stride, 1], padding = 'SAME')
            
            relu = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
        return relu
    def fc(name, input, out_length, regularizer = None, losses_collection = 'losses', dropout = False, training = False, prob = 0.5, constant = False, relu = True):
        with tf.variable_scope(name):
            fc_weights = tf.get_variable(
                'weights', [input.shape[-1], out_length],
                initializer = tf.truncated_normal_initializer(stddev = 0.1))
            if regularizer != None:
                tf.add_to_collection(losses_collection, regularizer(fc_weights))
            if constant:
                fc_biases = tf.get_variable('bias', [out_length], initializer = tf.constant_initializer(0.1))
            else:
                fc_biases = tf.get_variable('bias', [out_length], initializer = tf.truncated_normal_initializer(stddev = 0.1))
            if relu:
                fc    = tf.nn.relu(tf.matmul(input, fc_weights) + fc_biases)
            else:
                fc    = tf.matmul(input, fc_weights) + fc_biases
            if dropout:
                fc = tf.layers.dropout(fc, prob, training = training)
        return fc
    def build_network(self, regularizer, training, prob):
        self.conv1 = mnist.conv2d('conv1', CONV1_SIZE, self.input, CONV1_DEEP)
        self.pool1 = mnist.pooling('pool1', 2, 2, self.conv1)
        self.conv2 = mnist.conv2d('conv2', CONV2_SIZE, self.pool1, CONV2_DEEP)
        self.pool2 = mnist.pooling('pool2', 2, 2, self.conv2)
        self.fc_in = tf.layers.flatten(self.pool2)
        self.fc1   = mnist.fc('fc1', self.fc_in, FC_SIZE, regularizer = regularizer, dropout = True, training = training, prob = prob)
        self.out   = mnist.fc('fc2', self.fc1, OUTPUT_NODE, regularizer = regularizer, dropout = False, constant = True, relu = False)
    def __init__(self, regularizer = None, training = False, prob = 0.5):
        with tf.variable_scope('mnist'):
            self.input  = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 1], name = 'mnist_input')
            self.label  = tf.placeholder(tf.float32, [None, OUTPUT_NODE])
            self.build_network(regularizer, training, prob)
            self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.argmax(self.label, 1), logits = self.out)
            self.l_mean = tf.reduce_mean(self.losses)
            self.loss   = self.l_mean + tf.add_n(tf.get_collection('losses'))
            tf.summary.scalar('loss', self.loss)
