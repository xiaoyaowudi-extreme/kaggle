import tensorflow as tf
import numpy as np
import os
import network

batch_size           = 100
LEARNING_RATE_BASE   = 0.2
LEARNING_RATE_DECAY  = 0.9999
REGULARAZTION_RATE   = 0.0001
TRAINING_STEPS       = 60000
MOVING_AVERAGE_DECAY = 0.99
MODEL_PATH           = os.path.abspath('./log')
MODEL_NAME           = 'mnist.ckpt'
EXAMPLE_NUM          = 70000
DATASET_PATH         = os.path.abspath('./data/tfrecords/tfrecords_mnist_train.tfrecords')
NUM_CLASSES          = 10
IMAGE_SIZE           = 28

def get_batch():
    reader = tf.TFRecordReader()

    filename_queue = tf.train.string_input_producer(
        [DATASET_PATH])

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        })

    return tf.train.batch([tf.cast(tf.reshape(tf.decode_raw(features['image_raw'], tf.uint8), [IMAGE_SIZE, IMAGE_SIZE, 1]), tf.float32) * 1 / 255, \
        tf.one_hot(tf.cast(features['label'], tf.int64), NUM_CLASSES, dtype = tf.float32)], \
        batch_size = batch_size, capacity = 1000 + 3 * batch_size)

def train():
    regularizer = tf.keras.regularizers.l2(REGULARAZTION_RATE)
    net         = network.mnist(regularizer = regularizer, training = True)
    global_step = tf.Variable(0, trainable = False)

    variable_averages    = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    learing_rate         = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, EXAMPLE_NUM / batch_size, LEARNING_RATE_DECAY)
    train_step           = tf.train.GradientDescentOptimizer(learing_rate).minimize(net.loss, global_step = global_step)
    
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name = 'train')
    
    if os.path.exists(MODEL_PATH) == False or os.path.isdir(MODEL_PATH) == False:
        os.makedirs(MODEL_PATH)

    if not os.path.exists(DATASET_PATH):
        raise Exception('no dataset found! please run tfrecords.py first')

    saver  = tf.train.Saver()
    states = tf.train.get_checkpoint_state(MODEL_PATH)
    if states != None:
        checkpoint_paths = states.all_model_checkpoint_paths
        saver.recover_last_checkpoints(checkpoint_paths)

    image_batch, label_batch = get_batch()

    summary_merged = tf.summary.merge_all()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(MODEL_PATH, graph=sess.graph)

        tf.global_variables_initializer().run()

        coord   = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(TRAINING_STEPS):
            xs, ys                 = sess.run([image_batch, label_batch])
            _, loss, step, summary = sess.run([train_op, net.loss, global_step, summary_merged], feed_dict = {net.input : xs, net.label : ys})
            if i % 1000 == 0:
                print('After %d training step(s), loss on training batch is %g.' % (step, loss))
                summary_writer.add_summary(summary, i)
                saver.save(sess, os.path.join(MODEL_PATH, MODEL_NAME), global_step = global_step)

def main(argv = None):
    train()

if __name__ == '__main__':
    tf.app.run()