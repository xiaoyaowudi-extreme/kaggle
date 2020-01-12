import tensorflow as tf
import numpy as np
import os
import sys
import cv2

if len(sys.argv) < 2 or (sys.argv[1] != 'train' and sys.argv[1] != 'test'):
    raise Exception('wrong arguments! Usage: tfrecords.py [train|test]')

tfrecords_path = os.path.abspath('./data/tfrecords')
data_path      = os.path.abspath('./data')
data_path      = os.path.join(data_path, sys.argv[1])
labels         = [ i[:-4] for i in os.listdir(os.path.join(data_path, 'image')) if i.endswith('.jpg') ]
labels         = [ i for i in labels if os.path.exists(os.path.join(data_path, 'label', i + '.txt')) ]

if os.path.exists(tfrecords_path) == False or os.path.isdir(tfrecords_path) == False:
    os.makedirs(tfrecords_path)

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def read_label(path):
    with open(path, 'r') as f:
        res = int(float(f.read()))
    return res

tfrecords_name = os.path.join(tfrecords_path, 'tfrecords_mnist_' + sys.argv[1] + '.tfrecords')
writer         = tf.python_io.TFRecordWriter(tfrecords_name)

for i in labels:
    example = tf.train.Example(features=tf.train.Features(feature = {
        'label'     : _int64_feature(read_label(os.path.join(data_path, 'label', i + '.txt'))),
        'image_raw' : _bytes_feature(cv2.imread(os.path.join(data_path, 'image', i + '.jpg')).tostring())
    }))
    writer.write(example.SerializeToString())

writer.close()