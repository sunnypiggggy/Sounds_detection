import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_feature_utility import *




if __name__ =='__main__':


    config=tf.ConfigProto(device_count={'GPU':0},log_device_placement=True)

    with tf.Session(config=config) as sess:
        hello=tf.constant('shit happens')
        print(sess.run(hello))

        data_set=load_features('features')

        reader=tf.WholeFileReader()







