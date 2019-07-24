import tensorflow as tf
import numpy as np


class Classifier(object):

    def __init__(self):
        # init weights in random
        w1 = tf.Variable(tf.random_normal([26,128],stddev = 1,seed = 1))
        w2 = tf.Variable(tf.random_normal([128,256],stddev = 1,seed = 1))
        w3 = tf.Variable(tf.random_normal([256,256],stddev = 1,seed = 1))
        w4 = tf.Variable(tf.random_normal([256,1],stddev = 1,seed = 1))
        b1 = tf.Variable(tf.random_normal([64,1],stddev = 1,seed = 1))
        b2 = tf.Variable(tf.random_normal([64,1],stddev = 1,seed = 1))
        b3 = tf.Variable(tf.random_normal([64,1],stddev = 1,seed = 1))
        b4 = tf.Variable(tf.random_normal([64,1],stddev = 1,seed = 1))
        sess = tf.Session()
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        learning_rate = 0.001

    def train(self, dg_batch, sf_batch):

        #inds = np.arange(2048)
        #np.random.shuffle(inds)
        # input
        x = tf.placeholder(tf.float32,shape=(64,26),name="input")
        a = tf.matmul(x,w1) + b1
        b = tf.matmul(a,w2) + b2
        c = tf.matmul(b,w3) + b3
        y = tf.matmul(c,w4) + b4
        n_dg = len(dg_batch)
        n_sf = len(sf_batch)
        #for _ in range(4):
            #np.random.shuffle(inds)
            for start in range (0, n_dg, 64):
                end = start + 64
                if (end > n_dg):
                    break
                mbinds = dg_batch[start:end]
                #slices = (arr[mbinds] for arr in dg_batch)
                #
                y = tf.sigmoid(y)
                y_ = 1
                cross_entropy = -tf.reduce_mean(
                    y_ * tf.log(tf.clip_by_value(y,1e-10,1.0))
                    +(1-y_)*tf.log(tf.clip_by_value(1-y,1e-10,1.0)))
                train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

                sess.run(train_step,feed_dict={x:mbinds})
            
            for start in range (0, n_sf, 64):
                end = start + 64
                if (end > n_dg):
                    break
                mbinds = sf_batch[start:end]
                #slices = (arr[mbinds] for arr in sf_batch)
                #
                y = tf.sigmoid(y)
                y_ = 0
                cross_entropy = -tf.reduce_mean(
                    y_ * tf.log(tf.clip_by_value(y,1e-10,1.0))
                    +(1-y_)*tf.log(tf.clip_by_value(1-y,1e-10,1.0)))
                train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

                sess.run(train_step,feed_dict={x:mbinds})
            
        sess.close()
    def save(self):

        return w1, w2, w3, w4, b1, b2, b3, b4


        