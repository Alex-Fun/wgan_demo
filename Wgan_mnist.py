import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)

logging.debug("begin: {0}".format(time.time()))

seed = 72
weights_clip_threshold = 0.1
wgan_gp = False
batch_size = 64
output_dir = '/output'
# 定义Generator 从小到大
#   与分类网络结构相反，conv换为transpose_conv
def generator(z):
    with tf.variable_scope('generator'):
        gen = tf.layers.dense(inputs=z, activation=None, units=4 * 4 * 512)
        gen = tf.reshape(gen, [-1, 4, 4, 512])
        gen = tf.layers.batch_normalization(gen)
        gen = tf.nn.relu(gen)
        gen = tf.layers.conv2d_transpose(gen, filters=256, padding='SAME',
                                         kernel_initializer=tf.random_normal_initializer(0, 0.02),
                                         activation=None,
                                         kernel_size=3, strides=2)
        gen = tf.layers.batch_normalization(gen)
        gen = tf.nn.relu(gen)
        gen = tf.layers.conv2d_transpose(gen, filters=128, padding='SAME',
                                         kernel_initializer=tf.random_normal_initializer(0, 0.02),
                                         activation=None,
                                         kernel_size=3, strides=2)
        gen = tf.layers.batch_normalization(gen)
        gen = tf.nn.relu(gen)
        gen = tf.layers.conv2d_transpose(gen, filters=64, padding='SAME',
                                         kernel_initializer=tf.random_normal_initializer(0, 0.02),
                                         activation=None,
                                         kernel_size=3, strides=2)
        gen = tf.layers.batch_normalization(gen)
        gen = tf.nn.relu(gen)
        gen = tf.layers.conv2d_transpose(gen, filters=1, padding='SAME',
                                         kernel_initializer=tf.random_normal_initializer(0, 0.02),
                                         activation=tf.nn.tanh,
                                         kernel_size=3, strides=1)
        # kernel = tf.random_normal([4, 4, 512, 1], mean=0, stddev=0.02, seed=seed)
        # gen = tf.nn.conv2d_transpose(z, kernel, output_shape=[1, 4, 4, 512], strides=[1, 2, 2, 1], padding='SAME')
        # gen = tf.layers.batch_normalization(gen)
        # gen = tf.nn.relu(gen)
        #
        # kernel = tf.random_normal([3, 3, 256, 512], mean=0, stddev=0.02, seed=seed)
        # gen = tf.nn.conv2d_transpose(gen, kernel, output_shape=[1, 6, 6, 256], strides=[1, 2, 2, 1], padding='SAME')
        # gen = tf.layers.batch_normalization(gen)
        # gen = tf.nn.relu(gen)
        #
        # kernel = tf.random_normal([3, 3, 128, 256], mean=0, stddev=0.02, seed=seed)
        # gen = tf.nn.conv2d_transpose(gen, kernel, output_shape=[1, 12, 12, 128], strides=[1, 2, 2, 1], padding='SAME')
        # gen = tf.layers.batch_normalization(gen)
        # gen = tf.nn.relu(gen)
        #
        # kernel = tf.random_normal([3, 3, 64, 128], mean=0, stddev=0.02, seed=seed)
        # gen = tf.nn.conv2d_transpose(gen, kernel, output_shape=[1, 24, 24, 64], strides=[1, 2, 2, 1], padding='SAME')
        # gen = tf.layers.batch_normalization(gen)
        # gen = tf.nn.relu(gen)
        #
        # kernel = tf.random_normal([3, 3, 1, 64], mean=0, stddev=0.02, seed=seed)
        # gen = tf.nn.conv2d_transpose(gen, kernel, output_shape=[1, 32, 32, 1], strides=[1, 2, 2, 1], padding='SAME')

    return gen

# 定义Discriminator
#   分类网络结构
def discriminator(x, reuse = False):
    with tf.variable_scope('discriminator', reuse=reuse):
        kernel = tf.random_normal([3, 3, 1, 64], seed=seed)
        dis = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')
        dis = tf.nn.avg_pool(dis, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        dis = tf.layers.batch_normalization(dis)
        dis = tf.nn.relu(dis)

        kernel = tf.random_normal([3, 3, 64, 128], seed=seed)
        dis = tf.nn.conv2d(dis, kernel, strides=[1, 1, 1, 1], padding='SAME')
        dis = tf.nn.avg_pool(dis, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        dis = tf.layers.batch_normalization(dis)
        dis = tf.nn.relu(dis)

        kernel = tf.random_normal([3, 3, 128, 256], seed=seed)
        dis = tf.nn.conv2d(dis, kernel, strides=[1, 1, 1, 1], padding='SAME')
        dis = tf.nn.avg_pool(dis, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        dis = tf.layers.batch_normalization(dis)
        dis = tf.nn.relu(dis)

        kernel = tf.random_normal([3, 3, 256, 64], seed=seed)
        dis = tf.nn.conv2d(dis, kernel, strides=[1, 1, 1, 1], padding='SAME')
        dis = tf.nn.avg_pool(dis, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        dis = tf.layers.batch_normalization(dis)
        dis = tf.nn.relu(dis)

        kernel = tf.random_normal([3, 3, 64, 1], seed=seed)
        dis = tf.nn.conv2d(dis, kernel, strides=[1, 1, 1, 1], padding='SAME')
        dis = tf.nn.avg_pool(dis, ksize=[1, dis.shape[1], dis.shape[2], 1], strides=[1, 1, 1, 1], padding='VALID')
        dis = tf.layers.batch_normalization(dis)
        dis = tf.nn.relu(dis)

        kernel = tf.random_normal([1, 1, 1, 1], seed=seed)
        pre_logit = tf.nn.conv2d(dis, kernel, strides=[1, 1, 1, 1], padding='SAME')
        logits = tf.squeeze(pre_logit)

    return logits

gen_z_dim=128
real_img = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 1], name='real_img')
# 输入为随机值
z = tf.random_normal([tf.shape(real_img)[0], gen_z_dim], seed=seed)

# 定义G的loss和优化器
#   优化器中指定可变参数为G的权重
fake_img = generator(z)
# 定义D的loss和优化器
#   指定优化器优化参数范围为D的权重
#   训练时限制D权重的范围
fake_result = discriminator(fake_img)
real_result = discriminator(real_img, reuse=True)
loss_g_d = tf.log(real_result) + tf.log(1-fake_result)
loss_g = tf.log(1-fake_result)

learning_rate_g = tf.placeholder(dtype=tf.float32)
learning_rate_d = tf.placeholder(dtype=tf.float32)

global_step_d = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32)
global_step_g = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32)

var_list_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
var_list_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")


def clip_weights(weights):
    return tf.clip_by_value(weights, -weights_clip_threshold, weights_clip_threshold)

# train_d = tf.train.AdamOptimizer(learning_rate=learning_rate_d).minimize(loss_g_d, global_step=global_step_d, var_list=var_list_d)
train_d = tf.train.RMSPropOptimizer(learning_rate=learning_rate_d).minimize(loss_g_d, global_step=global_step_d, var_list=var_list_d)
train_g = tf.train.RMSPropOptimizer(learning_rate=learning_rate_g).minimize(loss_g, global_step=global_step_g, var_list=var_list_g)
# optimizer_d = tf.train.AdamOptimizer(learning_rate=learning_rate_d)
# grads_and_vars_d = optimizer_d.compute_gradients(loss=loss_g_d, var_list=var_list_d)
# train_d = optimizer_d.apply_gradients(grads_and_vars_d, global_step=global_step_d)

clips = [tf.assign(var_item, clip_weights(var_item)) for var_item in var_list_d]
with tf.control_dependencies([train_d]):
    # 先执行train_d，然后执行下面内容，即是将变量clip
    clip_disc_weights = tf.group(*clips)

init_op = tf.global_variables_initializer()
# D输入数据集为minist数据集，图片尺寸为28*28
    # 读入数据
dataset = input_data.read_data_sets("../data/mnist/input_data/", one_hot=True)

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
sess.run(init_op)
plt.ion()

def get_batch():
    real_img_val, _ = dataset.train.next_batch(batch_size)
    real_img_val = real_img_val * 2.0 - 1.0
    real_img_val = np.reshape(real_img_val, [-1, 28, 28])
    zero_padding =((0,0),(2,2),(2,2))
    real_img_val = np.pad(real_img_val, pad_width=zero_padding, mode='constant', constant_values=-1)
    real_img_val = np.expand_dims(real_img_val, axis=-1)
    return real_img_val

step = 0
max_step = 20000
learning_rate_global = 5e-5

# 训练
#   每10次打印一次两个loss
#   最后一次（或每100次）输出生成网络伪造的数据图片
while step <= max_step:
    d_train_step = np.max((100 - step*3, 5))
    if step < 25 or step % 500 == 0:
        d_train_step = 100
    for d_step in range(d_train_step):
        real_img_val = get_batch()
        _, loss_val_d = sess.run([clip_disc_weights, loss_g_d], feed_dict={real_img: real_img_val,
                                                   learning_rate_d: learning_rate_global,
                                                   learning_rate_g: learning_rate_global
                                                   })
    real_img_val = get_batch()
    _, loss_val_g = sess.run([train_g, loss_g], feed_dict={real_img:real_img_val,
                                           learning_rate_g:learning_rate_global,
                                           learning_rate_d:learning_rate_global
                                           })
    if step % 10 == 0:
        logging.debug('step {0}, loss_d:{1}， loss_g={2}'.format(step, loss_val_d, loss_val_g))

    if step % 500 == 0:
        z_val, fake_val, real_val, fake_result_val = sess.run([z, fake_img, real_img, fake_result], feed_dict={real_img: real_img_val})
        logging.debug('fake_result_val {0}'.format(fake_result_val))
        # plt.imshow(fake_val[0].squeeze())
        cv2.imwrite(os.path.join(output_dir, 'g_img/g_{0}.jpg'.format(step)), fake_val[0].squeeze())
    step += 1

saver = tf.train.Saver()
checkpoint_path = os.path.join(output_dir, 'wgan_checkpoint')
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
saver.save(sess, os.path.join(checkpoint_path, "model.ckpt"), global_step=step)