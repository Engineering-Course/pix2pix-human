from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

class pix2pix(object):
    def __init__(self, sess, image_size=368,
                 batch_size=1, sample_size=1, output_size=368,
                 gf_dim=32, df_dim=32, L2_lambda=0.001,
                 input_c_dim=3, output_c_dim=16, dataset_name='facades',
                 checkpoint_dir=None, sample_dir=None):
        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [256]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
            output_c_dim: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_grayscale = (input_c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size
        self.pose_size = 46
        # self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim
        self.L2_lambda = L2_lambda

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn_e1 = batch_norm(name='g_bn_e1')
        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
#------------------------network setting---------------------
        self.weights ={
            'wc1_1': tf.Variable(tf.truncated_normal([3, 3, 6, 64], stddev=0.01), name='wc1_1'),
            'wc1_2': tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.01), name='wc1_2'),
            'wc2_1': tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.01), name='wc2_1'),
            'wc2_2': tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.01), name='wc2_2'),
            'wc3_1': tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.01), name='wc3_1'),
            'wc3_2': tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.01), name='wc3_2'),
            'wc3_3': tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.01), name='wc3_3'),
            'wc3_4': tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.01), name='wc3_4'),
            'wc4_1': tf.Variable(tf.truncated_normal([3, 3, 256, 512], stddev=0.01), name='wc4_1'),
            'wc4_2': tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.01), name='wc4_2'),
            'wc4_3': tf.Variable(tf.truncated_normal([3, 3, 512, 256], stddev=0.01), name='wc4_3'),
            'wc4_4': tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.01), name='wc4_4'),
            'wc4_5': tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.01), name='wc4_5'),
            'wc4_6': tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.01), name='wc4_6'),
            'wc4_7': tf.Variable(tf.truncated_normal([3, 3, 256, 128], stddev=0.01), name='wc4_7'),
            'wc5_1': tf.Variable(tf.truncated_normal([1, 1, 128, 512], stddev=0.01), name='wc5_1'),
            'wc5_2': tf.Variable(tf.truncated_normal([1, 1, 512, 16], stddev=0.01), name='wc5_2'),
            'fc6_1': tf.Variable(tf.truncated_normal([23*23, 1024], stddev=0.01), name='fc6_1'),
            'fc6_2': tf.Variable(tf.truncated_normal([1024, 1], stddev=0.01), name='fc6_2')
        }
        self.biases = {
            'bc1_1': tf.Variable(tf.constant(0.0, shape=[64]), name='bc1_1'),
            'bc1_2': tf.Variable(tf.constant(0.0, shape=[64]), name='bc1_2'),
            'bc2_1': tf.Variable(tf.constant(0.0, shape=[128]), name='bc2_1'),
            'bc2_2': tf.Variable(tf.constant(0.0, shape=[128]), name='bc2_2'),
            'bc3_1': tf.Variable(tf.constant(0.0, shape=[256]), name='bc3_1'),
            'bc3_2': tf.Variable(tf.constant(0.0, shape=[256]), name='bc3_2'),
            'bc3_3': tf.Variable(tf.constant(0.0, shape=[256]), name='bc3_3'),
            'bc3_4': tf.Variable(tf.constant(0.0, shape=[256]), name='bc3_4'),
            'bc4_1': tf.Variable(tf.constant(0.0, shape=[512]), name='bc4_1'),
            'bc4_2': tf.Variable(tf.constant(0.0, shape=[512]), name='bc4_2'),
            'bc4_3': tf.Variable(tf.constant(0.0, shape=[256]), name='bc4_3'),
            'bc4_4': tf.Variable(tf.constant(0.0, shape=[256]), name='bc4_4'),
            'bc4_5': tf.Variable(tf.constant(0.0, shape=[256]), name='bc4_5'),
            'bc4_6': tf.Variable(tf.constant(0.0, shape=[256]), name='bc4_6'),
            'bc4_7': tf.Variable(tf.constant(0.0, shape=[128]), name='bc4_7'),
            'bc5_1': tf.Variable(tf.constant(0.0, shape=[512]), name='bc5_1'),
            'bc5_2': tf.Variable(tf.constant(0.0, shape=[16]), name='bc5_2'),
            'bc6_1': tf.Variable(tf.constant(0.0, shape=[1024]), name='bc6_1'),
            'bc6_2': tf.Variable(tf.constant(0.0, shape=[1]), name='bc6_2'),
        }
        self.build_model()

    def build_model(self):
        self.gen_data = tf.placeholder(tf.float32,
                                        [self.batch_size, self.image_size, self.image_size,
                                         self.input_c_dim * 2],
                                        name='images_and_parsing')
        self.real_data = tf.placeholder(tf.float32,
                                        [self.batch_size, self.pose_size, self.pose_size,
                                         self.input_c_dim * 2 + self.output_c_dim],
                                        name='real_A_and_B_images')
        self.point_data = tf.placeholder(tf.float32, [self.batch_size, 16], name='point_label')

        self.real_A = self.real_data[:, :, :, :self.input_c_dim * 2]
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        self.fake_B = self.generator(self.gen_data)

        self.real_AB = tf.concat(3, [self.real_A, self.real_B])
        self.fake_AB = tf.concat(3, [self.real_A, self.fake_B])

        self.D, self.D_logits = self.discriminator(self.real_AB, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.fake_AB, reuse=True)

        self.fake_B_sample = self.sampler(self.gen_data)

        self.d_sum = tf.histogram_summary("d", self.D)
        self.d__sum = tf.histogram_summary("d_", self.D_)
        self.fake_B_sum = tf.histogram_summary("fake_B", self.fake_B)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_))) \
                        + self.L2_lambda * tf.reduce_mean(tf.sqrt(tf.nn.l2_loss((self.real_B - self.fake_B))))
                        # + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B - self.fake_B))

        self.fake_P = []
        for i in xrange(16):
            heatmap = self.fake_B[:,:,:,i]
            fake_Pi = self.generator_pose(heatmap[:,:,:,np.newaxis])
            self.fake_P.append(fake_Pi)
            self.g_loss += self.L2_lambda * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_Pi, targets=self.point_data[:,i:i+1]))
        # print fake_P

        self.d_loss_real_sum = tf.scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'd_' not in var.name]

        self.saver = tf.train.Saver()

    def load_random_samples(self):
        with open('./datasets/human/list/val_rgb_id.txt', 'r') as list_file:
            lines = list_file.readlines()
        data = np.random.choice(lines, self.batch_size)
        batch_g = []
        batch_d = []
        batch_p = []
        # gen_sample, dis_sample = [load_lip_data(sample_file) for sample_file in data]
        for batch_file in data:
            g_, d_, p_ = load_lip_data(batch_file)
            batch_g.append(g_)
            batch_d.append(d_)
            batch_p.append(p_)

        sample_g = np.array(batch_g).astype(np.float32)
        sample_d = np.array(batch_d).astype(np.float32)

        return sample_g, sample_d, batch_p, data

    def sample_model(self, sample_dir, epoch, idx):
        sample_g, sample_d, batch_p, sample_files = self.load_random_samples()
        samples, samples_p, d_loss, g_loss = self.sess.run(
            [self.fake_B_sample, self.fake_P, self.d_loss, self.g_loss],
            feed_dict={self.real_data: sample_d, self.gen_data: sample_g, self.point_data: batch_p})
        save_lip_images(samples, samples_p, self.batch_size, sample_files, 'sample')

        pose_gt = sample_d[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]
        error_sum = np.linalg.norm(samples - pose_gt)
        print("l2 loss: {:.8f}.".format(error_sum / self.batch_size))
        print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

    def train(self, args):
        """Train pix2pix"""
        d_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)

        tf.initialize_all_variables().run()

        self.g_sum = tf.merge_summary([self.d__sum,
            self.fake_B_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.merge_summary([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.train.SummaryWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(args.epoch):
            with open('./datasets/human/list/train_rgb_id.txt', 'r') as list_file:
                data = list_file.readlines()
            np.random.shuffle(data)
            batch_idxs = min(len(data), args.train_size) // self.batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = data[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_g = []
                batch_d = []
                batch_p = []
                # batch_g, batch_d = [load_lip_data(batch_file) for batch_file in batch_files]
                for batch_file in batch_files:
                    g_, d_, p_ = load_lip_data(batch_file)
                    batch_g.append(g_)
                    batch_d.append(d_)
                    batch_p.append(p_)

                batch_images_g = np.array(batch_g).astype(np.float32)
                batch_images_d = np.array(batch_d).astype(np.float32)

                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={ self.real_data: batch_images_d, self.gen_data: batch_images_g, 
                                                           self.point_data: batch_p})
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={ self.real_data: batch_images_d, self.gen_data: batch_images_g, 
                                                           self.point_data: batch_p})
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={ self.real_data: batch_images_d, self.gen_data: batch_images_g, 
                                                           self.point_data: batch_p})
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.real_data: batch_images_d, self.gen_data: batch_images_g, 
                                                           self.point_data: batch_p})
                errD_real = self.d_loss_real.eval({self.real_data: batch_images_d, self.gen_data: batch_images_g, 
                                                           self.point_data: batch_p})
                errG = self.g_loss.eval({self.real_data: batch_images_d, self.gen_data: batch_images_g, 
                                                           self.point_data: batch_p})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake+errD_real, errG))

                if np.mod(counter, 3000) == 1:
                    self.sample_model(args.sample_dir, epoch, idx)

                if np.mod(counter, 3000) == 2:
                    self.save(args.checkpoint_dir, counter)

    def discriminator(self, image, y=None, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
        # h0 is (32 x 32 x self.df_dim)
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
        # h1 is (16 x 16 x self.df_dim*2)
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, d_h=1, d_w=1, name='d_h2_conv')))
        # h3 is (8 x 8 x self.df_dim*4)
        h3 = linear(tf.reshape(h2, [self.batch_size, -1]), 1, 'd_h2_lin')

        return tf.nn.sigmoid(h3), h3

    def generator(self, image, y=None):
        conv1_1 = tf.nn.conv2d(image, self.weights['wc1_1'], strides=[1, 1, 1, 1], padding='SAME')
        conv1_1 = tf.reshape(tf.nn.bias_add(conv1_1, self.biases['bc1_1']), conv1_1.get_shape())

        conv1_2 = tf.nn.conv2d(tf.nn.relu(conv1_1), self.weights['wc1_2'], strides=[1, 1, 1, 1], padding='SAME')
        conv1_2 = tf.reshape(tf.nn.bias_add(conv1_2, self.biases['bc1_2']), conv1_2.get_shape())

        e1 = self.g_bn_e1(conv1_2)
        pool1 = tf.nn.max_pool(tf.nn.relu(e1), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        conv2_1 = tf.nn.conv2d(tf.nn.relu(pool1), self.weights['wc2_1'], strides=[1, 1, 1, 1], padding='SAME')
        conv2_1 = tf.reshape(tf.nn.bias_add(conv2_1, self.biases['bc2_1']), conv2_1.get_shape())

        conv2_2 = tf.nn.conv2d(tf.nn.relu(conv2_1), self.weights['wc2_2'], strides=[1, 1, 1, 1], padding='SAME')
        conv2_2 = tf.reshape(tf.nn.bias_add(conv2_2, self.biases['bc2_2']), conv2_2.get_shape())

        e2 = self.g_bn_e2(conv2_2)
        pool2 = tf.nn.max_pool(tf.nn.relu(e2), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv3_1 = tf.nn.conv2d(tf.nn.relu(pool2), self.weights['wc3_1'], strides=[1, 1, 1, 1], padding='SAME')
        conv3_1 = tf.reshape(tf.nn.bias_add(conv3_1, self.biases['bc3_1']), conv3_1.get_shape())

        conv3_2 = tf.nn.conv2d(tf.nn.relu(conv3_1), self.weights['wc3_2'], strides=[1, 1, 1, 1], padding='SAME')
        conv3_2 = tf.reshape(tf.nn.bias_add(conv3_2, self.biases['bc3_2']), conv3_2.get_shape())

        conv3_3 = tf.nn.conv2d(tf.nn.relu(conv3_2), self.weights['wc3_3'], strides=[1, 1, 1, 1], padding='SAME')
        conv3_3 = tf.reshape(tf.nn.bias_add(conv3_3, self.biases['bc3_3']), conv3_3.get_shape())

        conv3_4 = tf.nn.conv2d(tf.nn.relu(conv3_3), self.weights['wc3_4'], strides=[1, 1, 1, 1], padding='SAME')
        conv3_4 = tf.reshape(tf.nn.bias_add(conv3_4, self.biases['bc3_4']), conv3_4.get_shape())

        e3 = self.g_bn_e3(conv3_4)
        pool3 = tf.nn.max_pool(tf.nn.relu(e3), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv4_1 = tf.nn.conv2d(tf.nn.relu(pool3), self.weights['wc4_1'], strides=[1, 1, 1, 1], padding='SAME')
        conv4_1 = tf.reshape(tf.nn.bias_add(conv4_1, self.biases['bc4_1']), conv4_1.get_shape())

        conv4_2 = tf.nn.conv2d(tf.nn.relu(conv4_1), self.weights['wc4_2'], strides=[1, 1, 1, 1], padding='SAME')
        conv4_2 = tf.reshape(tf.nn.bias_add(conv4_2, self.biases['bc4_2']), conv4_2.get_shape())

        conv4_3 = tf.nn.conv2d(tf.nn.relu(conv4_2), self.weights['wc4_3'], strides=[1, 1, 1, 1], padding='SAME')
        conv4_3 = tf.reshape(tf.nn.bias_add(conv4_3, self.biases['bc4_3']), conv4_3.get_shape())

        conv4_4 = tf.nn.conv2d(tf.nn.relu(conv4_3), self.weights['wc4_4'], strides=[1, 1, 1, 1], padding='SAME')
        conv4_4 = tf.reshape(tf.nn.bias_add(conv4_4, self.biases['bc4_4']), conv4_4.get_shape())

        conv4_5 = tf.nn.conv2d(tf.nn.relu(conv4_4), self.weights['wc4_5'], strides=[1, 1, 1, 1], padding='SAME')
        conv4_5 = tf.reshape(tf.nn.bias_add(conv4_5, self.biases['bc4_5']), conv4_5.get_shape())

        conv4_6 = tf.nn.conv2d(tf.nn.relu(conv4_5), self.weights['wc4_6'], strides=[1, 1, 1, 1], padding='SAME')
        conv4_6 = tf.reshape(tf.nn.bias_add(conv4_6, self.biases['bc4_6']), conv4_6.get_shape())

        conv4_7 = tf.nn.conv2d(tf.nn.relu(conv4_6), self.weights['wc4_7'], strides=[1, 1, 1, 1], padding='SAME')
        conv4_7 = tf.reshape(tf.nn.bias_add(conv4_7, self.biases['bc4_7']), conv4_7.get_shape())

        e4 = self.g_bn_e4(conv4_7)

        conv5_1 = tf.nn.conv2d(tf.nn.relu(e4), self.weights['wc5_1'], strides=[1, 1, 1, 1], padding='SAME')
        conv5_1 = tf.reshape(tf.nn.bias_add(conv5_1, self.biases['bc5_1']), conv5_1.get_shape())

        conv5_2 = tf.nn.conv2d(tf.nn.relu(conv5_1), self.weights['wc5_2'], strides=[1, 1, 1, 1], padding='SAME')
        conv5_2 = tf.reshape(tf.nn.bias_add(conv5_2, self.biases['bc5_2']), conv5_2.get_shape())

        return conv5_2

    def generator_pose(self, image, y=None):
        
        pool4 = tf.nn.max_pool(tf.nn.relu(image), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        fc6_1 = tf.reshape(pool4, [-1, self.weights['fc6_1'].get_shape().as_list()[0]])
        fc6_1 = tf.add(tf.matmul(fc6_1, self.weights['fc6_1']), self.biases['bc6_1'])
        fc6_1 = tf.nn.dropout(tf.nn.relu(fc6_1), 0.5)
        fc6_2 = tf.add(tf.matmul(fc6_1, self.weights['fc6_2']), self.biases['bc6_2'])

        return tf.nn.sigmoid(fc6_2)

    def sampler_pose(self, image, y=None):
        tf.get_variable_scope().reuse_variables()

        pool4 = tf.nn.max_pool(tf.nn.relu(image), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        fc6_1 = tf.reshape(pool4, [-1, self.weights['fc6_1'].get_shape().as_list()[0]])
        fc6_1 = tf.add(tf.matmul(fc6_1, self.weights['fc6_1']), self.biases['bc6_1'])
        fc6_1 = tf.nn.dropout(tf.nn.relu(fc6_1), 0.5)
        fc6_2 = tf.add(tf.matmul(fc6_1, self.weights['fc6_2']), self.biases['bc6_2'])

        return tf.nn.sigmoid(fc6_2)

    def sampler(self, image, y=None):
        tf.get_variable_scope().reuse_variables()

        conv1_1 = tf.nn.conv2d(image, self.weights['wc1_1'], strides=[1, 1, 1, 1], padding='SAME')
        conv1_1 = tf.reshape(tf.nn.bias_add(conv1_1, self.biases['bc1_1']), conv1_1.get_shape())

        conv1_2 = tf.nn.conv2d(tf.nn.relu(conv1_1), self.weights['wc1_2'], strides=[1, 1, 1, 1], padding='SAME')
        conv1_2 = tf.reshape(tf.nn.bias_add(conv1_2, self.biases['bc1_2']), conv1_2.get_shape())
        
        e1 = self.g_bn_e1(conv1_2)
        pool1 = tf.nn.max_pool(tf.nn.relu(e1), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        conv2_1 = tf.nn.conv2d(tf.nn.relu(pool1), self.weights['wc2_1'], strides=[1, 1, 1, 1], padding='SAME')
        conv2_1 = tf.reshape(tf.nn.bias_add(conv2_1, self.biases['bc2_1']), conv2_1.get_shape())

        conv2_2 = tf.nn.conv2d(tf.nn.relu(conv2_1), self.weights['wc2_2'], strides=[1, 1, 1, 1], padding='SAME')
        conv2_2 = tf.reshape(tf.nn.bias_add(conv2_2, self.biases['bc2_2']), conv2_2.get_shape())

        e2 = self.g_bn_e2(conv2_2)
        pool2 = tf.nn.max_pool(tf.nn.relu(e2), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv3_1 = tf.nn.conv2d(tf.nn.relu(pool2), self.weights['wc3_1'], strides=[1, 1, 1, 1], padding='SAME')
        conv3_1 = tf.reshape(tf.nn.bias_add(conv3_1, self.biases['bc3_1']), conv3_1.get_shape())

        conv3_2 = tf.nn.conv2d(tf.nn.relu(conv3_1), self.weights['wc3_2'], strides=[1, 1, 1, 1], padding='SAME')
        conv3_2 = tf.reshape(tf.nn.bias_add(conv3_2, self.biases['bc3_2']), conv3_2.get_shape())

        conv3_3 = tf.nn.conv2d(tf.nn.relu(conv3_2), self.weights['wc3_3'], strides=[1, 1, 1, 1], padding='SAME')
        conv3_3 = tf.reshape(tf.nn.bias_add(conv3_3, self.biases['bc3_3']), conv3_3.get_shape())

        conv3_4 = tf.nn.conv2d(tf.nn.relu(conv3_3), self.weights['wc3_4'], strides=[1, 1, 1, 1], padding='SAME')
        conv3_4 = tf.reshape(tf.nn.bias_add(conv3_4, self.biases['bc3_4']), conv3_4.get_shape())

        e3 = self.g_bn_e3(conv3_4)
        pool3 = tf.nn.max_pool(tf.nn.relu(e3), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv4_1 = tf.nn.conv2d(tf.nn.relu(pool3), self.weights['wc4_1'], strides=[1, 1, 1, 1], padding='SAME')
        conv4_1 = tf.reshape(tf.nn.bias_add(conv4_1, self.biases['bc4_1']), conv4_1.get_shape())

        conv4_2 = tf.nn.conv2d(tf.nn.relu(conv4_1), self.weights['wc4_2'], strides=[1, 1, 1, 1], padding='SAME')
        conv4_2 = tf.reshape(tf.nn.bias_add(conv4_2, self.biases['bc4_2']), conv4_2.get_shape())

        conv4_3 = tf.nn.conv2d(tf.nn.relu(conv4_2), self.weights['wc4_3'], strides=[1, 1, 1, 1], padding='SAME')
        conv4_3 = tf.reshape(tf.nn.bias_add(conv4_3, self.biases['bc4_3']), conv4_3.get_shape())

        conv4_4 = tf.nn.conv2d(tf.nn.relu(conv4_3), self.weights['wc4_4'], strides=[1, 1, 1, 1], padding='SAME')
        conv4_4 = tf.reshape(tf.nn.bias_add(conv4_4, self.biases['bc4_4']), conv4_4.get_shape())

        conv4_5 = tf.nn.conv2d(tf.nn.relu(conv4_4), self.weights['wc4_5'], strides=[1, 1, 1, 1], padding='SAME')
        conv4_5 = tf.reshape(tf.nn.bias_add(conv4_5, self.biases['bc4_5']), conv4_5.get_shape())

        conv4_6 = tf.nn.conv2d(tf.nn.relu(conv4_5), self.weights['wc4_6'], strides=[1, 1, 1, 1], padding='SAME')
        conv4_6 = tf.reshape(tf.nn.bias_add(conv4_6, self.biases['bc4_6']), conv4_6.get_shape())

        conv4_7 = tf.nn.conv2d(tf.nn.relu(conv4_6), self.weights['wc4_7'], strides=[1, 1, 1, 1], padding='SAME')
        conv4_7 = tf.reshape(tf.nn.bias_add(conv4_7, self.biases['bc4_7']), conv4_7.get_shape())

        e4 = self.g_bn_e4(conv4_7)

        conv5_1 = tf.nn.conv2d(tf.nn.relu(e4), self.weights['wc5_1'], strides=[1, 1, 1, 1], padding='SAME')
        conv5_1 = tf.reshape(tf.nn.bias_add(conv5_1, self.biases['bc5_1']), conv5_1.get_shape())

        conv5_2 = tf.nn.conv2d(tf.nn.relu(conv5_1), self.weights['wc5_2'], strides=[1, 1, 1, 1], padding='SAME')
        conv5_2 = tf.reshape(tf.nn.bias_add(conv5_2, self.biases['bc5_2']), conv5_2.get_shape())

        return conv5_2

    def save(self, checkpoint_dir, step):
        model_name = "pix2pix.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def test(self, args):
        """Test pix2pix"""
        tf.initialize_all_variables().run()

        #sample_files = glob('./datasets/human/val/*.jpg'.format(self.dataset_name))
        with open('./datasets/human/list/test_rgb_id.txt', 'r') as list_file:
            lines = list_file.readlines()
        # sample_files = lines[0:8]
        sample_files = np.random.choice(lines, 4)

        # load testing input
        print("Loading testing images ...")
        # sample_g, sample_d = [load_lip_data(sample_file) for sample_file in sample_files]
        batch_g = []
        batch_d = []
        batch_p = []
        for batch_file in sample_files:
            g_, d_, p_ = load_lip_data(batch_file)
            batch_g.append(g_)
            batch_d.append(d_)
            batch_p.append(p_)

        sample_images_g = np.array(batch_g).astype(np.float32)
        sample_images_d = np.array(batch_d).astype(np.float32)

        sample_images_g = [sample_images_g[i:i+self.batch_size]
                           for i in xrange(0, len(sample_images_g), self.batch_size)]
        sample_images_g = np.array(sample_images_g)
        sample_images_d = [sample_images_d[i:i+self.batch_size]
                           for i in xrange(0, len(sample_images_d), self.batch_size)]
        sample_images_d = np.array(sample_images_d)
        sample_p = [batch_p[i:i+self.batch_size] for i in xrange(0, len(batch_p), self.batch_size)]
        # print(sample_images_g.shape)
        # print(sample_images_d.shape)

        start_time = time.time()
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # for i, g_, d_ in enumerate(sample_images_g, sample_images_d):
        error_sum = 0
        for i in xrange(sample_images_g.shape[0]):
            idx = i
            print("sampling image ", idx)
            samples, samples_p = self.sess.run(
                [self.fake_B_sample, self.fake_P],
                feed_dict={self.real_data: sample_images_d[i], self.gen_data: sample_images_g[i],
                           self.point_data: sample_p[i]})
            # print (samples[0,:,:,0].shape)
            save_lip_images(samples, samples_p, self.batch_size, sample_files, 'test', idx)

            pose_gt = sample_images_d[i][:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]
            error_sum += np.linalg.norm(samples - pose_gt)
        print error_sum / self.batch_size / sample_images_g.shape[0]




