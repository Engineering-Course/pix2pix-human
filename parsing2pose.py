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
                 gf_dim=32, df_dim=32, L2_lambda=10, D_lambda=0.001,
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
        self.D_lambda = D_lambda

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
#------------------------network setting---------------------
        self.weights ={
            'wc1_1': tf.Variable(tf.truncated_normal([9, 9, 4, 128], stddev=0.01), name='wc1_1'),
            'wc2_1': tf.Variable(tf.truncated_normal([9, 9, 128, 128], stddev=0.01), name='wc2_1'),
            'wc3_1': tf.Variable(tf.truncated_normal([9, 9, 128, 128], stddev=0.01), name='wc3_1'),
            'wc4_1': tf.Variable(tf.truncated_normal([5, 5, 128, 32], stddev=0.01), name='wc4_1'),
            'wc5_1': tf.Variable(tf.truncated_normal([9, 9, 32, 512], stddev=0.01), name='wc5_1'),
            'wc6_1': tf.Variable(tf.truncated_normal([1, 1, 512, 512], stddev=0.01), name='wc6_1'),
            'wc7_1': tf.Variable(tf.truncated_normal([1, 1, 512, 16], stddev=0.01), name='wc7_1'),
            'fc8_1': tf.Variable(tf.truncated_normal([23*23, 1024], stddev=0.01), name='fc8_1'),
            'fc8_2': tf.Variable(tf.truncated_normal([1024, 1], stddev=0.01), name='fc8_2')
        }
        self.biases = {
            'bc1_1': tf.Variable(tf.constant(0.0, shape=[128]), name='bc1_1'),
            'bc2_1': tf.Variable(tf.constant(0.0, shape=[128]), name='bc2_1'),
            'bc3_1': tf.Variable(tf.constant(0.0, shape=[128]), name='bc3_1'),
            'bc4_1': tf.Variable(tf.constant(0.0, shape=[32]), name='bc4_1'),
            'bc5_1': tf.Variable(tf.constant(0.0, shape=[512]), name='bc5_1'),
            'bc6_1': tf.Variable(tf.constant(0.0, shape=[512]), name='bc6_1'),
            'bc7_1': tf.Variable(tf.constant(0.0, shape=[16]), name='bc7_1'),
            'bc8_1': tf.Variable(tf.constant(0.0, shape=[1024]), name='bc8_1'),
            'bc8_2': tf.Variable(tf.constant(0.0, shape=[1]), name='bc8_2'),
        }
        self.build_model()

    def build_model(self):
        self.gen_data = tf.placeholder(tf.float32,
                                        [self.batch_size, self.image_size, self.image_size,
                                         self.input_c_dim + 1],
                                        name='images_and_parsing')
        self.real_data = tf.placeholder(tf.float32,
                                        [self.batch_size, self.pose_size, self.pose_size,
                                         self.input_c_dim + 1 + self.output_c_dim],
                                        name='real_A_and_B_images')
        self.point_data = tf.placeholder(tf.float32, [self.batch_size, 16], name='point_label')

        self.real_A = self.real_data[:, :, :, :self.input_c_dim + 1]
        self.real_B = self.real_data[:, :, :, self.input_c_dim + 1 : self.input_c_dim + 1 + self.output_c_dim]

        self.fake_B = self.generator(self.gen_data)

        self.real_AB = tf.concat([self.real_A, self.real_B], 3)
        self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)

        self.D_real_logits = self.discriminator(self.real_AB, reuse=False)
        self.D_fake_logits = self.discriminator(self.fake_AB, reuse=True)

        self.fake_B_sample = self.sampler(self.gen_data)

        self.d_loss_real = tf.reduce_mean(self.D_real_logits)
        self.d_loss_fake = tf.reduce_mean(self.D_fake_logits)
        self.g_loss_d = self.D_lambda * (-self.d_loss_fake)
        self.g_loss_l2 = self.L2_lambda * tf.reduce_mean(tf.sqrt(tf.nn.l2_loss(self.real_B - self.fake_B) * 2))
        # self.g_loss_l2 = self.L2_lambda * tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.real_B, self .fake_B)))))
        # self.g_loss_l2 = 100 * tf.reduce_mean(tf.abs(self.real_B - self.fake_B))

        self.fake_P = []
        self.g_loss_p = 0
        for i in xrange(16):
            heatmap = self.fake_B[:,:,:,i]
            fake_Pi = self.generator_pose(heatmap[:,:,:,np.newaxis])
            self.fake_P.append(fake_Pi)
            self.g_loss_p += self.L2_lambda * tf.reduce_mean(tf.abs(fake_Pi-self.point_data[:,i:i+1]))

        # self.g_loss_l2 = self.L2_lambda * self.compute_loss_l2(self.real_B, self.fake_B, self.fake_P)

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        self.g_loss_d_sum = tf.summary.scalar("g_loss_d", self.g_loss_d)
        self.g_loss_l2_sum = tf.summary.scalar("g_loss_l2", self.g_loss_l2)
        self.g_loss_p_sum = tf.summary.scalar("g_loss_p", self.g_loss_p)

        self.d_loss = self.d_loss_fake - self.d_loss_real
        # self.g_loss = self.g_loss_d + self.g_loss_l2 + self.g_loss_p
        self.g_loss = self.g_loss_l2 + self.g_loss_p
        # self.g_loss = self.g_loss_l2

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'd_' not in var.name]

        self.saver = tf.train.Saver()

    def compute_loss_l2(self, real_B, fake_B, fake_P):
        loss_l2 = []
        for i in xrange(16):
            for j in xrange(self.batch_size):
                # pi = tf.round(fake_P[i][j][0])
                fake_i = fake_B[j, :, :, i]
                real_i = real_B[j, :, :, i]
                #fake_i = tf.scalar_mul(pi, fake_i)
                loss_l2.append(tf.sqrt(tf.nn.l2_loss(real_i - fake_i) * 2))
        return tf.reduce_mean(loss_l2)

    def load_random_samples(self, set_name):
        with open('./datasets/human/list/{}_rgb_id.txt'.format(set_name), 'r') as list_file:
            lines = list_file.readlines()
        data = np.random.choice(lines, self.batch_size)
        batch_g = []
        batch_d = []
        batch_p = []
        for batch_file in data:
            g_, d_, p_ = load_lip_data(batch_file)
            batch_g.append(g_)
            batch_d.append(d_)
            batch_p.append(p_)

        sample_g = np.array(batch_g).astype(np.float32)
        sample_d = np.array(batch_d).astype(np.float32)

        return sample_g, sample_d, batch_p, data

    def sample_model(self, sample_dir, epoch, idx):
        sample_g, sample_d, batch_p, sample_files = self.load_random_samples('val')
        samples, samples_p, d_loss, g_loss = self.sess.run(
            [self.fake_B_sample, self.fake_P, self.d_loss, self.g_loss],
            feed_dict={self.real_data: sample_d, self.gen_data: sample_g, self.point_data: batch_p})
        save_lip_images(samples, samples_p, self.batch_size, sample_files, 'sample')

        pose_gt = sample_d[:, :, :, self.input_c_dim + 1 : self.input_c_dim + 1 + self.output_c_dim]
        error_sum = np.linalg.norm(samples - pose_gt)
        print("l2 loss: {:.8f}.".format(error_sum / self.batch_size))
        print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

    def train(self, args):
        """Train pix2pix"""
        d_optim = (tf.train.RMSPropOptimizer(learning_rate=args.lr).minimize(self.d_loss, var_list=self.d_vars))
        g_optim = (tf.train.RMSPropOptimizer(learning_rate=args.lr).minimize(self.g_loss, var_list=self.g_vars))

        # clip D theta
        self.clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.d_vars]

        tf.global_variables_initializer().run()

        # self.g_sum = tf.summary.merge([self.g_loss_sum, self.g_loss_d_sum, self.g_loss_l2_sum, self.g_loss_p_sum])
        self.g_sum = tf.summary.merge([self.g_loss_sum, self.g_loss_l2_sum, self.g_loss_p_sum])
        # self.g_sum = tf.summary.merge([self.g_loss_sum, self.g_loss_l2_sum])
        self.d_sum = tf.summary.merge([self.d_loss_sum, self.d_loss_real_sum, self.d_loss_fake_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

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
                for batch_file in batch_files:
                    g_, d_, p_ = load_lip_data(batch_file)
                    batch_g.append(g_)
                    batch_d.append(d_)
                    batch_p.append(p_)

                batch_images_g = np.array(batch_g).astype(np.float32)
                batch_images_d = np.array(batch_d).astype(np.float32)

                # Update D network
                # for iter_d in range(4):
                #     D_sample_g, D_sample_d, D_batch_p, _ = self.load_random_samples('train')
                #     _, _ = self.sess.run([d_optim, self.clip_D],
                #                 feed_dict={ self.real_data: D_sample_d, self.gen_data: D_sample_g, 
                #                             self.point_data: D_batch_p})

                # _, summary_str, errD, _ = self.sess.run([d_optim, self.d_sum, self.d_loss, self.clip_D],
                #                                feed_dict={ self.real_data: batch_images_d, self.gen_data: batch_images_g, 
                #                                            self.point_data: batch_p})
                # self.writer.add_summary(summary_str, counter)
                # Update G network
                _, summary_str, errG = self.sess.run([g_optim, self.g_sum, self.g_loss],
                                               feed_dict={ self.real_data: batch_images_d, self.gen_data: batch_images_g, 
                                                           self.point_data: batch_p})
                self.writer.add_summary(summary_str, counter)

                errD = 0
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs, time.time() - start_time, errD, errG))

                if np.mod(counter, 3000) == 1:
                    self.sample_model(args.sample_dir, epoch, idx)

                if np.mod(counter, 3000) == 2:
                    self.save(args.checkpoint_dir, counter)

    def discriminator(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
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

            return h3

    def generator(self, image, y=None):
        with tf.variable_scope("generator") as scope:

            conv1_1 = tf.nn.conv2d(image, self.weights['wc1_1'], strides=[1, 1, 1, 1], padding='SAME')
            conv1_1 = tf.reshape(tf.nn.bias_add(conv1_1, self.biases['bc1_1']), conv1_1.get_shape())
            pool1 = tf.nn.max_pool(tf.nn.relu(conv1_1), ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            
            conv2_1 = tf.nn.conv2d(tf.nn.relu(pool1), self.weights['wc2_1'], strides=[1, 1, 1, 1], padding='SAME')
            conv2_1 = tf.reshape(tf.nn.bias_add(conv2_1, self.biases['bc2_1']), conv2_1.get_shape())
            pool2 = tf.nn.max_pool(tf.nn.relu(conv2_1), ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

            conv3_1 = tf.nn.conv2d(tf.nn.relu(pool2), self.weights['wc3_1'], strides=[1, 1, 1, 1], padding='SAME')
            conv3_1 = tf.reshape(tf.nn.bias_add(conv3_1, self.biases['bc3_1']), conv3_1.get_shape())
            pool3 = tf.nn.max_pool(tf.nn.relu(conv3_1), ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

            conv4_1 = tf.nn.conv2d(tf.nn.relu(pool3), self.weights['wc4_1'], strides=[1, 1, 1, 1], padding='SAME')
            conv4_1 = tf.reshape(tf.nn.bias_add(conv4_1, self.biases['bc4_1']), conv4_1.get_shape())

            conv5_1 = tf.nn.conv2d(tf.nn.relu(conv4_1), self.weights['wc5_1'], strides=[1, 1, 1, 1], padding='SAME')
            conv5_1 = tf.reshape(tf.nn.bias_add(conv5_1, self.biases['bc5_1']), conv5_1.get_shape())
            drop5_1 = tf.nn.dropout(tf.nn.relu(conv5_1), 0.5)

            conv6_1 = tf.nn.conv2d(tf.nn.relu(drop5_1), self.weights['wc6_1'], strides=[1, 1, 1, 1], padding='SAME')
            conv6_1 = tf.reshape(tf.nn.bias_add(conv6_1, self.biases['bc6_1']), conv6_1.get_shape())
            drop6_1 = tf.nn.dropout(tf.nn.relu(conv6_1), 0.5)

            conv7_1 = tf.nn.conv2d(tf.nn.relu(drop6_1), self.weights['wc7_1'], strides=[1, 1, 1, 1], padding='SAME')
            conv7_1 = tf.reshape(tf.nn.bias_add(conv7_1, self.biases['bc7_1']), conv7_1.get_shape())

            return conv7_1

    def generator_pose(self, image, y=None):
        with tf.variable_scope("generator_pose") as scope:
            pool4 = tf.nn.max_pool(tf.nn.relu(image), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            fc8_1 = tf.reshape(pool4, [-1, self.weights['fc8_1'].get_shape().as_list()[0]])
            fc8_1 = tf.add(tf.matmul(fc8_1, self.weights['fc8_1']), self.biases['bc8_1'])
            fc8_1 = tf.nn.dropout(tf.nn.relu(fc8_1), 0.5)
            fc8_2 = tf.add(tf.matmul(fc8_1, self.weights['fc8_2']), self.biases['bc8_2'])

            return tf.nn.sigmoid(fc8_2)

    def sampler_pose(self, image, y=None):
        with tf.variable_scope("generator_pose") as scope:
            tf.get_variable_scope().reuse_variables()

            pool4 = tf.nn.max_pool(tf.nn.relu(image), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            fc8_1 = tf.reshape(pool4, [-1, self.weights['fc8_1'].get_shape().as_list()[0]])
            fc8_1 = tf.add(tf.matmul(fc8_1, self.weights['fc8_1']), self.biases['bc8_1'])
            fc8_1 = tf.nn.dropout(tf.nn.relu(fc8_1), 0.5)
            fc8_2 = tf.add(tf.matmul(fc8_1, self.weights['fc8_2']), self.biases['bc8_2'])

            return tf.nn.sigmoid(fc8_2)

    def sampler(self, image, y=None):

         with tf.variable_scope("generator") as scope:
            # scope.reuse_variables()
            tf.get_variable_scope().reuse_variables()

            conv1_1 = tf.nn.conv2d(image, self.weights['wc1_1'], strides=[1, 1, 1, 1], padding='SAME')
            conv1_1 = tf.reshape(tf.nn.bias_add(conv1_1, self.biases['bc1_1']), conv1_1.get_shape())
            pool1 = tf.nn.max_pool(tf.nn.relu(conv1_1), ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            
            conv2_1 = tf.nn.conv2d(tf.nn.relu(pool1), self.weights['wc2_1'], strides=[1, 1, 1, 1], padding='SAME')
            conv2_1 = tf.reshape(tf.nn.bias_add(conv2_1, self.biases['bc2_1']), conv2_1.get_shape())
            pool2 = tf.nn.max_pool(tf.nn.relu(conv2_1), ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

            conv3_1 = tf.nn.conv2d(tf.nn.relu(pool2), self.weights['wc3_1'], strides=[1, 1, 1, 1], padding='SAME')
            conv3_1 = tf.reshape(tf.nn.bias_add(conv3_1, self.biases['bc3_1']), conv3_1.get_shape())
            pool3 = tf.nn.max_pool(tf.nn.relu(conv3_1), ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

            conv4_1 = tf.nn.conv2d(tf.nn.relu(pool3), self.weights['wc4_1'], strides=[1, 1, 1, 1], padding='SAME')
            conv4_1 = tf.reshape(tf.nn.bias_add(conv4_1, self.biases['bc4_1']), conv4_1.get_shape())

            conv5_1 = tf.nn.conv2d(tf.nn.relu(conv4_1), self.weights['wc5_1'], strides=[1, 1, 1, 1], padding='SAME')
            conv5_1 = tf.reshape(tf.nn.bias_add(conv5_1, self.biases['bc5_1']), conv5_1.get_shape())
            drop5_1 = tf.nn.dropout(tf.nn.relu(conv5_1), 0.5)

            conv6_1 = tf.nn.conv2d(tf.nn.relu(drop5_1), self.weights['wc6_1'], strides=[1, 1, 1, 1], padding='SAME')
            conv6_1 = tf.reshape(tf.nn.bias_add(conv6_1, self.biases['bc6_1']), conv6_1.get_shape())
            drop6_1 = tf.nn.dropout(tf.nn.relu(conv6_1), 0.5)

            conv7_1 = tf.nn.conv2d(tf.nn.relu(drop6_1), self.weights['wc7_1'], strides=[1, 1, 1, 1], padding='SAME')
            conv7_1 = tf.reshape(tf.nn.bias_add(conv7_1, self.biases['bc7_1']), conv7_1.get_shape())

            return conv7_1

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

        with open('./datasets/human/list/test_rgb_id.txt', 'r') as list_file:
            lines = list_file.readlines()
        sample_files = np.random.choice(lines, 4)

        # load testing input
        print("Loading testing images ...")
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

        start_time = time.time()
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        error_sum = 0
        for i in xrange(sample_images_g.shape[0]):
            idx = i
            print("sampling image ", idx)
            samples, samples_p = self.sess.run(
                [self.fake_B_sample, self.fake_P],
                feed_dict={self.real_data: sample_images_d[i], self.gen_data: sample_images_g[i],
                           self.point_data: sample_p[i]})
            save_lip_images(samples, samples_p, self.batch_size, sample_files, 'test', idx)

            pose_gt = sample_images_d[i][:, :, :, self.input_c_dim + 1 : self.input_c_dim + 1 + self.output_c_dim]
            error_sum += np.linalg.norm(samples - pose_gt)
        print error_sum / self.batch_size / sample_images_g.shape[0]




