"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
import cv2
from time import gmtime, strftime
from scipy.stats import multivariate_normal
import scipy.io as sio 
import matplotlib.pyplot as plt

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

point_name = ['R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine',
              'Neck', 'Head', 'R_Wrist', 'R_Elbow', 'R_Shoulder', 'L_Shoulder', 'L_Elbow', 'L_Wrist']

# -----------------------------
# new added functions for pix2pix

def load_data(image_path, flip=True, is_test=False):
    img_A, img_B = load_image(image_path)
    img_A, img_B = preprocess_A_and_B(img_A, img_B, flip=flip, is_test=is_test)

    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.

    img_AB = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB

def load_image(image_path):
    input_img = imread(image_path)
    w = int(input_img.shape[1])
    w2 = int(w/2)
    img_A = input_img[:, 0:w2]
    img_B = input_img[:, w2:w]

    return img_A, img_B

#---------------------------------
# new added function for lip dataset (task parsing to pose)
def load_lip_data(image_id, phrase):
    parsing_size = 368
    pose_size = 46
    image_id = image_id[:-1] 
    image_path = './datasets/human/images/{}.jpg'.format(image_id)
    img = scipy.misc.imread(image_path).astype(np.float)
    parsing_path = './datasets/human/segmentations/{}.png'.format(image_id)
    parsing = scipy.misc.imread(parsing_path).astype(np.float)
    rows = img.shape[0]
    cols = img.shape[1]
    origin_g = scipy.misc.imresize(img, [parsing_size, parsing_size])
    parsing_g = scipy.misc.imresize(parsing, [parsing_size, parsing_size])
    img_g = np.concatenate((origin_g, parsing_g[:,:,np.newaxis]), axis=2)
    if phrase == 'test':
        return img_g
    origin_d = scipy.misc.imresize(img, [pose_size, pose_size])
    parsing_d = scipy.misc.imresize(parsing, [pose_size, pose_size])
    heatmap = np.zeros((pose_size, pose_size, 16), dtype=np.float64)
    pid = np.zeros((16), dtype=np.int)
    with open('./datasets/human/pose/{}.txt'.format(image_id), 'r') as f:
        lines = f.readlines()
    points = lines[0].split(',')
    for idx, point in enumerate(points):
        if idx % 2 == 0:
            c_ = int(point)
            c_ = min(c_, cols-1)
            c_ = max(c_, 0)
            c_ = int(pose_size * 1.0 * c_ / cols)
        else:
            r_ = int(point)
            r_ = min(r_, rows-1)
            r_ = max(r_, 0)
            r_ = int(pose_size * 1.0 * r_ / rows)
            if c_ + r_ == 0:
                heatmap[:,:,int(idx / 2)] = 0
                pid[int(idx/2)] = 0
                continue
            var = multivariate_normal(mean=[r_, c_], cov=2)
            pid[int(idx/2)] = 1
            for i in xrange(pose_size):
                for j in xrange(pose_size):
                    heatmap[i, j, int(idx / 2)] = var.pdf([i, j]) * 10
            # plt.clf()
            # plt.imshow(heatmap[:,:,int(idx/2)].T)
            # plt.show()
            # wait = raw_input()

    # origin_g = origin_g / 127.5 - 1.
    # origin_d = origin_d / 127.5 - 1.
    # parsing_g = parsing_g / 127.5 - 1.
    # parsing_d = parsing_d / 127.5 - 1.
    
    img_d = np.concatenate((origin_d, parsing_d[:,:,np.newaxis], heatmap), axis=2)
    return img_g, img_d, pid

# new added function for task, pose to parsing
def load_lip_data_t2(image_id, flip=False, is_test=False):
    fine_size=64
    image_id = image_id[:-1] 
    image_path = './datasets/human/masks/{}.png'.format(image_id)
    img_A = scipy.misc.imread(image_path).astype(np.float)
    rows = img_A.shape[0]
    cols = img_A.shape[1]
    img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
    img_B = np.zeros((fine_size, fine_size), dtype=np.float64)
    with open('./datasets/human/pose/{}.txt'.format(image_id), 'r') as f:
        lines = f.readlines()
    points = lines[0].split(',')
    for idx, point in enumerate(points):
        if idx % 2 == 0:
            c_ = int(point)
            c_ = min(c_, cols-1)
            c_ = max(c_, 0)
            c_ = int(fine_size * 1.0 * c_ / cols)
        else:
            r_ = int(point)
            r_ = min(r_, rows-1)
            r_ = max(r_, 0)
            r_ = int(fine_size * 1.0 * r_ / rows)
            if c_ + r_ == 0:
                continue
            var = multivariate_normal(mean=[r_, c_], cov=2)
            for i in xrange(fine_size):
                for j in xrange(fine_size):
                    img_B[i, j] += var.pdf([i, j]) * 1.0
    img_A = img_A/127.5 - 1.
    img_BA = np.concatenate((img_B[:,:,np.newaxis], img_A), axis=2)
    # print img_BA.shape
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_BA

#------------------------------------------------------------
def preprocess_lip_A_and_B(img_A, fine_size=256, flip=False, is_test=False):

    for i in xrange(img_B.shape[0]):
        for j in xrange(img_B.shape[1]):
            if img_B[i,j] != 0:
                reimg_B[int(fine_size*1.0/img_B.shape[0]*i), 
                        int(fine_size*1.0/img_B.shape[1]*j)] = img_B[i,j]

    return img_A, reimg_B

def preprocess_A_and_B(img_A, img_B, load_size=286, fine_size=256, flip=True, is_test=False):
    if is_test:
        img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
        img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])
    else:
        img_A = scipy.misc.imresize(img_A, [load_size, load_size])
        img_B = scipy.misc.imresize(img_B, [load_size, load_size])

        h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
        img_B = img_B[h1:h1+fine_size, w1:w1+fine_size]

        if flip and np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)

    return img_A, img_B

# -----------------------------

# new added function for lip dataset, saving pose
def save_lip_images(images, points, batch_size, sample_files, output_set, batch_idx=0):

    for i, image in enumerate(images):
        img_id = sample_files[batch_size * batch_idx + i][:-1]
        image_path = './datasets/human/masks/{}.png'.format(img_id)
        print img_id
        img_A = scipy.misc.imread(image_path).astype(np.float)
        rows = img_A.shape[0]
        cols = img_A.shape[1]
        with open('./{}/pose/{}.txt'.format(output_set, img_id), 'w') as f:
            for p in xrange(image.shape[2]):
                channel_ = image[:,:,p]
                p_ = points[p][i]
                r_ = 0
                c_ = 0
                channel_ = scipy.misc.imresize(channel_, [rows, cols])
                if p_ > 0.5:
                    r_, c_ = np.unravel_index(channel_.argmax(), channel_.shape)
                    # r_ = r_ * rows * 1.0 / channel_.shape[0]
                    # c_ = c_ * cols * 1.0 / channel_.shape[1]
                    # save_path = './{}/pose/{}_{}.png'.format(output_set, img_id, point_name[p])
                    # scipy.misc.imsave(save_path, channel_)
                f.write('%d %d ' % (int(c_), int(r_)))
                # print ('id: {}, p_: {}, r_: {}, c_: {}'.format(p, p_, r_, c_))
                # plt.clf()
                # plt.imshow(channel_.T)
                # plt.show()
                # wait = raw_input()
        # sio.savemat('./{}/pose/{}.mat'.format(output_set, img_id), {'result': image})

# new added function for lip dataset, saving parsing
def save_lip_images_t2(images, batch_size, sample_files, output_set, batch_idx=0):
    print images.shape
    for i, image in enumerate(images):
        img_id = sample_files[batch_size * batch_idx + i][:-1]
        image_path = './{}/parsing/{}.png'.format(output_set, img_id)
        scipy.misc.imsave(image_path, image)

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.


