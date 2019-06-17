from __future__ import print_function
from __future__ import division
import os
import time
import random
import datetime
import scipy.misc

from datetime import datetime
from util.util import *
from util.BasicConvLSTMCell import *
from util.SEBlock import  *
from util.SE_ResNeXt import *

import os, time, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import rawpy
import glob
from random import shuffle


class DEBLUR(object):
    def __init__(self, args):
        self.args = args
        self.n_levels = 3
        self.scale = 0.5

        self.chns = 3
        self.crop_size = 512
        self.ps = 512
        self.input_dir = './dataset/Sony/short/'
        self.gt_dir = './dataset/Sony/long/'

        self.result_dir = './result_Sony_no_lstm_v1/'
        self.save_freq = 1000

        self.train_fns = glob.glob(self.gt_dir + '0*.ARW')
        self.train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in self.train_fns]

        random.shuffle(self.train_ids)

        # 加载模型 
        self.train_dir = os.path.join('./checkpoints', args.model)

        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.data_size = (len(self.train_ids)) // self.batch_size
        self.max_steps = int(self.epoch * self.data_size)
        self.learning_rate = args.learning_rate
        self.ratio = 64#128

        self.training = tf.placeholder(tf.bool)

        if args.phase == 'train':
            self.training = tf.cast(True, tf.bool)
        else:
            self.training = tf.cast(False, tf.bool)


    def input_producer(self,batch_size):
        def pack_raw(raw):
            # pack Bayer image to 4 channels
            im = raw.raw_image_visible.astype(np.float32)
            im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

            im = np.expand_dims(im, axis=2)
            img_shape = im.shape
            H = img_shape[0]
            W = img_shape[1]

            out = np.concatenate((im[0:H:2, 0:W:2, :],
                                  im[0:H:2, 1:W:2, :],
                                  im[1:H:2, 1:W:2, :],
                                  im[1:H:2, 0:W:2, :]), axis=2)
            return out

        def read_data():
            def get_batches_fn(batch_size):

                index = [ind for ind in np.random.permutation(len(self.train_ids))]
                shuffle(index)

                for batch_i in range(0, len(index), self.batch_size):
                    images = []
                    gt_images = []
                    for ind in index[batch_i:batch_i + self.batch_size]:
                        train_id = self.train_ids[ind]
                        in_files = glob.glob(self.input_dir + '%05d_00*.ARW' % train_id)
                        in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
                        in_fn = os.path.basename(in_path)

                        gt_files = glob.glob(self.gt_dir + '%05d_00*.ARW' % train_id)
                        gt_path = gt_files[0]
                        gt_fn = os.path.basename(gt_path)

                        in_exposure = float(in_fn[9:-5])  # 曝光时间
                        gt_exposure = float(gt_fn[9:-5])
                        # 手动输入曝光比例
                        ratio = min(gt_exposure / in_exposure, 300)
                        # 直接乘以ratio
                        ###############################
                        raw = rawpy.imread(in_path)
                        raw = pack_raw(raw) * ratio
                        ###############################
                        gt_raw = rawpy.imread(gt_path)
                        im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
                        im = np.float32(im / 65535.0)

                        # crop
                        H = raw.shape[0]
                        W = raw.shape[1]

                        xx = np.random.randint(0, W - self.ps)
                        yy = np.random.randint(0, H - self.ps)

                        input_patch = raw[yy:yy + self.ps, xx:xx + self.ps, :]
                        gt_patch = im[yy * 2:yy * 2 + self.ps * 2, xx * 2:xx * 2 + self.ps * 2, :]


                        input_patch = np.minimum(input_patch, 1.0)

                        images.append(input_patch)
                        gt_images.append(gt_patch)
                    yield np.array(images), np.array(gt_images)

            return get_batches_fn

        with tf.variable_scope('input'):
            get_batches_fn = read_data()

        return get_batches_fn

    def generator(self, inputs, reuse=True, scope='g_net'):
        # 输入数据的数量 高度 宽度 通道数
        inputs = tf.convert_to_tensor(inputs)
        n, h, w, c = inputs.get_shape().as_list()

        x_unwrap = []
        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                activation_fn=tf.nn.relu, padding='SAME', normalizer_fn=None,
                                weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                biases_initializer=tf.constant_initializer(0.0)):

                inputs =  slim.conv2d(inputs, 3, [5, 5], scope='enc0_1')
                inp_pred = inputs
                for i in range(self.n_levels):
                    # 三轮循环
                    scale = self.scale ** (self.n_levels - i - 1)
                    hi = int(round(h * scale))
                    wi = int(round(w * scale))
                    inp_blur = tf.image.resize_images(inputs, [hi, wi], method=0)
                    inp_pred = tf.stop_gradient(tf.image.resize_images(inp_pred, [hi, wi], method=0))
                    inp_all = tf.concat([inp_blur, inp_pred], axis=3, name='inp')

                    if self.args.architecture == 'ResnetBlock':
                        # encoder  编码器
                        conv1_1 = slim.conv2d(inp_all, 32, [5, 5], scope='enc1_1')
                        conv1_2 = ResnetBlock(conv1_1, 32, 5, scope='enc1_2')
                        conv1_3 = ResnetBlock(conv1_2, 32, 5, scope='enc1_3')
                        #conv1_4 = ResnetBlock(conv1_3, 32, 5, scope='enc1_4')

                        conv2_1 = slim.conv2d(conv1_3, 64, [5, 5], stride=2, scope='enc2_1')
                        conv2_2 = ResnetBlock(conv2_1, 64, 5, scope='enc2_2')
                        conv2_3 = ResnetBlock(conv2_2, 64, 5, scope='enc2_3')
                        #conv2_4 = ResnetBlock(conv2_3, 64, 5, scope='enc2_4')

                        conv3_1 = slim.conv2d(conv2_3, 128, [5, 5], stride=2, scope='enc3_1')
                        conv3_2 = ResnetBlock(conv3_1, 128, 5, scope='enc3_2')
                        conv3_3 = ResnetBlock(conv3_2, 128, 5, scope='enc3_3')
                        #conv3_4 = ResnetBlock(conv3_3, 128, 5, scope='enc3_4')

                        deconv3_3 = conv3_3

                        # decoder 解码器
                        #deconv3_3 = ResnetBlock(deconv3_4, 128, 5, scope='dec3_3')
                        deconv3_2 = ResnetBlock(deconv3_3, 128, 5, scope='dec3_2')
                        deconv3_1 = ResnetBlock(deconv3_2, 128, 5, scope='dec3_1')
                        deconv2_4 = slim.conv2d_transpose(deconv3_1, 64, [4, 4], stride=2, scope='dec2_4')
                        cat2 = deconv2_4 + conv2_3 # conv2_4

                        #deconv2_3 = ResnetBlock(cat2, 64, 5, scope='dec2_3')
                        deconv2_2 = ResnetBlock(cat2, 64, 5, scope='dec2_2')
                        deconv2_1 = ResnetBlock(deconv2_2, 64, 5, scope='dec2_1')
                        deconv1_4 = slim.conv2d_transpose(deconv2_1, 32, [4, 4], stride=2, scope='dec1_4')

                        cat1 = deconv1_4 + conv1_3 #conv1_4
                        deconv1_3 = ResnetBlock(cat1, 32, 5, scope='dec1_3')
                        deconv1_2 = ResnetBlock(deconv1_3, 32, 5, scope='dec1_2')
                        deconv1_1 = ResnetBlock(deconv1_2, 32, 5, scope='dec1_1')

                        deconv1_0 = slim.conv2d(deconv1_1, 12, [1, 1], rate=1, activation_fn=None, scope='dec1_0')
                        inp_pred = tf.depth_to_space(deconv1_0, 2)


                    # num_feature*2 后  通道数分为两部分  一部分为细胞状态 一部分为隐藏状态

                    if i >= 0:
                        x_unwrap.append(inp_pred)
                    if i == 0:
                        tf.get_variable_scope().reuse_variables()

            return x_unwrap


    def build_model(self):

        get_batches_fn = self.input_producer(self.batch_size)
        img_in, img_gt = next(get_batches_fn(self.batch_size))

        tf.summary.image('img_in', im2uint8(img_in))
        tf.summary.image('img_gt', im2uint8(img_gt))

        # generator 计算多尺度损失
        x_unwrap = self.generator(img_in, reuse=False, scope='g_net')  # False

        self.loss_total = 0

        for i in range(self.n_levels):
            # 逐层计算损失 MSE
            _, hi, wi, _ = x_unwrap[i].get_shape().as_list()
            gt_i = tf.image.resize_images(img_gt, [hi, wi], method=0)
            loss = tf.reduce_mean((gt_i - x_unwrap[i]) ** 2)

            self.loss_total += loss

            tf.summary.image('out_' + str(i), im2uint8(x_unwrap[i]))
            tf.summary.scalar('loss_' + str(i), loss)
        # losses
        tf.summary.scalar('loss_total', self.loss_total)


        # training vars
        all_vars = tf.trainable_variables()
        self.all_vars = all_vars
        self.g_vars = [var for var in all_vars if 'g_net' in var.name]
        self.lstm_vars = [var for var in all_vars if 'LSTM' in var.name]
        for var in all_vars:
            print(var.name)



    def train(self):
        def get_optimizer(loss, global_step=None, var_list=None, is_gradient_clip=False):
            train_op = tf.train.AdamOptimizer(self.lr)
            if is_gradient_clip:
                grads_and_vars = train_op.compute_gradients(loss, var_list=var_list)
                unchanged_gvs = [(grad, var) for grad, var in grads_and_vars if not 'LSTM' in var.name]
                rnn_grad = [grad for grad, var in grads_and_vars if 'LSTM' in var.name]
                rnn_var = [var for grad, var in grads_and_vars if 'LSTM' in var.name]
                capped_grad, _ = tf.clip_by_global_norm(rnn_grad, clip_norm=3)
                capped_gvs = list(zip(capped_grad, rnn_var))
                train_op = train_op.apply_gradients(grads_and_vars=capped_gvs + unchanged_gvs, global_step=global_step)
            else:
                train_op = train_op.minimize(loss, global_step, var_list)
            return train_op

        global_step = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False)
        self.global_step = global_step

        self.build_model()

        # learning rate decay
        self.lr = tf.train.polynomial_decay(self.learning_rate, global_step, self.max_steps, end_learning_rate=0.0,
                                            power=0.3)
        tf.summary.scalar('learning_rate', self.lr)

        # training operators
        train_gnet = get_optimizer(self.loss_total, global_step, self.all_vars)

        # session and thread
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess = sess

        sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(max_to_keep=25, keep_checkpoint_every_n_hours=1)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # training summary
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph, flush_secs=30)

        for step in range(sess.run(global_step), self.max_steps + 1):

            start_time = time.time()

            # update G network
            _, loss_total_val = sess.run([train_gnet, self.loss_total])

            duration = time.time() - start_time
            # print loss_value
            assert not np.isnan(loss_total_val), 'Model diverged with loss = NaN'

            if step % 5 == 0:
                num_examples_per_step = self.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = (%.5f) (%.1f data/s; %.3f s/bch)')
                print(format_str % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), step, loss_total_val, examples_per_sec, sec_per_batch))

            if step % 20 == 0:
                # summary_str = sess.run(summary_op, feed_dict={inputs:batch_input, gt:batch_gt})
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or step == self.max_steps:
                checkpoint_path = os.path.join(self.train_dir, 'checkpoints')
                #print("checkpoint_path:", checkpoint_path)
                self.save(sess, checkpoint_path, step)


    def save(self, sess, checkpoint_dir, step):
        model_name = "deblur.model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, sess, checkpoint_dir, step=None):
        print(" [*] Reading checkpoints...")
        model_name = "deblur.model"
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if step is not None:
            ckpt_name = model_name + '-' + str(step)
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Reading intermediate checkpoints... Success")
            return str(step)
        elif ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            ckpt_iter = ckpt_name.split('-')[1]
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Reading updated checkpoints... Success")
            return ckpt_iter
        else:
            print(" [*] Reading checkpoints... ERROR")
            return False
    """
    def test(self, height, width, input_path, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        imgsName = sorted(os.listdir(input_path))

        H, W = height, width
        inp_chns = 3 if self.args.model == 'color' else 1
        self.batch_size = 1 if self.args.model == 'color' else 3
        inputs = tf.placeholder(shape=[self.batch_size, H, W, inp_chns], dtype=tf.float32)
        outputs = self.generator(inputs, reuse=False)

        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

        self.saver = tf.train.Saver()
        checkpoint_path = os.path.join(self.train_dir, 'checkpoints')  # ./checkpoints/SE/checkpoints
        self.load(sess, checkpoint_path, step=272000)

        for imgName in imgsName:
            blur = scipy.misc.imread(os.path.join(input_path, imgName))
            h, w, c = blur.shape
            # make sure the width is larger than the height
            rot = False
            if h > w:
                blur = np.transpose(blur, [1, 0, 2])
                rot = True
            h = int(blur.shape[0])
            w = int(blur.shape[1])
            resize = False
            if h > H or w > W:
                scale = min(1.0 * H / h, 1.0 * W / w)
                new_h = int(h * scale)
                new_w = int(w * scale)
                blur = scipy.misc.imresize(blur, [new_h, new_w], 'bicubic')
                resize = True
                blurPad = np.pad(blur, ((0, H - new_h), (0, W - new_w), (0, 0)), 'edge')
            else:
                blurPad = np.pad(blur, ((0, H - h), (0, W - w), (0, 0)), 'edge')
            blurPad = np.expand_dims(blurPad, 0)
            if self.args.model != 'color':
                blurPad = np.transpose(blurPad, (3, 1, 2, 0))

            start = time.time()
            deblur = sess.run(outputs, feed_dict={inputs: blurPad / 255.0})
            duration = time.time() - start
            print('Saving results: %s ... %4.3fs' % (os.path.join(output_path, imgName), duration))
            res = deblur[-1]
            if self.args.model != 'color':
                res = np.transpose(res, (3, 1, 2, 0))
            res = im2uint8(res[0, :, :, :])
            # crop the image into original size
            if resize:
                res = res[:new_h, :new_w, :]
                res = scipy.misc.imresize(res, [h, w], 'bicubic')
            else:
                res = res[:h, :w, :]

            if rot:
                res = np.transpose(res, [1, 0, 2])
            scipy.misc.imsave(os.path.join(output_path, imgName), res)
    """
    def test(self, height, width):
        def pack_raw(raw):
            im = raw.raw_image_visible.astype(np.float32)
            im = np.maximum(im - 512, 0) / (16383 - 512)  

            im = np.expand_dims(im, axis=2)
            img_shape = im.shape
            H = img_shape[0]
            W = img_shape[1]

            out = np.concatenate((im[0:H:2, 0:W:2, :],
                                  im[0:H:2, 1:W:2, :],
                                  im[1:H:2, 1:W:2, :],
                                  im[1:H:2, 0:W:2, :]), axis=2)
            return out

        input_dir = './dataset/Sony/short/'
        gt_dir = './dataset/Sony/long/'
        checkpoint_dir = './checkpoints/nolstm/'
        
        checkpoint_dir = os.path.join(checkpoint_dir, 'checkpoints')
        result_dir = './result_Sony_no_lstm_v1/'# ./result_Sony_no_lstm_v1/
        # get test IDs
        test_fns = glob.glob(gt_dir + '/1*.ARW')
        test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]
        #print("test_ids:",test_ids)

        H, W = height, width
        inp_chns = 4 
       
        in_image = tf.placeholder(shape=[self.batch_size, 1424, 2128, inp_chns], dtype=tf.float32)
        out_image = self.generator(in_image, reuse=False)

        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

        #saver = tf.train.Saver()
        self.saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        self.load(sess, checkpoint_dir, step=132000)

        """
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt:
            print('loaded.............. ' + ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        """

        for test_id in test_ids:
            # test the first image in each sequence
            in_files = glob.glob(input_dir + '%05d_00*.ARW' % test_id)
            #print("in_files:",in_files)
            for k in range(len(in_files)):
                in_path = in_files[k]
                in_fn = os.path.basename(in_path)
                print(in_fn)
                gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % test_id)
                gt_path = gt_files[0]
                gt_fn = os.path.basename(gt_path)
                in_exposure = float(in_fn[9:-5])
                gt_exposure = float(gt_fn[9:-5])
                ratio = min(gt_exposure / in_exposure, 300)

                raw = rawpy.imread(in_path)
                input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio

                im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
                # scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0)*ratio
                scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

                gt_raw = rawpy.imread(gt_path)
                im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
                gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

                #print("type(input_full):",type(input_full)) #  <class 'numpy.ndarray'>
                input_full = np.minimum(input_full, 1.0)

                output = sess.run(out_image, feed_dict={in_image: input_full})

                output = np.array(output[2])
                #print("type(output):",type(output)) # <class 'list'>
                output = np.minimum(np.maximum(output, 0), 1)

                output = output[0, :, :, :]       
                gt_full = gt_full[0, :, :, :]
                scale_full = scale_full[0, :, :, :]
                scale_full = scale_full * np.mean(gt_full) / np.mean(
                    scale_full)  # scale the low-light image to the same mean of the groundtruth

                if not os.path.isdir(result_dir + '%05d_%d' % (test_id,ratio)):
                    os.makedirs(result_dir + '%05d_%d' % (test_id,ratio))
                    # output
                    scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(
                        result_dir + '%05d_%d/%5d_00_%d_out.png' % (test_id, ratio,test_id, ratio))
                    # input
                    scipy.misc.toimage(scale_full * 255, high=255, low=0, cmin=0, cmax=255).save(
                        result_dir + '%05d_%d/%5d_00_%d_scale.png' % (test_id, ratio,test_id, ratio))
                    # ground truth
                    scipy.misc.toimage(gt_full * 255, high=255, low=0, cmin=0, cmax=255).save(
                        result_dir + '%05d_%d/%5d_00_%d_gt.png' % (test_id, ratio,test_id, ratio))

                    



