""" Using convolutional net
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import numpy as np
import time
import tensorflow as tf
from pathlib import Path
from dataclass import Dataclass
import pandas as pd
import utils_upgraded as utils
from pathlib import Path
from dataclass import Dataclass
from tensorflow.python import debug as tf_debug


class Logger(object):
    def __init__(self):
        self.info = print


def write_df2csv(self, df, fname):
    self.logger.info("Save results to:")
    self.logger.info(f"{fname}")
    df.to_csv(fname)


def check_dir_or_mkdir(self, _pathobj):
    if not _pathobj.exists():
        self.logger.info(f"Directory created {_pathobj.as_posix()}")
        _pathobj.mkdir(parents=True, exist_ok=True)


def batch_normalization(conv, scope_name='batch_normalization'):
    with tf.compat.v1.variable_scope(scope_name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
        _mean, _var = tf.nn.moments(conv, [0, 1, 2])
        normed = tf.nn.batch_normalization(conv, _mean, _var, 0, 1, 0.0001)
    return normed


def conv_relu(inputs, filters, k_size, stride, padding, scope_name):
    '''
    A method that does convolution + relu on inputs
    '''
    with tf.compat.v1.variable_scope(scope_name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
        in_channels = inputs.shape[-1]
        kernel = tf.compat.v1.get_variable('kernel',
                                [k_size, k_size, in_channels, filters],
                                initializer=tf.compat.v1.truncated_normal_initializer())
        biases = tf.compat.v1.get_variable('biases',
                                [filters],
                                initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(input=inputs, filters=kernel, strides=[1, stride, stride, 1], padding=padding)
        conv = batch_normalization(conv, scope_name='batch_normalization')
    #return tf.nn.relu(conv + biases, name=scope.name)
    return tf.nn.tanh(conv + biases, name=scope.name)


def maxpool(inputs, ksize, stride, padding='VALID', scope_name='pool'):
    '''A method that does max pooling on inputs'''
    with tf.compat.v1.variable_scope(scope_name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
        pool = tf.nn.max_pool2d(input=inputs,
                            ksize=[1, ksize, ksize, 1],
                            strides=[1, stride, stride, 1],
                            padding=padding)
    return pool


def fully_connected(inputs, out_dim, scope_name='fc'):
    '''
    A fully connected linear layer on inputs
    '''
    with tf.compat.v1.variable_scope(scope_name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
        in_dim = inputs.shape[-1]
        w = tf.compat.v1.get_variable('weights', [in_dim, out_dim],
                            initializer=tf.compat.v1.truncated_normal_initializer())
        b = tf.compat.v1.get_variable('biases', [out_dim],
                            initializer=tf.compat.v1.constant_initializer(0.0))
        out = tf.matmul(inputs, w) + b
    return out


class ConvNetAqueduct(object):
    def __init__(self):
        self.lr = 0.001
        self.momentum = 0.9
        self.batch_size = 8
        self.keep_prob = tf.constant(0.75)
        self.gstep = tf.Variable(0, dtype=tf.int32,
                                 trainable=False, name='global_step')
        self.best_acc = 0
        self.n_classes = 2
        self.skip_step = 10
        self.n_test = 8
        self.is_training = True
        self.ratio_val = 0.2
        self.num_train = None
        self.join_indices = True
        self.indices = False
        self.assumed_n_subject = None
        self.logits_train = list()
        self.logits_validation = list()
        self.label_train = list()
        self.label_validation = list()
        self.logger = Logger()
        self.debug_level = 0
        self.d_results = Path("/home/foucault/projects/iNPH/results_code_review")
        self.branch = "aqueduct"
        self.fold_id = ""


    def get_function_name(self, fn, debug=1):
        if debug == 1:
            self.logger.info(f"Call: {fn.__qualname__}")
        elif debug == 2:
            self.logger.info(f"Call: {fn.__qualname__}")
            self.logger.info(f"From source: <ConvNet> in dataclass.py")


    def prepare_dataset(self, train_set, test_set, buffer_size=None):
        train_data = tf.data.Dataset.from_tensor_slices(train_set)
        if not buffer_size == None:
            train_data = train_data.shuffle(buffer_size) # if you want to shuffle your data
        train_data = train_data.batch(self.batch_size)
        test_data = tf.data.Dataset.from_tensor_slices(test_set)
        test_data = test_data.batch(self.batch_size)
        return train_data, test_data


    def get_data(self):
        with tf.compat.v1.name_scope('data'):
            dataset_key = self.branch
            dc = Dataclass()
            dc.logger = self.logger
            dc.indices = self.indices
            if not self.num_train:
                if self.join_indices == True:
                    self.num_train = int(np.round(dc.assumed_joined_n_subject*(1 - self.ratio_val)))
                else:
                    self.num_train = int(np.round(dc.dict_n_subject[dataset_key]*(1 - self.ratio_val)))
            self.n_test = dc.dict_n_subject[dataset_key] - self.num_train
            self.logger.info(f"num_train: {self.num_train}")
            dc.get_function_name(dc.get_adueduct_data_ml_train, debug=2)
            train_np, val_np = dc.get_adueduct_data_ml_train(self.num_train)
            train_data, test_data = self.prepare_dataset(train_np, val_np)
            iterator = tf.compat.v1.data.Iterator.from_structure(tf.compat.v1.data.get_output_types(train_data),
                                                   tf.compat.v1.data.get_output_shapes(train_data))
            img, self.label = iterator.get_next()
            reshape = [-1]
            reshape.extend([i for i in dc.dict_image_reshape_slice[dataset_key]])
            self.logger.info(f"dataset: {dataset_key}")
            self.logger.info(f"img shape: {reshape}")
            self.img = tf.reshape(img, shape=reshape)
            # reshape the image to make it work with tf.nn.conv2d

            self.train_init = iterator.make_initializer(train_data)  # initializer for train_data
            self.test_init = iterator.make_initializer(test_data)    # initializer for train_data


    def inference(self):
        conv1 = conv_relu(inputs=self.img,
                          filters=16,
                          k_size=8,
                          stride=1,
                          padding='SAME',
                          scope_name='conv1')
        pool1 = maxpool(conv1, 2, 2, 'VALID', 'pool1')
        conv2 = conv_relu(inputs=pool1,
                          filters=8,
                          k_size=3,
                          stride=1,
                          padding='SAME',
                          scope_name='conv2')
        pool2 = maxpool(conv2, 2, 2, 'VALID', 'pool2')
        feature_dim = pool2.shape[1] * pool2.shape[2] * pool2.shape[3]
        pool2 = tf.reshape(pool2, [-1, feature_dim])
        fc = fully_connected(pool2, 8, 'fc')
        self.logits = fully_connected(fc, self.n_classes, 'logits')


    def loss(self):
        '''
        define loss function
        use softmax cross entropy with logits as the loss function
        compute mean cross entropy, softmax is applied internally
        '''
        #
        with tf.compat.v1.name_scope('loss'):
            entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.stop_gradient(self.label),
                logits=self.logits)
            self.loss = tf.reduce_mean(input_tensor=entropy, name='loss')


    def optimize(self):
        '''
        Define training op
        #using Adam Gradient Descent to minimize cost
        using Stochastic Gradient Descent with momentum to minimize cost
        '''
        #self.opt = tf.compat.v1.train.AdamOptimizer(
        #    self.lr).minimize(self.loss, global_step=self.gstep)
        self.opt = tf.compat.v1.train.MomentumOptimizer(
            self.lr, self.momentum).minimize(self.loss, global_step=self.gstep)


    def summary(self):
        '''
        Create summaries to write on TensorBoard
        '''
        with tf.compat.v1.name_scope('summaries'):
            tf.compat.v1.summary.scalar('loss', self.loss)
            tf.compat.v1.summary.scalar('accuracy', self.accuracy)
            tf.compat.v1.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.compat.v1.summary.merge_all()


    def eval(self):
        '''
        Count the number of right predictions in a batch
        '''
        with tf.compat.v1.name_scope('predict'):
            preds = tf.nn.softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(input=preds, axis=1), tf.argmax(input=self.label, axis=1))
            self.accuracy = tf.reduce_sum(input_tensor=tf.cast(correct_preds, tf.float32))
            self.f1_score = tf.contrib.metrics.f1_score(self.label, preds)


    # def eval(self):
        # '''
        # Count the number of right predictions in a batch
        # '''
        # with tf.compat.v1.name_scope('predict'):
            # self.preds = tf.nn.softmax(self.logits)
            # self.correct_preds = tf.equal(tf.argmax(input=self.preds, axis=1), tf.argmax(input=self.label, axis=1))
            # self.accuracy = tf.reduce_sum(input_tensor=tf.cast(self.correct_preds, tf.float32))


    def build(self):
        '''
        Build the computation graph
        '''
        if self.join_indices == True and not np.any(self.indices):
            self.logger.info(f"join_indices: {self.join_indices}")
            dc = Dataclass()
            self.assumed_joined_n_subject = dc.assumed_joined_n_subject
            self.indices = np.random.permutation(self.assumed_joined_n_subject)
        self.get_function_name(self.get_data)
        self.get_data()
        self.get_function_name(self.inference)
        self.inference()
        self.get_function_name(self.loss)
        self.loss()
        self.get_function_name(self.optimize)
        self.optimize()
        self.get_function_name(self.eval)
        self.eval()
        self.get_function_name(self.summary)
        self.summary()


    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.is_training = True
        total_loss = 0
        n_batches = 0
        try:
            while True:
                (_label,
                 _,
                 l,
                 logits,
                 summaries) = sess.run([
                     self.label,
                     self.opt,
                     self.loss,
                     self.logits,
                     self.summary_op])
                self.label_train.extend(_label)
                self.logits_train.extend(logits)
                if self.debug_level == 1:
                    self.debug_get_simple_size(_train=_label)
                writer.add_summary(summaries, global_step=step)
                if (step + 1) % self.skip_step == 0:
                    self.logger.info('Loss at step {0}: {1}'.format(step, l))
                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        fold_id = ""
        if self.fold_id:
            fold_id = "_fold"+self.fold_id
        f_save = self.d_results.joinpath("checkpoints", f"{self.branch}", f"{self.branch}"+fold_id)
        saver.save(sess, f_save, step)
        self.logger.info('Average loss at epoch {0}: {1}'.format(epoch, total_loss/n_batches))
        self.logger.info('Time taken: {0} seconds'.format(time.time() - start_time))
        return step


    def eval_once(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.is_training = False
        total_correct_preds = 0
        total_f1_score = 0
        count = 0
        n_label = 0
        try:
            while True:
                (_label,
                 accuracy_batch,
                 f1_score,
                 logits,
                 summaries) = sess.run([
                     self.label,
                     self.accuracy,
                     self.f1_score,
                     self.logits,
                     self.summary_op])
                self.label_validation.extend(_label)
                self.logits_validation.extend(logits)
                writer.add_summary(summaries, global_step=step)
                total_correct_preds += accuracy_batch
                total_f1_score += f1_score[1]
                count += 1
                n_label += len(_label)
                if self.debug_level == 1:
                    self.debug_get_simple_size(_validation=_label)
                    print(f"{accuracy_batch}")
                    print(f"{total_correct_preds}")
                    print(f"{count}")
                    print(f"{n_label}")
        except tf.errors.OutOfRangeError:
            pass
        acc = total_correct_preds/n_label
        self.logger.info('Accuracy at epoch {0}: {1} '.format(epoch, acc))
        self.logger.info('F1 score at epoch {0}: {1} '.format(epoch, total_f1_score/count))
        #self.logger.info('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds/(self.batch_size*count)))
        self.logger.info('Time taken: {0} seconds'.format(time.time() - start_time))
        return acc


    def save_best_model(self, saver, sess, step):
        fold_id = ""
        if self.fold_id:
            fold_id = "_fold"+self.fold_id
        f_save = self.d_results.joinpath("checkpoints",
                    f"{self.branch}",
                    f"{self.branch}"+fold_id+"_best").as_posix()
        saver.save(sess, f_save)


    def train(self, n_epochs):
        '''
        The train function alternates between training one epoch and evaluating
        '''
        checkpoint_dir = self.d_results.joinpath("checkpoints", f"{self.branch}")
        if not checkpoint_dir.exists():
            self.logger.info(f"Directory created {checkpoint_dir.as_posix()}")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        #writer = tf.compat.v1.summary.FileWriter('../results/graphs/convnet_aqueduct', tf.compat.v1.get_default_graph())

        with tf.compat.v1.Session() as sess:
            fold_id = ""
            if self.fold_id:
                fold_id = "_fold"+self.fold_id
            write_path = self.d_results.joinpath("graphs",
                f"{self.branch}+{fold_id}").as_posix()
            writer = tf.compat.v1.summary.FileWriter(write_path, sess.graph)
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.local_variables_initializer())
            if self.debug_level == 2:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            saver = tf.compat.v1.train.Saver()
            # ckpt = tf.train.get_checkpoint_state(os.path.dirname(
                # '../results/checkpoints/convnet_aqueduct/checkpoint'))
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            # if ckpt and ckpt.model_checkpoint_path:
                # saver.restore(sess, ckpt.model_checkpoint_path)

            step = self.gstep.eval()

            for epoch in range(n_epochs):
                step = self.train_one_epoch(sess, saver, self.train_init, writer, epoch, step)
                acc = self.eval_once(sess, self.test_init, writer, epoch, step)
                if acc >= self.best_acc:
                    self.best_acc = acc
                    self.logger.info('update model')
                    self.save_best_model(saver, sess, step)
                    self.label_train_best = self.label_train
                    self.logits_train_best = self.logits_train
                    self.label_validation_best = self.label_validation
                    self.logits_validation_best = self.logits_validation
                self.label_train = list()
                self.logits_train = list()
                self.label_validation = list()
                self.logits_validation = list()
            writer.close()
        #writer.close()


class ConvNetCSF(ConvNetAqueduct):
    def __init__(self):
        super().__init__()
        self.batch_size = 8
        self.n_classes = 2
        self.skip_step = 10
        self.n_test = 8
        self.is_training = True
        self.ratio_val = 0.2
        self.num_train = None
        self.branch = "csf_mask"


    def get_data(self):
        with tf.compat.v1.name_scope('data'):
            dataset_key = self.branch
            dc = Dataclass()
            dc.logger = self.logger
            dc.indices = self.indices
            if not self.num_train:
                if self.join_indices == True:
                    self.num_train = int(np.round(dc.assumed_joined_n_subject*(1 - self.ratio_val)))
                else:
                    self.num_train = int(np.round(dc.dict_n_subject[dataset_key]*(1 - self.ratio_val)))
            self.n_test = dc.dict_n_subject[dataset_key] - self.num_train
            self.logger.info(f"num_train: {self.num_train}")
            dc.get_function_name(dc.get_csf_data_ml_train, debug=2)
            train_np, val_np = dc.get_csf_data_ml_train(self.num_train)
            train_data, test_data = self.prepare_dataset(train_np, val_np)
            iterator = tf.compat.v1.data.Iterator.from_structure(tf.compat.v1.data.get_output_types(train_data),
                                                   tf.compat.v1.data.get_output_shapes(train_data))
            img, self.label = iterator.get_next()
            reshape = [-1]
            reshape.extend([i for i in dc.dict_image_reshape_slice[dataset_key]])
            self.logger.info(f"dataset: {dataset_key}")
            self.logger.info(f"img shape: {reshape}")
            self.img = tf.reshape(img, shape=reshape)
            self.train_init = iterator.make_initializer(train_data)
            self.test_init = iterator.make_initializer(test_data)


    def inference(self):
        conv1 = conv_relu(inputs=self.img,
                          filters=32,
                          k_size=16,
                          stride=1,
                          padding='SAME',
                          scope_name='conv1')
        pool1 = maxpool(conv1, 2, 2, 'VALID', 'pool1')
        conv2 = conv_relu(inputs=pool1,
                          filters=16,
                          k_size=3,
                          stride=1,
                          padding='SAME',
                          scope_name='conv2')
        pool2 = maxpool(conv2, 2, 2, 'VALID', 'pool2')
        feature_dim = pool2.shape[1] * pool2.shape[2] * pool2.shape[3]
        pool2 = tf.reshape(pool2, [-1, feature_dim])
        fc = fully_connected(pool2, 8, 'fc')
        self.logits = fully_connected(fc, self.n_classes, 'logits')


    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.is_training = True
        total_loss = 0
        n_batches = 0
        try:
            while True:
                #_, l, summaries = sess.run([self.opt, self.loss, self.summary_op])
                (_label,
                 _,
                 l,
                 logits,
                 summaries) = sess.run([
                     self.label,
                     self.opt,
                     self.loss,
                     self.logits,
                     self.summary_op])
                self.label_train.extend(_label)
                self.logits_train.extend(logits)
                if self.debug_level == 1:
                    self.debug_get_simple_size(_train=_label)
                writer.add_summary(summaries, global_step=step)
                if (step + 1) % self.skip_step == 0:
                    self.logger.info('Loss at step {0}: {1}'.format(step, l))
                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        fold_id = ""
        if self.fold_id:
            fold_id = "_fold"+self.fold_id
        f_save = self.d_results.joinpath("checkpoints", f"{self.branch}", f"{self.branch}"+fold_id)
        saver.save(sess, f_save, step)
        self.logger.info('Average loss at epoch {0}: {1}'.format(epoch, total_loss/n_batches))
        self.logger.info('Time taken: {0} seconds'.format(time.time() - start_time))
        return step


    def eval_once(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.is_training = False
        total_correct_preds = 0
        total_f1_score = 0
        count = 0
        n_label = 0
        try:
            while True:
                (_label,
                 accuracy_batch,
                 f1_score,
                 logits,
                 summaries) = sess.run([
                     self.label,
                     self.accuracy,
                     self.f1_score,
                     self.logits,
                     self.summary_op])
                self.label_validation.extend(_label)
                self.logits_validation.extend(logits)
                ##accuracy_batch, summaries = sess.run([self.accuracy, self.summary_op])
                #accuracy_batch, f1_score, summaries = sess.run([self.accuracy,
                #                                                self.f1_score,
                #                                                self.summary_op])
                #f1_score = sess.run(self.f1_score)
                writer.add_summary(summaries, global_step=step)
                total_correct_preds += accuracy_batch
                #print(f"f1: {f1_score}")
                total_f1_score += f1_score[1]
                count += 1
                n_label += len(_label)
                if self.debug_level == 1:
                    self.debug_get_simple_size(_validation=_label)
                    print(f"{accuracy_batch}")
                    print(f"{total_correct_preds}")
                    print(f"{count}")
        except tf.errors.OutOfRangeError:
            pass
        acc = total_correct_preds/n_label
        self.logger.info('Accuracy at epoch {0}: {1} '.format(epoch, acc))
        self.logger.info('F1 score at epoch {0}: {1} '.format(epoch, total_f1_score/count))
        #self.logger.info('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds/(self.batch_size*count)))
        self.logger.info('Time taken: {0} seconds'.format(time.time() - start_time))
        return acc


    def train(self, n_epochs):
        '''
        The train function alternates between training one epoch and evaluating
        '''
        checkpoint_dir = self.d_results.joinpath("checkpoints", f"{self.branch}")
        if not checkpoint_dir.exists():
            self.logger.info(f"Directory created {checkpoint_dir.as_posix()}")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        #writer = tf.compat.v1.summary.FileWriter('../results/graphs/convnet_csf', tf.compat.v1.get_default_graph())

        with tf.compat.v1.Session() as sess:
            fold_id = ""
            if self.fold_id:
                fold_id = "_fold"+self.fold_id
            write_path = self.d_results.joinpath("graphs",
                f"{self.branch}+{fold_id}").as_posix()
            writer = tf.compat.v1.summary.FileWriter(write_path, sess.graph)
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.local_variables_initializer())
            if self.debug_level == 2:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            saver = tf.compat.v1.train.Saver()
            # ckpt = tf.train.get_checkpoint_state(os.path.dirname(
                # '../results/checkpoints/convnet_csf/checkpoint'))
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            # if ckpt and ckpt.model_checkpoint_path:
                # saver.restore(sess, ckpt.model_checkpoint_path)

            step = self.gstep.eval()

            for epoch in range(n_epochs):
                step = self.train_one_epoch(sess, saver, self.train_init, writer, epoch, step)
                acc = self.eval_once(sess, self.test_init, writer, epoch, step)
                if acc >= self.best_acc:
                    self.best_acc = acc
                    self.logger.info('update model')
                    self.save_best_model(saver, sess, step)
                    self.label_train_best = self.label_train
                    self.logits_train_best = self.logits_train
                    self.label_validation_best = self.label_validation
                    self.logits_validation_best = self.logits_validation
                self.label_train = list()
                self.logits_train = list()
                self.label_validation = list()
                self.logits_validation = list()
            writer.close()
        #writer.close()


class Conv2Branch(ConvNetAqueduct):
    def __init__(self):
        super().__init__()
        self.batch_size = 8
        self.n_classes = 2
        self.skip_step = 10
        self.n_test = 8
        self.is_training = True
        self.ratio_val = 0.2
        self.join_indices = True
        self.indices = False
        self.num_train = None
        self.n_epochs = 3
        self.fold_id = ""
        self.best_acc = 0
        self.best_f1score = 0
        self.d_intermediate = Path("/home/foucault/projects/iNPH/results_code_review/intermediate")
        self.branch = "convnet_2branch_fc"
        self.dict_final_representation = {"aqueduct": self.restore_aqueduct_logits,
                                          "csf_mask": self.restore_csf_logits}


    def build_aqueduct_branch(self):
        model = ConvNetAqueduct()
        model.join_indices = self.join_indices
        model.ratio_val = self.ratio_val
        model.num_train = self.num_train
        model.batch_size = self.batch_size
        if np.any(self.indices):
            model.indices = self.indices
            model.build()
        else:
            model.build()
            self.indices = model.indices
        print(model.indices)
        if self.is_training:
            fold_id = ""
            if self.fold_id:
                fold_id = "_fold"+self.fold_id
            model.train(n_epochs=self.n_epochs)
            save2file = self.d_intermediate.joinpath("logits_aqueduct_train"+fold_id+".npy").as_posix()
            np.save(save2file, (np.array(model.logits_train_best),
                                np.array(model.label_train_best)))
            save2file = self.d_intermediate.joinpath("logits_aqueduct_validation"+fold_id+".npy").as_posix()
            np.save(save2file, (np.array(model.logits_validation_best),
                                np.array(model.label_validation_best)))
        return model


    def build_CSF_branch(self):
        model = ConvNetCSF()
        model.join_indices = self.join_indices
        model.ratio_val = self.ratio_val
        model.num_train = self.num_train
        model.batch_size = self.batch_size
        if np.any(self.indices):
            model.indices = self.indices
            model.build()
        else:
            model.build()
            self.indices = model.indices
        print(model.indices)
        if self.is_training:
            fold_id = ""
            if self.fold_id:
                fold_id = "_fold"+self.fold_id
            model.train(n_epochs=self.n_epochs)
            save2file = self.d_intermediate.joinpath("logits_csf_train"+fold_id+".npy").as_posix()
            np.save(save2file, (np.array(model.logits_train_best),
                                np.array(model.label_train_best)))
            save2file = self.d_intermediate.joinpath("logits_csf_validation"+fold_id+".npy").as_posix()
            np.save(save2file, (np.array(model.logits_validation_best),
                                np.array(model.label_validation_best)))
        return model


    def restore_aqueduct_logits(self):
        fold_id = ""
        if self.fold_id:
            fold_id = "_fold"+self.fold_id
        load_file = self.d_intermediate.joinpath("logits_aqueduct_train"+fold_id+".npy").as_posix()
        logits_aqueduct_train = np.load(load_file).astype(np.float32)
        load_file = self.d_intermediate.joinpath("logits_aqueduct_validation"+fold_id+".npy").as_posix()
        logits_aqueduct_validation = np.load(load_file).astype(np.float32)
        return logits_aqueduct_train, logits_aqueduct_validation


    def restore_csf_logits(self):
        fold_id = ""
        if self.fold_id:
            fold_id = "_fold"+self.fold_id
        load_file = self.d_intermediate.joinpath("logits_csf_train"+fold_id+".npy").as_posix()
        logits_csf_train = np.load(load_file).astype(np.float32)
        load_file = self.d_intermediate.joinpath("logits_csf_validation"+fold_id+".npy").as_posix()
        logits_csf_validation = np.load(load_file).astype(np.float32)
        return logits_csf_train, logits_csf_validation


    def merger_branch(self):
        if not self.d_intermediate.exists():
            self.logger.info(f"Directory created {self.d_intermediate.as_posix()}")
            self.d_intermediate.mkdir(parents=True, exist_ok=True)
        with tf.Graph().as_default() as g_1:
            aqueduct_model = self.build_aqueduct_branch()
        with tf.Graph().as_default() as g_2:
            csf_model = self.build_CSF_branch()


    def get_data(self):
        with tf.compat.v1.name_scope('data'):
            csf_train_np, csf_val_np = self.dict_final_representation["csf_mask"]()
            aqueduct_train_np, aqueduct_val_np = self.dict_final_representation["aqueduct"]()
            if (np.array_equal(csf_train_np[1], aqueduct_train_np[1]) and
                np.array_equal(csf_val_np[1], aqueduct_val_np[1])):
                train_np = (np.concatenate((aqueduct_train_np[0], csf_train_np[0]), axis=1),
                            aqueduct_train_np[1])
                val_np = (np.concatenate((aqueduct_val_np[0], csf_val_np[0]), axis=1),
                          aqueduct_val_np[1])
            train_data, test_data = self.prepare_dataset(train_np, val_np, buffer_size=None)
            iterator = tf.compat.v1.data.Iterator.from_structure(tf.compat.v1.data.get_output_types(train_data),
                                                   tf.compat.v1.data.get_output_shapes(train_data))
            img, self.label = iterator.get_next()
            dataset_key = "representation"
            reshape = [-1, 4]
            self.logger.info(f"dataset: {dataset_key}")
            self.logger.info(f"img shape: {reshape}")
            self.img = tf.reshape(img, shape=reshape)
            self.train_init = iterator.make_initializer(train_data)
            self.test_init = iterator.make_initializer(test_data)


    def inference(self):
        #fc = fully_connected(self.img, 2, 'fc')
        self.logits = fully_connected(self.img, self.n_classes, 'logits')


    def build(self):
        '''
        Build the computation graph
        '''
        self.get_function_name(self.get_data)
        self.get_data()
        self.get_function_name(self.inference)
        self.inference()
        self.get_function_name(self.loss)
        self.loss()
        self.get_function_name(self.optimize)
        self.optimize()
        self.get_function_name(self.eval)
        self.eval()
        self.get_function_name(self.summary)
        self.summary()


    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.is_training = True
        total_loss = 0
        n_batches = 0
        try:
            while True:
                (_,
                 l,
                 summaries) = sess.run([
                     self.opt,
                     self.loss,
                     self.summary_op])
                writer.add_summary(summaries, global_step=step)
                if (step + 1) % self.skip_step == 0:
                    self.logger.info('Loss at step {0}: {1}'.format(step, l))
                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        fold_id = ""
        if self.fold_id:
            fold_id = "_fold"+self.fold_id
        f_save = self.d_results.joinpath("checkpoints", f"{self.branch}", f"{self.branch}"+fold_id)
        saver.save(sess, f_save, step)
        self.logger.info('Average loss at epoch {0}: {1}'.format(epoch, total_loss/n_batches))
        self.logger.info('Time taken: {0} seconds'.format(time.time() - start_time))
        return step


    def eval_once(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.is_training = False
        total_correct_preds = 0
        total_f1_score = 0
        count = 0
        n_label = 0
        try:
            while True:
                (_label,
                 accuracy_batch,
                 f1_score,
                 summaries) = sess.run([
                     self.label,
                     self.accuracy,
                     self.f1_score,
                     self.summary_op])
                writer.add_summary(summaries, global_step=step)
                total_correct_preds += accuracy_batch
                total_f1_score += f1_score[1]
                count += 1
                n_label += len(_label)
                if self.debug_level == 1:
                    print(f"n_label={n_label}")
                    print(f"{accuracy_batch}")
                    print(f"{total_correct_preds}")
                    print(f"{count}")
        except tf.errors.OutOfRangeError:
            pass
        acc = total_correct_preds/n_label
        f1score = total_f1_score/count
        self.logger.info('Accuracy at epoch {0}: {1} '.format(epoch, acc))
        self.logger.info('F1 score at epoch {0}: {1} '.format(epoch, f1score))
        self.logger.info('Time taken: {0} seconds'.format(time.time() - start_time))
        return acc


    def train(self):
        '''
        The train function alternates between training one epoch and evaluating
        '''
        self.build()
        checkpoint_dir = self.d_results.joinpath("checkpoints", f"{self.branch}")
        if not checkpoint_dir.exists():
            self.logger.info(f"Directory created {checkpoint_dir.as_posix()}")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with tf.compat.v1.Session() as sess:
            fold_id = ""
            if self.fold_id:
                fold_id = "_fold"+self.fold_id
            write_path = self.d_results.joinpath("graphs",
                f"{self.branch}+{fold_id}").as_posix()
            writer = tf.compat.v1.summary.FileWriter(write_path, sess.graph)
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.local_variables_initializer())
            if self.debug_level == 2:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            saver = tf.compat.v1.train.Saver()
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            # if ckpt and ckpt.model_checkpoint_path:
                # saver.restore(sess, ckpt.model_checkpoint_path)

            step = self.gstep.eval()

            for epoch in range(self.n_epochs):
                step = self.train_one_epoch(sess, saver, self.train_init, writer, epoch, step)
                acc = self.eval_once(sess, self.test_init, writer, epoch, step)
                if acc >= self.best_acc:
                    self.best_acc = acc
                    self.logger.info('update model')
                    self.save_best_model(saver, sess, step)
                    self.label_train_best = self.label_train
                    self.logits_train_best = self.logits_train
                    self.label_validation_best = self.label_validation
                    self.logits_validation_best = self.logits_validation
                self.label_train = list()
                self.logits_train = list()
                self.label_validation = list()
                self.logits_validation = list()
            writer.close()


class CV10(object):
    def __init__(self):
        self.d_intermediate = Path("/home/foucault/projects/iNPH/results_code_review/intermediate")
        self.d_csv = Path("/home/foucault/projects/iNPH/results_code_review/csv")
        self.n_splits = 10
        self.num_train = None
        self.is_training = True
        self.cv_acc_list = list()
        self.batch_size = 8
        self.logger = Logger()


    def get_skf(self, shuffle=False):
        from sklearn.model_selection import StratifiedKFold
        if shuffle == True:
            random_state = np.random.randint(1, 100)
            self.logger.info(f"shuffle: {shuffle}")
            self.logger.info(f"random_state: {random_state}")
        skf = StratifiedKFold(n_splits=self.n_splits,
                              random_state=random_state,
                              shuffle=shuffle)
        self.logger.info(f"Initiate skf, n_splits: {self.n_splits}")
        return skf


    def get_skf_index(self, X, y):
        skf = self.get_skf(shuffle=True)
        indices = list()
        for train_index_, test_index_ in skf.split(X, y):
            indices.append(np.concatenate((train_index_, test_index_)))
        return indices, len(train_index_)


    def get_skf_index_by_dataset(self):
        dc = Dataclass()
        imgs_c, labels_c = dc.dataset_merge_target_control(
            dc.dict_fn_get_data["csf_mask"],
            dc.d_iNPH_mri_con1, dc.d_NC_mri)
        imgs_a, labels_a = dc.dataset_merge_target_control(
            dc.dict_fn_get_data["aqueduct"],
            dc.d_iNPH_aqueduct_con1, dc.d_NC_aqueduct)
        if labels_a.sum() == labels_c.sum():
            indices_list, num_train = self.get_skf_index(imgs_c, labels_c)
        else:
            raise ValueError("Labels mismatch")
        return indices_list, num_train


    def save_indices(self, fname, indices_list):
        np.save(fname, indices_list)


    def restore_indices(self, fname):
        indices_list = np.load(fname)
        return indices_list


    def get_indices_list(self):
        fname = self.d_intermediate.joinpath("indices_list.npy")
        if fname.exists():
            indices_list = self.restore_indices(fname.as_posix())
        if self.is_training:
            indices_list, self.num_train = self.get_skf_index_by_dataset()
            check_dir_or_mkdir(self, self.d_intermediate)
            self.save_indices(fname.as_posix(), indices_list)
        return indices_list


    def run_cv(self):
        indices_list = self.get_indices_list()
        for fold_id, indices in enumerate(indices_list):
            with tf.Graph().as_default() as g_3:
                model = Conv2Branch()
                model.fold_id = str(fold_id)
                model.is_training = self.is_training
                model.n_epochs = 3
                model.batch_size = self.batch_size
                if self.is_training:
                    model.merger_branch() # re-train each branch
                model.train()
                self.cv_acc_list.append(model.best_acc)
        self.logger.info(self.cv_acc_list)
        df = pd.DataFrame(data=self.cv_acc_list).T
        df["mean"] = np.mean(self.cv_acc_list, dtype=np.float32)
        fname = self.d_csv.joinpath("aqueduct_csf_acc_normalized_input.csv").as_posix()
        check_dir_or_mkdir(self, self.d_csv)
        write_df2csv(self, df, fname)


if __name__ == '__main__':
    #test_branches()
    model = CV10()
    model.is_training = True
    #model.is_training = False
    model.run_cv()
