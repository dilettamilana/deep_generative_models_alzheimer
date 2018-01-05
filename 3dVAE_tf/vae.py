import os
import sys
import time
import collections

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.contrib.tensorboard.plugins import projector

from scipy.ndimage.morphology import binary_dilation
from scipy import interpolate
from sklearn.metrics import *

from ops import *
import training_notes_helper as tn
import visualising_helper as visualising
import metric

from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(0)
tf.set_random_seed(0)

def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)

class VariationalAutoencoder(object):

    def __init__(self, session, training_notes, is_training, learning_rate=0.0001, epochs=10, enable_shuffle=True):
        self.sess=session
        self.training_notes=training_notes
        self.is_training=is_training
        self.learning_rate = learning_rate
        self.learning_rate_classify= learning_rate
        self.size_batch = 10
        self.n_z = 150
        self.epochs=epochs
        self.classifier_epochs=30
        self.display_step=5
        self.size_kernel=5
        self.save_dir = directory = "C:\\Users\\Diletta\\Documents\\tensorboard_output\\vae_3d_tf_" + time.strftime("%d%m_%H%M")+"\\"
        self.num_encoder_channels=32
        self.num_gen_channels=32
        self.enable_shuffle=enable_shuffle
        self.gradients_dilation=True
        self.min_weight=0
        self.max_weight=0
        self.times_l2_loss=5
        self.visualise_step=10
        self.save_step=30 #ma devo inserire Keyboard interrup!
        self.balance_loss=True #apply weights depending on class frequency to the classifier cost
        self.freeze_autoencoder_when_classifier= False
        self.get_only_balanced_data=False
        self.keep_prob=0.5
        self.progression=True

        assert not( self.balance_loss and self.get_only_balanced_data), "you either balance the loss or the data, not both!!"

        print("TENSORBOARD:\npython -m tensorflow.tensorboard --logdir=" + str(self.save_dir))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.define_inputs()
        self.current_weights_placeholder=tf.placeholder(tf.float32, shape=[self.size_batch, self.x_train.shape[1], self.x_train.shape[2], self.x_train.shape[3], 1])
        self.current_weights_var= tf.get_variable('current_cost_weights', shape=[self.size_batch, self.x_train.shape[1], self.x_train.shape[2], self.x_train.shape[3], 1])
        #print("current cost shape: ", self.current_cost_weights.shape)
        if (self.gradients_dilation):
            self.w_train=apply_gradients_dilation(self.x_train)
            self.w_test=apply_gradients_dilation(self.x_test)

        #self.class_weights_test= balance_classes(self.y_test) non lo userò mai, uso i pesi solo x il training
        self.class_weights_train= balance_classes(self.y_train)

        np.save(os.path.join(self.save_dir, "labels_train_set.npy"), self.y_train)
        np.save(os.path.join(self.save_dir, "labels_test_set.npy"), self.y_test)

        epochs_report = os.path.join(self.save_dir, 'epochs_report.txt')
        sys.stdout = open(epochs_report, 'a+')
        #(img_x, img_y, img_z, x_train, x_test, y_train, y_test)
        self.visualiser = visualising.Visualiser(self, _3d=True, n_z=self.n_z, size_batch=self.size_batch, save_dir=self.save_dir, img_x=self.img_x, img_y=self.img_y, img_z=self.img_z, x_train=self.x_train, x_test=self.x_test, y_train=self.y_train, y_test=self.y_test)

        self.save_training_notes()

        print("\n\nLet's start!")
        # tf Graph input
        self.x = tf.placeholder(
            tf.float32,
            shape=(self.size_batch, self.img_x, self.img_y, self.img_z, 1),
            name="input_images")
        # print("\n\n61")

        self.y=tf.placeholder(
            tf.int16,
            shape=(self.size_batch, 3),
            name="input_labels"
        )

        self.y_clustering=tf.placeholder(
            tf.int32,
            shape=(self.size_batch),
            name="input_labels"
        )

        self.class_balancing_weights=tf.placeholder(
            tf.float32,
            shape=(self.size_batch),
            name="class_balancing_weights"
        )

        self.dropout_keep_prob=tf.placeholder(
            tf.float32,
            shape=(),
            name="dropout_keep_prob"
        )

        # Create autoencoder network
        self._create_network()
        # Define loss function based variational upper-bound and
        # corresponding optimizer
        self._create_loss_optimizer()
        self.array_of_accuracy_test=[]
        self.array_of_accuracy_train=[]
        self.mean_dice_test=[]
        self.mean_dice_train=[]

        ###### SUMMARY ######
        self.min_weight_summary=tf.summary.scalar('min_weight', self.min_weight)
        self.max_weight_summary=tf.summary.scalar('max_weight', self.max_weight)
        self.z_summary = tf.summary.histogram('z', self.z)
        self.loss_total_summary = tf.summary.scalar('total_loss', self.cost)
        self.loss_reconstr_summary = tf.summary.scalar('reconstr_loss', tf.reduce_mean(self.reconstr_loss))
        self.loss_latent_summary = tf.summary.scalar('latent_loss', tf.reduce_mean(self.latent_loss))

        self.summary = tf.summary.merge_all()
        self.loss_classifier_summary=tf.summary.scalar('classifier_loss', tf.reduce_mean(self.classify_cost))

        self.train_writer = tf.summary.FileWriter(self.save_dir + '/train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.save_dir + '/test', self.sess.graph)
        self.embeddings_writer = tf.summary.FileWriter(os.path.join(self.save_dir, 'projector'), self.sess.graph)

        self.model_saver = tf.train.Saver()
        total_trainable_params= self.total_trainable_params()

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def total_trainable_params(self):
        total_trainable_params = 0
        for var in tf.trainable_variables():
            shape = var.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_trainable_params += variable_parameters

        print("TRAINABLE PARAMETERS: {0:,}".format(total_trainable_params))
        return total_trainable_params

    def define_inputs(self):
        ######  DATASET   #####
        self.dataset_list = ["Harp", "OASIS", "AIBL"]
        self.nc_only = False
        self.max_images = 5000

        ######  IMAGES   #####
        #self.image_info = "orig_patch26_32x32x64"
        self.image_info="orig_patch16"
        self.conv3d = True
        # 2 coronal
        # 1 sagittal
        # 0 axial
        self.selected_slice = None

        ######  LABELS   #####
        #self.label_info = "patch26"
        self.label_info="patch16"
        # fields = 6
        self.allFields = True
        self.oneHot = False

        if "16" in self.label_info:
            self.labelId=16
        elif "26" in self.label_info:
            self.labelId=26
        elif "17" in self.label_info:
            self.labelId = 17
        elif "27" in self.label_info:
            self.labelId = 27
        elif "6" in self.label_info:
            self.labelId=6

            #assert False, "network not ready for this input"

        x_train, y_train, x_test, y_test = import_mri_dataset(self.dataset_list, self.image_info, self.label_info,
                                                              self.selected_slice, self.oneHot, self.allFields, self.conv3d,
                                                              self.max_images, self.nc_only, self.get_only_balanced_data)

        x_train = np.reshape(x_train,
                             newshape=[x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3], 1])
        x_test = np.reshape(x_test, newshape=[x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3], 1])

        size_data = x_train.shape[0] + x_test.shape[0]

        self.x_train=x_train
        self.x_test=x_test
        self.y_train=y_train
        self.y_test=y_test
        self.img_x = x_train.shape[1]
        self.img_y = x_train.shape[2]
        self.img_z = x_train.shape[3]
        self.n_samples=x_train.shape[0]
        print(self.img_x)
        print(self.img_y)
        print(self.img_z)

        if (self.progression):
            self.x_progression, self.y_progression=import_progression_dataset(self.conv3d, self.image_info)
            self.x_progression=np.reshape(self.x_progression, newshape=[self.x_progression.shape[0], self.x_progression.shape[1], self.x_progression.shape[2], self.x_progression.shape[3], 1])

    def save_training_notes(self):

        # print(abspath(getsourcefile(lambda: 0)))
        dict = collections.OrderedDict([('notes', self.training_notes),
                                        ('file name', os.path.realpath(__file__)),
                                        ('dataset list', self.dataset_list),
                                        ('load conv_3d data', self.conv3d),
                                        ('load only nc data', self.nc_only),
                                        ('image_info', self.image_info),
                                        ('label_info', self.label_info),
                                        ('epochs', self.epochs),
                                        ('classifier epochs', self.classifier_epochs),
                                        ('size_batch', self.size_batch),
                                        ('train_set_size', self.x_train.shape[0]),
                                        ('test_set_size', self.x_test.shape[0]),
                                        ('max_images', self.max_images),
                                        ('learning_rate', self.learning_rate),
                                        #('beta1', self.beta1),
                                        #('decay_rate', self.decay_rate),
                                        ('size_kernel', self.size_kernel),
                                        ('img_x', self.img_x),
                                        ('img_y', self.img_y),
                                        ('img_z', self.img_z),
                                        #('num_encoder_channels', self.num_encoder_channels),
                                        ('n_z', self.n_z),
                                        ('num_gen_channels', self.num_gen_channels),
                                        #('enable_tile_label', self.enable_tile_label),
                                        #('tile_ratio', self.tile_ratio)
                                        ('times_l2_loss', self.times_l2_loss),
                                        ('visualise_step', self.visualise_step),
                                        ('save_step', self.save_step)
                                        ])
        tn.write_training_notes(self.save_dir, dict)

    def _create_network(self):
        # Initialize autoencode network weights and biases

        # Use recognition network to determine mean and
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.z_log_sigma_sq =\
                self._recognition_network(self.x)

        # Draw one sample z from Gaussian distribution
        eps = tf.random_normal((self.size_batch, self.n_z), 0, 1,
                               dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean,
                        tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        self.x_reconstr_mean = \
                self._generator_network()

        self.classified=self._classifier_network()

    def _recognition_network(self, image):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.

        print("\n\n\nENCODER\n")
        min_side = min(self.img_x, self.img_y, self.img_z)
        num_layers = int(np.log2(min_side)) - int(self.size_kernel / 2)
        # print("\n\n\n\n\nLARA: ", image.shape)
        current = image

        # the rest of the conv layers with stride 2
        assert tf.get_variable_scope().reuse == False, "scoping problem!"
        for i in range(1, num_layers):
            name = 'E_conv' + str(i)
            with tf.name_scope(name) as scope:
                current = conv3d(
                    input_map=current,
                    num_output_channels=self.num_encoder_channels * (2 ** i),
                    size_kernel=self.size_kernel,
                    add_bias=False,
                    name=name,
                    batchnorm=False
                )
                print(str(current.name) + " - " + str(current.get_shape()))
                current = tf.nn.relu(current)
                print(str(current.name) + " - " + str(current.get_shape()))


        # fully connection layer
        name = 'E_fc'
        with tf.name_scope(name) as scope:
            current = fc(
                input_vector=tf.reshape(current, [self.size_batch, -1]),
                num_output_length=self.n_z,
                name=name,
                batchnorm=False
            )
            print(str(current.name) + " - " + str(current.get_shape()))
            current = tf.nn.tanh(current)
            print(str(current.name) + " - " + str(current.get_shape()))
            z_mean=fc(current, num_output_length=self.n_z, name="E_mean", batchnorm=False)
            z_log_sigma_sq=fc(current, num_output_length=self.n_z, name="E_std", batchnorm=False)
            return (z_mean, z_log_sigma_sq)

    def _generator_network(self):
        print("\n\n\nGENERATOR\n")

        min_side = min(self.img_x, self.img_y, self.img_z)
        num_layers = int(np.log2(min_side)) - int(self.size_kernel / 2)  # resta due perchè ad ogni layer raddoppio
        size_mini_map = int(min_side / 2 ** num_layers)
        # fc layer

        name = 'G_fc'
        with tf.name_scope(name) as scope:
            current = fc(
                input_vector=self.z,
                num_output_length=self.num_gen_channels * size_mini_map * size_mini_map * size_mini_map,
                name=name,
                batchnorm=False
            )
            print(str(current.name) + " - " + str(current.get_shape()))
            # reshape to cube for deconv
            current = tf.reshape(current, [-1, size_mini_map, size_mini_map, size_mini_map, self.num_gen_channels])
            print(str(current.name) + " - " + str(current.get_shape()))
            current = tf.nn.relu(current)
            print(str(current.name) + " - " + str(current.get_shape()))
        # deconv layers with stride 2
        for i in range(num_layers):
            name = 'G_deconv' + str(i)
            with tf.name_scope(name) as scope:
                current = deconv3d(
                    input_map=current,
                    output_shape=[self.size_batch,
                                  size_mini_map * 2 ** (i + 1),
                                  size_mini_map * 2 ** (i + 1),
                                  size_mini_map * 2 ** (i + 1),  # added to make the sizes work
                                  int(self.num_gen_channels / 2 ** (i + 1))],
                    size_kernel=self.size_kernel,
                    add_bias=False,
                    name=name,
                    batchnorm=False
                )
                print(str(current.name) + " - " + str(current.get_shape()))
                current = tf.nn.relu(current)
                print(str(current.name) + " - " + str(current.get_shape()))


        if self.labelId==16:
            complete_stride_for_last_layer = [1, 2, 2, 1, 1]
        elif self.labelId==26:
            complete_stride_for_last_layer = [1, 1, 1, 2, 1]
        elif self.labelId==19 or self.labelId==27:
            complete_stride_for_last_layer = [1, 1, 1, 1, 1]
        elif self.labelId==6  or self.labelId==19:
            complete_stride_for_last_layer = [1, 1, 1, 2, 1]

        name = 'G_deconv' + str(i + 1)
        with tf.name_scope(name) as scope:
            current = deconv3d(
            input_map=current,
            output_shape=[self.size_batch,
                          self.img_x,
                          self.img_y,
                          self.img_z,
                          1],
            size_kernel=self.size_kernel,
            complete_stride_for_last_layer=complete_stride_for_last_layer,
            last_deconv=True,
            add_bias=False,
            name=name,
            batchnorm=False
            )
            print(str(current.name) + " - " + str(current.get_shape()))
            # output
            current = tf.nn.relu(current)  # because I have a binary map
            print(str(current.name) + " - " + str(current.get_shape()))
        # x_reconstr_mean = \
        #     tf.nn.sigmoid(tf.add(tf.matmul(current, weights['out_mean']),
        #                          biases['out_mean']))
        return current

    def _classifier_network(self):
        name='classifier_fc_1'
        with tf.name_scope(name) as scope:
            current = fc(
                input_vector=self.z_mean,
                num_output_length=int(self.n_z/2),
                name=name
            )
            current = tf.nn.relu(current)
            current = tf.nn.dropout(current, keep_prob=self.keep_prob)

        name='classifier_fc_2'
        with tf.name_scope(name) as scope:
            current = fc(
                input_vector=current,
                num_output_length=int(self.n_z / 4),
                name=name
            )
            current=tf.nn.relu(current)
            current = tf.nn.dropout(current, keep_prob=self.keep_prob)
            # current = fc(
            #     input_vector=current,
            #     num_output_length=int(self.n_z / 8),
            #     name='classifier_fc_3'
            # )
            # current=tf.nn.relu(current)

        name = 'classifier_fc_3'
        with tf.name_scope(name) as scope:
            current = fc(
                input_vector=current,
                num_output_length=3,
                name=name,
            )
        return current

    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluation of log(0.0)
        print("350: ",self.x_reconstr_mean.get_shape())
        print("351: ",self.x.get_shape())

        #reconstr_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.x_reconstr_mean, labels=self.x))
        #reconstr_loss=tf.nn.l2_loss(self.x_reconstr_mean - self.x)
        #|(anyscalar)w(x_act - x_pred)|2
        #print(self.current_cost_weights)
        before_mul_time=time.time()
        #self.reconstr_loss=np.multiply((self.x_reconstr_mean - self.x), self.current_weights_var)
        #weights = self.current_weights_var.eval()
        with tf.name_scope('reconstruction_loss') as scope:
            self.reconstr_loss = tf.nn.l2_loss(tf.multiply((self.x_reconstr_mean - self.x)*self.times_l2_loss,
                                         (tf.add(tf.ones([self.size_batch, self.x_train.shape[1], self.x_train.shape[2], self.x_train.shape[3], 1]), self.current_weights_var))))
        total_mul_time = time.time() - before_mul_time
        print("\tIt took %02d:%02d to multiply" %
              (int(total_mul_time % 3600 / 60), total_mul_time % 60))
        #self.reconstr_loss=tf.square((self.x_reconstr_mean - self.x))
        # reconstr_loss = \
        #      -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)
        #                     + (1 - self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean),
        #                     1)
        # # # 2.) The latent loss, which is defined as the Kullback Leibler divergence
        ##    between the distribution in latent space induced by the encoder on
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        print("362: ", self.z_log_sigma_sq.get_shape())
        print("363: ", self.z_mean.get_shape() )
        #latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
        #                                    - tf.square(self.z_mean)
        #                                    - tf.exp(self.z_log_sigma_sq), 1)
        #latent_loss = - 0.5 *tf.reduce_mean(1 + self.z_log_sigma_sq - tf.square(self.z_mean) - tf.exp(self.z_log_sigma_sq), axis=-1)
        with tf.name_scope('latent_loss') as scope:
            self.latent_loss = - 0.5 * tf.reduce_sum(
            1.0 +  self.z_log_sigma_sq -
            tf.square(self.z_mean) - tf.exp(self.z_log_sigma_sq), 1)
        print("\n\n LATENT LOSS", self.latent_loss.shape)

        self.loss_1d, self.loss_2d, self.loss_3d, _, _, _= metric.sparse_ml(n_clusters=3, n_code=self.n_z, feat=self.z, label=self.y_clustering) #info_type='scalar' and send onehot
        with tf.name_scope('total_vae_loss_optimiser') as scope:
            #self.cost = tf.reduce_mean(latent_loss+reconstr_loss)  # average over batch
            #self.cost = tf.reduce_mean(self.reconstr_loss)+tf.reduce_mean(self.latent_loss)
            self.cost = tf.reduce_mean(self.reconstr_loss)+tf.reduce_mean(self.latent_loss) #+ self.loss_1d+self.loss_2d+ self.loss_3d  # average over batch
            self.optimizer = \
                tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        with tf.name_scope('classifier_loss') as scope:
            self.classified_sigmoid= tf.argmax(tf.nn.softmax(self.classified),1)
            #if self.balance:
            #    self.classify_cost= tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=self.classified, onehot_labels=self.y, weights=self.class_balancing_weights))
            #else:
            if self.balance_loss:
                #self.classify_cost = tf.reduce_mean(tf.multiply(self.class_balancing_weights,tf.nn.softmax_cross_entropy_with_logits(logits=self.classified, labels=self.y)))
                self.classify_cost = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=self.classified, labels=tf.argmax(self.y, 1), weights=self.class_balancing_weights ))
            else:
                self.classify_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.classified, labels=self.y))

            if self.freeze_autoencoder_when_classifier:
                print("EXTRACTING CLASSIFIER VARIABLES")
                classifier_var_list = self._get_variable_list(name="classifier_fc")
                self.classifier_optimiser=tf.train.AdamOptimizer(learning_rate=self.learning_rate_classify).minimize(self.classify_cost, var_list=classifier_var_list)
            else:
                self.classifier_optimiser=tf.train.AdamOptimizer(learning_rate=self.learning_rate_classify).minimize(self.classify_cost)

        with tf.name_scope('classifier_accuracy') as scope:
            self.correct_prediction = tf.equal(tf.argmax(self.classified, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def _get_variable_list(self, name=""):
        variables = []
        for var in tf.trainable_variables():
            if name in var.name:
                print(var.name)
                variables.append(var)
        return variables

    def partial_fit(self, X, labels, dropout_keep_prob):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        weights=self.current_weights_var.eval()
        self.min_weight = np.min(weights)
        print("min weight: ", self.min_weight)
        self.max_weight = np.max(weights)
        print("max weight: ", self.max_weight)
        print("AFTER ASSIGNMENT\n\n\n: ", weights.shape)
        print("counting occurrencies before returning x")
        unique, counts = np.unique(weights, return_counts=True)
        print(dict(zip(unique, counts)))

        opt, cost = self.sess.run((self.optimizer, self.cost),
                                  feed_dict={self.x: X, self.dropout_keep_prob: dropout_keep_prob, self.y_clustering:labels})
        return cost

    def classifier_fit(self, X, labels, weights, dropout_keep_prob):
        opt, cost = self.sess.run((self.classifier_optimiser, self.classify_cost),
                                  feed_dict={self.x: X, self.y: labels, self.class_balancing_weights: weights, self.dropout_keep_prob:dropout_keep_prob})
        return cost

    def reconstruct(self, X, dropout_keep_prob):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.x: X, self.dropout_keep_prob:dropout_keep_prob})

    def calculate_embeddings(self):

        embedded_data=None
        for ind_batch in range(self.num_batches):
            # non ci vuole controllo per il limite dell'array! Niente index out of bound exception! Che figata!!!
            batch_images = self.x_train[ind_batch * self.size_batch:(ind_batch + 1) * self.size_batch]
            batch_labels = self.y_train[ind_batch * self.size_batch:(ind_batch + 1) * self.size_batch]
            new_embeddings= self.sess.run(
                self.z,
                feed_dict={self.x: batch_images}
            )
            print("new embeddings: ",new_embeddings.shape)
            if embedded_data is None:
                embedded_data=new_embeddings
            else:
                embedded_data = np.append(embedded_data, new_embeddings, axis=0)
        #with tf.get_variable_scope():
            #tf.get_variable_scope().reuse_variables()
        #embedding = tf.get_variable(embedded_data, trainable=False, name='embedding')
        embedding_var = tf.get_variable("embedding", initializer=embedded_data)
        self.sess.run(tf.global_variables_initializer())
        #tf.global_variables_initializer().run()
        print("TOTAL EMBEDDING SHAPE: ", embedding_var.shape)
        self.metadata = os.path.join(self.save_dir, 'projector', 'metadata.tsv')

        with open(self.metadata, 'w') as metadata_file:
            for i in range(self.y_train.shape[0]):
                metadata_file.write('%d\n' % self.y_train[i, 0])

        config = projector.ProjectorConfig()
        embed = config.embeddings.add();
        embed.tensor_name = embedding_var.name
        embed.metadata_path = self.metadata
        projector.visualize_embeddings(self.embeddings_writer, config)

    def train_vae(self):
        print("TRAIN VAE")
        self.num_batches = self.x_train.shape[0] // self.size_batch

        for epoch in range(self.epochs):
            avg_cost=0
            for ind_batch in range(self.num_batches):
                start_time = time.time()

                # non ci vuole controllo per il limite dell'array! Niente index out of bound exception! Che figata!!!
                batch_images = self.x_train[ind_batch * self.size_batch:(ind_batch + 1) * self.size_batch]
                batch_labels = self.y_train[ind_batch * self.size_batch:(ind_batch + 1) * self.size_batch]

                # potrei avere una batch eccezionalmente piu corta qui
                # current_batch_size = (ind_batch + 1) * self.size_batch - ind_batch * self.size_batch
                batch_images = np.array(batch_images).astype(np.float32)
                #weights=apply_gradients_dilation(batch_images)
                assign_op = self.current_weights_var.assign(self.current_weights_placeholder)
                weights= self.w_train[ind_batch * self.size_batch:(ind_batch + 1) * self.size_batch]
                print("BEFORE ASSIGNMENT\n\n\n: ", weights.shape)
                print("min: ", np.min(weights))
                print("max: ", np.max(weights))
                print("counting occurrencies before returning x")
                unique, counts = np.unique(weights, return_counts=True)
                print(dict(zip(unique, counts)))
                self.sess.run(assign_op, feed_dict={self.current_weights_placeholder: weights})
                print("NEW RUN, PARTIAL FIT")
                batch_labels=np.reshape(batch_labels[:,0], newshape=[self.size_batch])
                cost=self.partial_fit(batch_images, batch_labels, dropout_keep_prob=1)

               # import ipdb; ipdb.set_trace()
                avg_cost += cost / self.n_samples * self.size_batch

            self.test_plot(epoch)
            print("\nEpoch:[%3d/%3d]\n\tcost=%.4f" %(epoch + 1, self.epochs, avg_cost))
            if (np.mod(epoch, self.save_step)==0):
                self.save_latent(to_save="train")
                self.save_latent(to_save="test")

            if (np.mod(epoch, self.visualise_step)==0):
                print("READY TO VISUALISE")
                self.visualise(epoch)

            #### writing to TENSORBOARD ####
            train_batch_images = self.x_train[0: self.size_batch]
            test_batch_images = self.x_test[0: self.size_batch]

            write_test_summary = self.sess.run(self.summary,
                                               feed_dict={self.x: test_batch_images, self.y_clustering:batch_labels, self.dropout_keep_prob: 1})
            self.test_writer.add_summary(write_test_summary, epoch)
            write_train_summary = self.sess.run(self.summary,
                                                feed_dict={self.x: train_batch_images, self.y_clustering: batch_labels, self.dropout_keep_prob: 1})
            self.train_writer.add_summary(write_train_summary, epoch)

            # estimate left run time
            elapse = time.time() - start_time
            time_left = ((self.epochs - epoch - 1) * self.num_batches + (self.num_batches - ind_batch - 1)) * elapse
            print("\tTime left: %02d:%02d:%02d" %
                  (int(time_left / 3600), int(time_left % 3600 / 60), time_left % 60))
            self.model_saver.save(self.sess, os.path.join(self.save_dir, 'projector','model.ckpt'))
            #self.calculate_embeddings()

    def train_classifier(self):
        print("TRAIN CLASSIFIER")
        self.num_batches = self.x_train.shape[0] // self.size_batch
        print("SHAPING x_train", self.x_train.shape)
        print("SHAPING x_test", self.x_test.shape)
        print("SHAPING y_train", self.y_train.shape)
        print("SHAPING y_test", self.y_test.shape)
        for epoch in range(self.classifier_epochs):
            avg_cost = 0
            avg_acc=0
            # I cannot shuffle in the middle: remember that I apply a weight mask that depends on the ind_batch
            for ind_batch in range(self.num_batches):
                start_time = time.time()
                print("ind batch ", ind_batch)

                # non ci vuole controllo per il limite dell'array! Niente index out of bound exception! Che figata!!!
                batch_images = self.x_train[ind_batch * self.size_batch:(ind_batch + 1) * self.size_batch]
                batch_labels = self.y_train[ind_batch * self.size_batch:(ind_batch + 1) * self.size_batch][:,0]

                batch_labels=oneHot(batch_labels)
                batch_weights= self.class_weights_train[ind_batch * self.size_batch:(ind_batch + 1) * self.size_batch]

                assert np.ndim(batch_labels)==2 and batch_labels.shape[1]==3, "WRONG! batch_labels has shape "+ str(batch_labels.shape)

                # potrei avere una batch eccezionalmente piu corta qui
                # current_batch_size = (ind_batch + 1) * self.size_batch - ind_batch * self.size_batch
                batch_images = np.array(batch_images).astype(np.float32)

                # import ipdb; ipdb.set_trace()
                cost = self.classifier_fit(batch_images, batch_labels, batch_weights, dropout_keep_prob=self.keep_prob)
                avg_cost += cost / self.n_samples * self.size_batch

            #self.test_plot(epoch)
            print("\nEpoch:[%3d/%3d]\n\tcost=%.4f" % (epoch + 1, self.classifier_epochs, avg_cost))
            print("HERE ARE THE WEIGHTS:")
            print(self.class_weights_train)
            self.calculate_accuracy(self.x_test, self.y_test, epoch, "test")
            self.calculate_accuracy(self.x_train, self.y_train, epoch, "train")

            # estimate left run time
            elapse = time.time() - start_time
            time_left = ((self.classifier_epochs - epoch - 1) * self.num_batches + (self.num_batches - ind_batch - 1)) * elapse
            print("\tTime left: %02d:%02d:%02d" %
                  (int(time_left / 3600), int(time_left % 3600 / 60), time_left % 60))
            #self.model_saver.save(self.sess, os.path.join(self.save_dir, 'projector', 'model.ckpt'))
            # self.calculate_embeddings()
        self.visualise_supervised_finetuning()

    def save_latent(self, to_save):

        if to_save=='train':
            x=self.x_train
            y=self.y_train

        else:
            x = self.x_test
            y = self.y_test

        latents=self.encode_entire_dataset(x,y)

        print("latents shape: ", latents.shape)
        np.save(os.path.join(self.save_dir, "encoder_mean_"+ str(to_save)+".npy"), latents)
        print("latents "+str(to_save)+" saved.")

    def calculate_dice_loss(self, x, y, epoch, test_or_train):

        num_batches = x.shape[0] // self.size_batch
        dice_score=np.zeros(shape=[x.shape[0]], dtype=np.float32)
        for ind_batch in range(num_batches):
            batch_images = x[ind_batch * self.size_batch:(ind_batch + 1) * self.size_batch]
            batch_labels = y[ind_batch * self.size_batch:(ind_batch + 1) * self.size_batch][:, 0]

            batch_images = np.array(batch_images).astype(np.float32)
            batch_labels = oneHot(batch_labels)
            reconstructed = self.reconstruct(batch_images, dropout_keep_prob=1)
            print(test_or_train)
            for i in range(self.size_batch):
                print("i ", i)
                print("index: ", ind_batch*self.size_batch+i)
                dice= self.dice_score(batch_images[i], reconstructed[i])
                #print("DICE ", dice)
                dice_score[ind_batch*self.size_batch+i] = dice

        print("total dice", dice_score)
        mean=float(np.mean(dice_score))
        print("mean", mean)

        if (test_or_train == "test"):
            self.mean_dice_test.append(mean)
        else:
            self.mean_dice_train.append(mean)

        if epoch == (self.epochs - 1):
            print("dice plot")

            if (test_or_train == "test"):
                y = self.mean_dice_test
            else:
                y = self.mean_dice_train
            print("y ", y)
            x = np.arange(0, self.epochs);
            print("x : ", x)
            plt.plot(x, y)
            plt.savefig(os.path.join(self.save_dir, "mean_dice_"+str(test_or_train)+".png"))
            plt.close()

    def calculate_accuracy(self, x, y, epoch, test_or_train):

        num_batches = x.shape[0] // self.size_batch
        #la shape non è più y.shape[0] perchè ricordo che l'ultimo batch è da 6, non da 10
        #quindi se facessi fino a y.shape[0] gli ultimi 6 numeri sarebbero ancora da np.empty
        #cioè sarebbero numeri a caso. Devo risolvere questa questione, ma per ora risolvo così
        y_true=np.empty(shape=[num_batches*self.size_batch], dtype=np.int16)
        y_pred=np.empty(shape=[num_batches*self.size_batch], dtype=np.int16)

        #epoch_accuracy = 0
        accuracy_list= np.zeros(shape=[num_batches], dtype=np.float32)
        cost_list = np.zeros(shape=[num_batches], dtype=np.float32)

        for ind_batch in range(num_batches):
            index_0=ind_batch * self.size_batch
            index_1=(ind_batch + 1) * self.size_batch
            batch_images = x[index_0:index_1]
            batch_labels = y[index_0:index_1][:,0]

            # potrei avere una batch eccezionalmente piu corta qui
            # current_batch_size = (ind_batch + 1) * self.size_batch - ind_batch * self.size_batch
            batch_images = np.array(batch_images).astype(np.float32)
            batch_labels = oneHot(batch_labels)
            y_true[index_0:index_1] = np.argmax(batch_labels,1)
            accuracy, classified = self.sess.run([self.accuracy,self.classified_sigmoid], feed_dict={self.x: batch_images, self.y: batch_labels, self.dropout_keep_prob:1})
            # cost, summary = self.sess.run([self.classify_cost, self.loss_classifier_summary],
            #                           feed_dict={self.x: batch_images, self.y: batch_labels, self.class_balancing_weights: tf.ones([self.size_batch]),
            #                                      self.dropout_keep_prob: 1})
            # cost_list.append(cost)
            # if (test_or_train=="test"):
            #     self.test_writer.add_summary(summary, epoch)
            # else:
            #     self.train_writer.add_summary(summary, epoch)

            y_pred[index_0:index_1] = classified
            accuracy_list[ind_batch]=accuracy

        ("DONE WITH CALCULATE_ACCURACY. HERE ARE THE RESULTS")
        print("y_true shape: ", y_true.shape)
        print("y_pred shape: ", y_pred.shape)
        print("y_true: ", y_true)
        print("y_pred: ", y_pred)

        print(accuracy_list)
        mean = float(np.mean(accuracy_list))
        print("mean accuracy", mean)

        if (test_or_train=="test"):
            self.array_of_accuracy_test.append(mean)
        else:
            self.array_of_accuracy_train.append(mean)

        #if epoch==(self.classifier_epochs-1):
        print("confusion matrix")
        cm= confusion_matrix(y_true, y_pred)
        print(cm)
        dir=os.path.join(self.save_dir,"supervised_finetuning", test_or_train)
        if not os.path.exists(dir):
            os.makedirs(dir)
            print("CREATED: ", dir)

        #plot_confusion_matrix(confusion_matrix, classes=["NC", "MCI", "AD"],  save_dir= self.save_dir, title='confusion_matrix_' + str(test_or_train)+".png")
        plot_confusion_matrix(cm, save_dir=dir, title=str(epoch)+'_confusion_matrix.png')

        print("READY TO PLOT")
        if (test_or_train=="test"):
            y = self.array_of_accuracy_test
        else:
            y=self.array_of_accuracy_train
        print("y ", y)
        x = np.arange(0, len(y));
        print("x : ", x)
        plt.plot(x,y)
        plt.ylim(0,1)
        plt.savefig(os.path.join(dir, "accuracy_plot.png"))
        plt.close()

    def encode_entire_dataset(self, x, y):

        num_batches = x.shape[0] // self.size_batch
        latents = np.zeros([x.shape[0], self.n_z])
        print("LATENTS STILL 0 SHAPE: ", latents.shape)

        for ind_batch in range(num_batches):
            # non ci vuole controllo per il limite dell'array! Niente index out of bound exception! Che figata!!!
            batch_images = x[ind_batch * self.size_batch:(ind_batch + 1) * self.size_batch]
            batch_labels = y[ind_batch * self.size_batch:(ind_batch + 1) * self.size_batch]

            # potrei avere una batch eccezionalmente piu corta qui
            # current_batch_size = (ind_batch + 1) * self.size_batch - ind_batch * self.size_batch
            batch_images = np.array(batch_images).astype(np.float32)
            z, _ = self.sess.run(
                [self.z_mean, self.z_log_sigma_sq],
                feed_dict={
                    self.x: batch_images,
                    self.dropout_keep_prob:1
                }
            )
            if (ind_batch == 0):
                latents = z
            else:
                latents = np.append(latents, z, axis=0)
        print("DONE WITH ENCODING! SHAPE:", latents.shape)
        return latents

    def dice_score(self, original_image, reconstructed_image):
        original_flattened=original_image.flatten()
        reconstructed_flattened=reconstructed_image.flatten()

        dice= 2*np.sum(np.multiply(original_flattened, reconstructed_flattened)) / (np.sum(original_flattened)+np.sum(reconstructed_flattened))
        print("dice: " + str(dice) +" of type: "+str(type(dice)))
        dice_format = float("%.2f" % dice)
        #print("dice_format: " + str(dice_format) + " of type: " + str(type(dice_format)))
        return dice_format

    def plotting_3d(self, image, epoch, ax, title="", reference_image=None):
        fontdict= {'fontsize': 30}

        image = np.reshape(image, newshape=[image.shape[0], image.shape[1], image.shape[2]])
        x,y,z=image.nonzero() #I use this only for the next print
        print("before round. Nonzero: "+str(x.shape)+" Min: " + str(np.min(image))+ ", Max: "+str(np.max(image)))
        image=np.round(image)

        print("after round. Min: " + str(np.min(image))+ ", Max: "+str(np.max(image)))

        x, y, z = image.nonzero()

        for i in zip(x,y,z):
            print(i)
        print("VOGLIO VEDERE LA LUNGHEZZA: ", len(x))

        ##OLD CODE USED TO PLOT IN THE SAME IMAGE BOTH THE RECONSTRUCTED AND THE ORIGINAL IMAGE
        # if reference_image is not None:
        #     reference_image=np.reshape(reference_image,newshape=[image.shape[0], image.shape[1], image.shape[2]])
        #     x_1, y_1, z_1 =reference_image.nonzero()
        #     print("VOGLIO VEDERE LA LUNGHEZZA REFERENCE: ", len(x_1))
        #     for i in zip(x_1, y_1,z_1):
        #         print(i)
        #
        #     ax.set_title(title + ": #nonzero is " + str(x.shape), fontdict=fontdict)
        #     total_points = image.shape[0] * image.shape[1] * image.shape[2]
        #     print("\n\n#nonzero elements: " + str(x.shape) + "/" + str(total_points) + "\n\n")
        #     print("image shape: ", image.shape)
        #
        #     # plt.scatter(range(0,128), range(0,128), range(0,128), )
        #     ax.scatter(x_1, y_1, z_1, s=500, c="r", marker=".")
        #     ax.scatter(x, y, z, s=500, c="b", marker=".")
        #     return

        if (title!=""):
            ax.set_title(title+": #nonzero is "+str(x.shape), fontdict=fontdict)
        total_points=image.shape[0]*image.shape[1]*image.shape[2]
        print("\n\n#nonzero elements: "+str(x.shape)+"/"+ str(total_points)+"\n\n")
        print("image shape: ",image.shape)

        # plt.scatter(range(0,128), range(0,128), range(0,128), )
        if (reference_image is None): #se è l'originale la plotto rossa
            ax.scatter(x, y, z, s=1000, c="r", marker=".")
        else: #altrimenti la plotto blu
            ax.scatter(x, y, z, s=1000, c="b", marker=".")

        #ax.plot(x,y,z) #sharp lines

    def plot_different_views(self, fig, test_image, epoch, title, reference_image=None, start=1):
        if (reference_image is None):
            ax = fig.add_subplot(2,4,start+0, projection='3d')
            self.plotting_3d(test_image, epoch=epoch, ax=ax, title=title)

            ax = fig.add_subplot(2,4, start+1, projection='3d')
            self.plotting_3d(test_image, epoch=epoch, ax=ax)
            ax.view_init(30, 90)

            ax = fig.add_subplot(2,4, start+2, projection='3d')
            self.plotting_3d(test_image, epoch=epoch, ax=ax)
            ax.view_init(30, 180)

            ax = fig.add_subplot(2,4,start+3, projection='3d')
            self.plotting_3d(test_image, epoch=epoch, ax=ax)
            ax.view_init(30, 270)
        else:
            dice=self.dice_score(reference_image, test_image)
            ax = fig.add_subplot(2,4, start + 0, projection='3d')
            self.plotting_3d(test_image, epoch=epoch, ax=ax, title=title+" dice="+str(dice), reference_image=reference_image)

            ax = fig.add_subplot(2,4, start + 1, projection='3d')
            self.plotting_3d(test_image, epoch=epoch, ax=ax, reference_image=reference_image)
            ax.view_init(30, 90)

            ax = fig.add_subplot(2,4, start + 2, projection='3d')
            self.plotting_3d(test_image, epoch=epoch, ax=ax, reference_image=reference_image)
            ax.view_init(30, 180)

            ax = fig.add_subplot(2,4, start + 3, projection='3d')
            self.plotting_3d(test_image, epoch=epoch, ax=ax, reference_image=reference_image)
            ax.view_init(30, 270)

    def test_plot(self, epoch):
        self.calculate_dice_loss(self.x_test, self.y_test, epoch, "test")
        self.calculate_dice_loss(self.x_train, self.y_train, epoch, "train")

        orig=True
        test_index=np.random.randint(0, self.x_test.shape[0]-self.size_batch)
        fig = plt.figure(figsize=[40,20])
        test_image = self.x_test[test_index]
        self.plot_different_views(fig, test_image, epoch=epoch, title="original")

        print("test shape: ", self.x_test.shape)
        x_test=self.x_test[test_index:test_index+self.size_batch]
        print("x test shape: ", x_test.shape)
        #questo è chiaramente sbagliato!! test_index è sempre il primo nel batch che pesco!!
        #potrebbe essere la causa degli errori precedenti (nei plotting dell'accuratezza per batch)
        #x_reconstruct = self.reconstruct(x_test)[np.mod(test_index, self.size_batch)]
        x_reconstruct = self.reconstruct(x_test, dropout_keep_prob=1)[0]
        np.save(os.path.join(self.save_dir,"reconstruct.npy"), x_reconstruct)
        #ax = fig.add_subplot(122, projection='3d')
        self.plot_different_views(fig, x_reconstruct, epoch, title="reconstruct", reference_image=test_image, start=5)

        plt.savefig(os.path.join(self.save_dir, str(epoch) + ".png"))

        plt.close()

    def visualise_supervised_finetuning(self):
        supervised_finetuning_path_test = self.save_dir + "\\supervised_finetuning\\test\\"
        if not os.path.exists(supervised_finetuning_path_test):
            os.makedirs(supervised_finetuning_path_test)
            print("CREATED: ", supervised_finetuning_path_test)

        self.visualiser.directory = supervised_finetuning_path_test
        print("SUPERVISED FINETUNING PEARSON CORRELATION TEST")
        #self.visualiser.ScatterOfLatentsssssssssssSpace(dataset_images=self.x_test, dataset_encoded_images=self.encode_entire_dataset(self.x_test, self.y_test), dataset_labels=self.y_test)
        #self.different_perplexities(dir=supervised_finetuning_path_test + "perplexity_3_", dataset_images=self.x_test, dataset_labels=self.y_test, tsne_perplexity=3)
        #self.different_perplexities(dir=supervised_finetuning_path_test + "perplexity_5_", dataset_images=self.x_test,  dataset_labels=self.y_test, tsne_perplexity=5)
        #self.different_perplexities(dir=supervised_finetuning_path_test + "perplexity_10_", dataset_images=self.x_test, dataset_labels=self.y_test, tsne_perplexity=10)
        #self.different_perplexities(dir=supervised_finetuning_path_test + "perplexity_20_", dataset_images=self.x_test, dataset_labels=self.y_test, tsne_perplexity=20)
        #self.different_perplexities(dir=supervised_finetuning_path_test + "perplexity_30_", dataset_images=self.x_test, dataset_labels=self.y_test, tsne_perplexity=30)
        self.different_perplexities(dir=supervised_finetuning_path_test + "perplexity_40_", dataset_images=self.x_test, dataset_labels=self.y_test, tsne_perplexity=40)
        self.different_perplexities(dir=supervised_finetuning_path_test + "perplexity_50_", dataset_images=self.x_test, dataset_labels=self.y_test, tsne_perplexity=50)
        self.different_perplexities(dir=supervised_finetuning_path_test + "perplexity_60_", dataset_images=self.x_test, dataset_labels=self.y_test, tsne_perplexity=60)

        supervised_finetuning_path_train = self.save_dir + "\\supervised_finetuning\\train\\"
        if not os.path.exists(supervised_finetuning_path_train):
            os.makedirs(supervised_finetuning_path_train)
            print("CREATED: ", supervised_finetuning_path_train)

        #self.visualiser.directory = supervised_finetuning_path_train
        #self.visualiser.ScatterOfLatentSpace(dataset_images=self.x_train, dataset_encoded_images=self.encode_entire_dataset(self.x_train,self.y_train), dataset_labels=self.y_train)
        # self.different_perplexities(dir=supervised_finetuning_path_train + "perplexity_3_", dataset_images=self.x_train,  dataset_labels=self.y_train, tsne_perplexity=3)
        # self.different_perplexities(dir=supervised_finetuning_path_train + "perplexity_5_", dataset_images=self.x_train,  dataset_labels=self.y_train, tsne_perplexity=5)
        # self.different_perplexities(dir=supervised_finetuning_path_train + "perplexity_10_", dataset_images=self.x_train, dataset_labels=self.y_train, tsne_perplexity=10)
        # self.different_perplexities(dir=supervised_finetuning_path_train + "perplexity_20_", dataset_images=self.x_train, dataset_labels=self.y_train, tsne_perplexity=20)
        # self.different_perplexities(dir=supervised_finetuning_path_train + "perplexity_30_", dataset_images=self.x_train, dataset_labels=self.y_train, tsne_perplexity=30)
        #self.different_perplexities(dir=supervised_finetuning_path_train + "perplexity_40_", dataset_images=self.x_train, dataset_labels=self.y_train, tsne_perplexity=40)
        #self.different_perplexities(dir=supervised_finetuning_path_train + "perplexity_50_", dataset_images=self.x_train, dataset_labels=self.y_train, tsne_perplexity=50)
        #self.different_perplexities(dir=supervised_finetuning_path_train + "perplexity_60_", dataset_images=self.x_train, dataset_labels=self.y_train, tsne_perplexity=60)

    def different_perplexities(self, dir, dataset_images, dataset_labels, tsne_perplexity):
        self.visualiser.directory = dir
        self.visualiser.ScatterOfLatentSpace(dataset_images=dataset_images,dataset_encoded_images=self.encode_entire_dataset(dataset_images,dataset_labels), dataset_labels=dataset_labels)
        self.visualiser.ScatterOfLatentSpace(dataset_images=dataset_images, dataset_encoded_images=self.encode_entire_dataset(dataset_images, dataset_labels),dataset_labels=dataset_labels, n_components=3)
        # self.visualiser.computeTSNEProjectionOfLatentSpace(dataset_images=self.x_test, dataset_encoded_images=self.encode_entire_dataset(self.x_test, self.y_test), dataset_labels=self.y_test, n_components=3, _3d=True)
        # self.visualiser.computeTSNEProjectionOfPixelSpace(dataset_images=x_test, dataset_labels=y_test, n_components=3,_3d=True)
        # self.visualiser.computeTSNEProjectionOfLatentSpace(dataset_images=x_test, dataset_labels=y_test, n_components=2, _3d=True)

    def visualise(self, epoch):
        print("637 VISUALISING")
        test_path=self.save_dir + "\\latent_space_test_set\\"
        if not os.path.exists(test_path):
            os.makedirs(test_path)
            #print("CREATED: ", test_path)

        train_path=self.save_dir+"\\latent_space_train_set\\"
        if not os.path.exists(train_path):
            os.makedirs(train_path)
            #print("CREATED: ", train_path)

        progression_path = self.save_dir + "\\progression\\"
        if not os.path.exists(progression_path):
            os.makedirs(progression_path)
            #print("CREATED: ", progression_path)

        #writing in the test path directory
        self.visualiser.directory = test_path
        self.different_perplexities(dir=test_path + "perplexity_3_", dataset_images=self.x_test, dataset_labels=self.y_test, tsne_perplexity=3)
        self.different_perplexities(dir=test_path + "perplexity_5_", dataset_images=self.x_test, dataset_labels=self.y_test, tsne_perplexity=5)
        self.different_perplexities(dir=test_path + "perplexity_10_", dataset_images=self.x_test, dataset_labels=self.y_test, tsne_perplexity=10)
        self.different_perplexities(dir=test_path + "perplexity_20_", dataset_images=self.x_test, dataset_labels=self.y_test, tsne_perplexity=20)
        self.different_perplexities(dir=test_path + "perplexity_30_", dataset_images=self.x_test, dataset_labels=self.y_test, tsne_perplexity=30)
        self.different_perplexities(dir=test_path + "perplexity_40_", dataset_images=self.x_test, dataset_labels=self.y_test, tsne_perplexity=40)
        self.different_perplexities(dir=test_path + "perplexity_50_", dataset_images=self.x_test, dataset_labels=self.y_test, tsne_perplexity=50)
        self.different_perplexities(dir=test_path + "perplexity_60_", dataset_images=self.x_test, dataset_labels=self.y_test, tsne_perplexity=60)

        # writing in the train path directory
        self.visualiser.directory = train_path
        self.different_perplexities(dir=train_path + "perplexity_3_", dataset_images=self.x_train, dataset_labels=self.y_train, tsne_perplexity=3)
        self.different_perplexities(dir=train_path + "perplexity_5_", dataset_images=self.x_train, dataset_labels=self.y_train, tsne_perplexity=5)
        self.different_perplexities(dir=train_path + "perplexity_10_", dataset_images=self.x_train, dataset_labels=self.y_train, tsne_perplexity=10)
        self.different_perplexities(dir=train_path + "perplexity_20_", dataset_images=self.x_train, dataset_labels=self.y_train, tsne_perplexity=20)
        self.different_perplexities(dir=train_path + "perplexity_30_", dataset_images=self.x_train, dataset_labels=self.y_train, tsne_perplexity=30)
        self.different_perplexities(dir=train_path + "perplexity_40_", dataset_images=self.x_train, dataset_labels=self.y_train, tsne_perplexity=40)
        self.different_perplexities(dir=train_path + "perplexity_50_", dataset_images=self.x_train, dataset_labels=self.y_train, tsne_perplexity=50)
        self.different_perplexities(dir=train_path + "perplexity_60_", dataset_images=self.x_train, dataset_labels=self.y_train, tsne_perplexity=60)

        self.visualiser.directory = progression_path
        self.visualiser.plot_progression(dataset_images=self.x_progression, dataset_encoded_images=self.encode_entire_dataset(self.x_progression, self.y_progression[:,0:6]), dataset_labels=self.y_progression, epoch=epoch)