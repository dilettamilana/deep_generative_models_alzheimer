from __future__ import division
from scipy.misc import imread, imresize, imsave
from scipy.ndimage.morphology import binary_dilation
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
import scipy.stats
from scipy.io import savemat
from ops import *
import training_notes_helper as tn
import sys
from os.path import abspath
import collections
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import smtplib
import email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import visualising_helper as visualising

class BrainAging(object):
    def __init__(self,
                 session,  # TensorFlow session
                 epochs,
                 learning_rate,  # learning rate of optimizer
                 G_img_param,
                 E_z_param,
                 tv_param,
                 size_image=256,  # size the input images
                 size_kernel=5,  # size of the kernels in convolution and deconvolution
                 size_batch=10,  # mini-batch size for training and testing, must be square of an integer
                 num_input_channels=1,  # number of channels of input images
                 num_encoder_channels=32,  # number of channels of the first conv layer of encoder
                 num_z_channels=50,  # number of channels of the layer z (noise or code)
                 num_categories=10,  # number of categories (age segments) in the training dataset
                 num_gen_channels=32,  # number of channels of the first deconv layer of generator
                 enable_tile_label=True,  # enable to tile the label
                 tile_ratio=1.0,  # ratio of the length between tiled label and z
                 is_training=True,  # flag for training or testing mode
                 ):
        self.session = session
        self.epochs=epochs
        self.learning_rate=learning_rate
        self.G_img_param = G_img_param
        self.E_z_param= E_z_param
        self.tv_param=tv_param
        self.gradients_dilation = False

        self.image_value_range = (0, 1)
        self.size_image = size_image
        self.size_kernel = size_kernel
        self.size_batch = size_batch
        self.num_input_channels = num_input_channels
        self.num_encoder_channels = num_encoder_channels
        self.num_z_channels = num_z_channels
        self.num_categories = num_categories
        self.num_gen_channels = num_gen_channels
        self.enable_tile_label = enable_tile_label
        self.tile_ratio = tile_ratio
        self.is_training = is_training
        self.save_dir = './runs/'+time.strftime("%m%d_%H%M")
        self.disc_updates=1
        self.gen_updates=1
        self.send_email_report_frequency=4
        self.num_age_progression_tests=3
        self.visualise_custom_test_frequency=10
        self.visualise_latent_frequency=10

        self.mean_dice_test = []
        self.mean_dice_train = []
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        print("Let's start!")

        # ************************************* input and data ********************************************************
        self.define_inputs()

        if (self.gradients_dilation):
            self.w_train=apply_gradients_dilation(np.reshape(self.x_train, newshape=[self.x_train.shape[0], self.img_x, self.img_y, self.img_z,1]))
            self.w_test=apply_gradients_dilation(np.reshape(self.x_test, newshape=[self.x_test.shape[0], self.img_x, self.img_y, self.img_z,1]))

            self.current_weights_placeholder = tf.placeholder(tf.float32, shape=[self.size_batch, self.x_train.shape[1],
                                                                             self.x_train.shape[2],
                                                                             self.x_train.shape[3], 1])
            self.current_weights_var = tf.get_variable('current_cost_weights',
                                                   shape=[self.size_batch, self.x_train.shape[1], self.x_train.shape[2],
                                                          self.x_train.shape[3], 1])

        self.input_image = tf.placeholder(
            tf.float32,
            [self.size_batch, self.img_x, self.img_y, self.img_z, 1],
            name='input_images'
        )
        self.age = tf.placeholder(
            tf.float32,
            [None, self.num_categories],
            name='age_labels'
        )
        self.classes = tf.placeholder(
            tf.float32,
            [None, 3],
            name='class_labels'
        )
        # ************************************* build the graph *******************************************************
        # set stdout:
        model_architecture = os.path.join(self.save_dir, 'model_architecture.txt')
        sys.stdout = open(model_architecture, 'a+')
        print('\n\tBuilding graph ...')

        # encoder: input image --> z
        self.z = self.encoder(
            image=self.input_image
        )

        # generator: z + label --> generated image
        self.G = self.generator(
            z=self.z,
            y=self.age,
            classes=self.classes,
            enable_tile_label=self.enable_tile_label,
            tile_ratio=self.tile_ratio
        )


        # discriminator on G #sometimes I send the FAKE image as INPUT
        self.D_G, self.D_G_logits = self.discriminator_img(
            image=self.G,
            y=self.age,
            classes=self.classes,
            is_training=self.is_training
        )

        # discriminator on input image: sometimes I send the ORIGINAL image as INPUT
        self.D_input, self.D_input_logits = self.discriminator_img(
            image=self.input_image,
            y=self.age,
            classes=self.classes,
            is_training=self.is_training,
            reuse_variables=True
        )
        total_trainable_params = self.total_trainable_params()

        # ************************************* loss functions *******************************************************
        # loss function of encoder + generator
        # tf.nn.l2_loss( penalising_factor*(binary_map+1) * (input_image-G) )
        if self.gradients_dilation:
            self.EG_loss = tf.nn.l2_loss(tf.multiply(self.input_image - self.G,
                                    (tf.add(tf.ones([self.size_batch, self.x_train.shape[1],
                                                     self.x_train.shape[2], self.x_train.shape[3], 1]),
                                            self.current_weights_var))))
        else:
            self.EG_loss = tf.nn.l2_loss(self.input_image - self.G)*10

  # L2 loss
        #self.EG_loss = tf.reduce_mean(tf.abs(self.input_image - self.G))  # L1 loss
        print("147", str(self.EG_loss.get_shape()))

        #### LOSS FUNCTION ON DISCRIMINATOR IMAGE #### keep these three
        # Ha a che fare con fake vs real images

        # Tutte queste loss vengono minimizzata.
        # io voglio che il discriminator sia 0 quando gli do in input un'immagine generata. Quindi:
        # minimizzo la distanza tra immagine vera e 1 e
        # loss function of discriminator on image
        self.D_img_loss_input = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_input_logits, tf.ones_like(self.D_input_logits))
        )
        print("185", str(self.D_img_loss_input.get_shape()))

        #  minimizzo la distanza tra 0 e immagine finta
        self.D_img_loss_G = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_G_logits, tf.zeros_like(self.D_G_logits))
        )
        print("191", str(self.D_img_loss_G.get_shape()))

        # ovviamente per il generatore vale l'opposto di quella appena sopra: lui cerca di minimizzare
        # la distanza tra il suo risultato (immagine finta) e 1 (vuol dire che al discriminatore sembrerà vera)
        self.G_img_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_G_logits, tf.ones_like(self.D_G_logits))
        )

        # *********************************** trainable variables ****************************************************
        trainable_variables = tf.trainable_variables()
        # variables of encoder
        self.E_variables = [var for var in trainable_variables if 'E_' in var.name]
        # variables of generator
        self.G_variables = [var for var in trainable_variables if 'G_' in var.name]
        # variables of discriminator on image
        self.D_img_variables = [var for var in trainable_variables if 'D_img_' in var.name]

        # ************************************* collect the summary ***************************************
        self.z_summary = tf.summary.histogram('z', self.z)
        self.EG_loss_summary = tf.summary.scalar('EG_loss', self.EG_loss)
        self.D_img_loss_input_summary = tf.summary.scalar('D_img_loss_input', self.D_img_loss_input)
        self.D_img_loss_G_summary = tf.summary.scalar('D_img_loss_G', self.D_img_loss_G)
        self.G_img_loss_summary = tf.summary.scalar('G_img_loss', self.G_img_loss)
        self.D_G_logits_summary = tf.summary.histogram('D_G_logits', self.D_G_logits)
        self.D_input_logits_summary = tf.summary.histogram('D_input_logits', self.D_input_logits)
        # for saving the graph and variables
        self.saver = tf.train.Saver(max_to_keep=10)
        #model_architecture.close()

    def define_inputs(self):
        ######  DATASET   #####
        self.dataset_list = ["Harp", "OASIS", "AIBL"]
        self.nc_only = False
        self.max_images = 1000

        ######  IMAGES   #####
        self.image_info="orig_patch26_32x32x64"
        #self.image_info = "orig_patch16"
        self.conv3d = True
        # 2 coronal
        # 1 sagittal
        # 0 axial
        self.selected_slice = None

        ######  LABELS   #####
        self.label_info="patch26"
        #self.label_info = "patch16"
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

        self.x_train, self.y_train, self.x_test, self.y_test = import_mri_dataset(self.dataset_list, self.image_info, self.label_info,
                                                                     self.selected_slice, self.oneHot, self.allFields, self.conv3d,
                                                                     self.max_images, self.nc_only)

        self.size_data = self.x_train.shape[0] + self.x_test.shape[0]
        self.img_x = self.x_train.shape[1]
        self.img_y = self.x_train.shape[2]
        self.img_z = self.x_train.shape[3]
        self.visualiser = visualising.Visualiser(self, _3d=False, n_z=self.num_z_channels, size_batch=self.size_batch, save_dir=self.save_dir, img_x=self.size_image, img_y=self.size_image, img_z=1, x_train=self.x_train, x_test=self.x_test, y_train=self.y_train, y_test=self.y_test)

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

    def train(self,
              beta1=0.5,  # parameter for Adam optimizer
              decay_rate=1,  # learning rate decay (0, 1], 1 means no decay
              enable_shuffle=True,  # enable shuffle of the dataset
              use_trained_model=True,  # used the saved checkpoint to initialize the model
              ):

        #set stdout:
        training_steps = os.path.join(self.save_dir, 'epochs.txt')
        sys.stdout = open(training_steps, 'a+')

        # *********************************** optimizer **************************************************************

        # over all, there are three loss functions, weights may differ from the paper because of different datasets
        #self.loss_EG = self.EG_loss + self.G_img_param * self.G_img_loss + self.E_z_param * self.E_z_loss + self.tv_param * self.tv_loss # slightly increase the params
        self.loss_EG = self.EG_loss + self.G_img_loss
        self.loss_Di = self.D_img_loss_input + self.D_img_loss_G

        #self.loss_Di = tf.reduce_mean(logits_real - logits_fake)
        #self.loss_EG = tf.reduce_mean(logits_fake)

        # set learning rate decay
        self.EG_global_step = tf.Variable(0, trainable=False, name='global_step')
        EG_learning_rate = tf.train.exponential_decay(
            learning_rate=self.learning_rate,
            global_step=self.EG_global_step,
            decay_steps=self.size_data / self.size_batch * 2,
            decay_rate=decay_rate,
            staircase=True
        )

        # optimizer for encoder + generator
        self.EG_optimizer = tf.train.AdamOptimizer(
            learning_rate=EG_learning_rate,
            #beta1=beta1 only adam
        ).minimize(
            loss=self.loss_EG,
            global_step=self.EG_global_step,
            var_list=self.E_variables + self.G_variables
        )

        # optimizer for discriminator on image
        self.D_img_optimizer = tf.train.AdamOptimizer(
            learning_rate=EG_learning_rate,
            #beta1=beta1 only adam
        ).minimize(
            loss=self.loss_Di,
            var_list=self.D_img_variables
        )

        # *********************************** training notes *********************************************************
        training_notes = "reuniting the losses, didn't work out before"
        def save_training_notes():
            notes = training_notes

            #print(abspath(getsourcefile(lambda: 0)))
            dict = collections.OrderedDict([('notes', notes),
                                            ('file name', os.path.realpath(__file__)),
                                            ('dataset list', self.dataset_list),
                                            ('load conv_3d data', self.conv3d),
                                            ('load only nc data', self.nc_only),
                                            ('image_info', self.image_info),
                                            ('label_info', self.label_info),
                                            ('epochs', self.epochs),
                                            ('batch_size', self.size_batch),
                                            ('n_samples', self.max_images),
                                            ('train_set_size', self.x_train.shape[0]),
                                            ('test_set_size', self.x_test.shape[0]),
                                            ('learning_rate', self.learning_rate),
                                            ('beta1', beta1),
                                            ('decay_rate', decay_rate),
                                            ('size_image', self.size_image),
                                            ('size_kernel', self.size_kernel),
                                            ('num_input_channels', self.num_input_channels),
                                            ('num_encoder_channels', self.num_encoder_channels),
                                            ('num_z_channels', self.num_z_channels),
                                            ('num_gen_channels', self.num_gen_channels),
                                            ('enable_tile_label', self.enable_tile_label),
                                            ('tile_ratio', self.tile_ratio),
                                            ('selected_slice', self.selected_slice),
                                            ('G_img_param', self.G_img_param),
                                            ('E_z_param', self.E_z_param),
                                            ('tv_param', self.tv_param),
                                            ('discriminator updates per every generator update', self.disc_updates)
            ])
            tn.write_training_notes(self.save_dir, dict)

        save_training_notes()

        # *********************************** tensorboard *************************************************************
        # for visualization (TensorBoard): $ tensorboard --logdir path/to/log-directory
        self.EG_learning_rate_summary = tf.summary.scalar('EG_learning_rate', EG_learning_rate)
        self.summary = tf.summary.merge([
            self.z_summary,
            self.EG_loss_summary,
            self.D_img_loss_input_summary, self.D_img_loss_G_summary,
            self.G_img_loss_summary, self.EG_learning_rate_summary,
            self.D_G_logits_summary, self.D_input_logits_summary
        ])
        self.writer = tf.summary.FileWriter(os.path.join(self.save_dir, 'summary'), self.session.graph)

        # ************* get some random samples as testing data to visualize the learning process *********************

        sample_images, sample_label_age, sample_label_classes = self.sample_batches()
        # ******************************************* training *******************************************************
        print('\n\tPreparing for training ...')

        # initialize the graph
        tf.global_variables_initializer().run()

        # load check point
        if use_trained_model:
            if self.load_checkpoint():
                print("\tSUCCESS ^_^")
            else:
                print("\tFAILED >_<!")

        # epoch iteration
        num_batches = self.x_train.shape[0] // self.size_batch
        for epoch in range(self.epochs):
            # if enable_shuffle:
            #     shuffle(x_train, y_train)
            start_time = time.time()
            print("Epoch: {}/{}".format(epoch+1, self.epochs))
            for ind_batch in range(num_batches):
                print("Batch: {}/{}".format(ind_batch, num_batches))
                # non ci vuole controllo per il limite dell'array! Niente index out of bound exception! Che figata!!!
                batch_images = self.x_train[ind_batch * self.size_batch:(ind_batch + 1) * self.size_batch]
                batch_labels = self.y_train[ind_batch * self.size_batch:(ind_batch + 1) * self.size_batch]

                # potrei avere una batch eccezionalmente piu corta qui
                current_batch_size = (ind_batch + 1) * self.size_batch - ind_batch * self.size_batch

                if self.num_input_channels == 1:
                    batch_images = np.array(batch_images).astype(np.float32)[:, :, :, : , None]
                else:
                    batch_images = np.array(batch_images).astype(np.float32)
                batch_label_age = np.ones(
                    shape=(current_batch_size, self.num_categories),
                    dtype=np.float
                ) * self.image_value_range[0] #cioe' -1 o 0, sto inizializzando

                batch_label_classes = np.ones(
                    shape=(current_batch_size, 3),
                    dtype=np.float
                ) * self.image_value_range[0] #cioe' -1 o 0, sto inizializzando

                # age
                for i, label in enumerate(batch_labels):
                    # label = int(str(batch_files[i]).split('/')[-1].split('_')[0])
                    label = batch_labels[i, 3]
                    if 31 <= label <= 50:
                        label = 0
                    elif 51 <= label <= 60:
                        label = 1
                    elif 61 <= label <= 65:
                        label = 2
                    elif 66 <= label <= 70:
                        label = 3
                    elif 71 <= label <= 75:
                        label = 4
                    elif 76 <= label <= 80:
                        label = 5
                    elif 81 <= label <= 85:
                        label = 6
                    elif 86 <= label <= 90:
                        label = 7
                    elif 91 <= label <= 100:
                        label = 8
                    else:
                        label = 9

                    # tra tutti i -1 dell'inizializzazione, ora assegno 1 solo a quelli corrispondenti al vero label
                    batch_label_age[i, label] = self.image_value_range[-1]  # ultimo elemento, cioe' 1
                    classes = int(batch_labels[i, 0])
                    batch_label_classes[i, classes] = self.image_value_range[-1]  # ultimo elemento cioe' 1

                if self.gradients_dilation:
                    assign_op = self.current_weights_var.assign(self.current_weights_placeholder)
                    weights = self.w_train[ind_batch * self.size_batch:(ind_batch + 1) * self.size_batch]
                    unique, counts = np.unique(weights, return_counts=True)
                    print(dict(zip(unique, counts)))
                    self.session.run(assign_op, feed_dict={self.current_weights_placeholder: weights})

                updating_disc=False
                updating_gen=False
                #update the generator only when mod=0
                if (np.mod(epoch, self.gen_updates)==0):
                    updating_gen=True
                    _, EG_err, Gi_err, DiG_err, Di_err = self.session.run(
                        fetches = [
                            self.EG_optimizer,
                            self.EG_loss,
                            self.G_img_loss,
                            self.D_img_loss_G,
                            self.D_img_loss_input,
                        ],
                        feed_dict={
                            self.input_image: batch_images,
                            self.age: batch_label_age,
                            self.classes: batch_label_classes,
                        }
                    )

                if (np.mod(epoch, self.disc_updates)==0):
                    #update disc every other time
                    updating_disc=True
                    _, EG_err,  Gi_err, DiG_err, Di_err= self.session.run(
                        fetches=[
                            self.D_img_optimizer,
                            self.EG_loss,
                            self.G_img_loss,
                            self.D_img_loss_G,
                            self.D_img_loss_input,
                        ],
                        feed_dict={
                            self.input_image: batch_images,
                            self.age: batch_label_age,
                            self.classes: batch_label_classes
                        }
                    )
                if (updating_disc and updating_gen):
                    turn="Discriminator and Generator"
                elif (updating_disc):
                    turn="Discriminator"
                else:
                    turn="Generator"

            # add to summary
            summary = self.summary.eval(
                feed_dict={
                    self.input_image: batch_images,
                    self.age: batch_label_age,
                    self.classes: batch_label_classes
                }
            )
            self.writer.add_summary(summary, self.EG_global_step.eval())
            print("\nUpdating [%s] | Epoch [%3d/%3d]\n\tEG_err=%.4f" %
                  (turn, epoch + 1, self.epochs, EG_err))
            print("\tGi=%.4f\tDi=%.4f\tDiG=%.4f" % (Gi_err, Di_err, DiG_err))

            name = '{:02d}'.format(epoch + 1)
            if (epoch%1==0):
                if (epoch==0):
                    np.save(os.path.join(self.save_dir, "labels_train.npy"), self.y_train)
                    np.save(os.path.join(self.save_dir, "labels_test.npy"), self.y_test)

                    self.sample(sample_images, sample_label_age, sample_label_classes, name, epoch, original=True)
                else:
                    self.sample(sample_images, sample_label_age, sample_label_classes, name, epoch)

                #for test_num in range(self.num_age_progression_tests): non ha senso, tanto prima le sovrascrivvo quindi non le vedevo comunque e ora uso il t-test
                self.test_and_t_test(name="test_"+str(name), epoch=epoch)

            # save checkpoint for each 10 epoch
            if ((np.mod(epoch, self.epochs-1) == 0 and epoch>0) or np.mod(epoch, 5)):
                self.save_checkpoint()

            if (np.mod(epoch,self.visualise_custom_test_frequency)==0):
                 self.custom_test()

            # self.save_latent(to_save="train")
            # self.save_latent(to_save="test")
            # if (np.mod(epoch, self.visualise_latent_frequency) == 0):
            #      self.visualise(epoch)
            #
            elapse = time.time() - start_time
            time_left = (self.epochs - epoch - 1) * elapse
            print("\tTime left: %02d:%02d:%02d" %
                  (int(time_left / 3600), int(time_left % 3600 / 60), time_left % 60))

            if np.mod(epoch, int(self.epochs/self.send_email_report_frequency))==0 or epoch==(self.epochs-1) or epoch==0:
                self.send_email(epoch, time_left)

        self.writer.close()

    def sample_batches(self, x=None, y=None, start_index=None):

        if x is None or y is None:
            x=self.x_test
            y=self.y_test

        if start_index is None:
            start_index=np.random.randint(0, x.shape[0]-11)

        sample_images = x[start_index:start_index+self.size_batch]
        #print("SAMPLE IMAGES: ",sample_images.shape)
        sample_labels = y[start_index:start_index+self.size_batch]
        #print("SAMPLE LABELS: ",sample_labels.shape)

        if self.num_input_channels == 1:
            sample_images = np.array(sample_images).astype(np.float32)[:, :, :, :, None]
        else:
            sample_images = np.array(sample_images).astype(np.float32)

        print("PARA: ",sample_images.shape)
        # for i in range(sample_images.shape[0]):
        #     image=sample_images[i]
        #     imsave(os.path.join(self.save_dir, str(i)+".png"),(np.reshape(image, newshape=[image.shape[0],image.shape[1]])))

        sample_label_age = np.ones(
            shape=(self.size_batch, self.num_categories),
            dtype=np.float32
        ) * self.image_value_range[0]

        sample_label_classes = np.ones(
            shape=(self.size_batch, 3),
            dtype=np.float32
        ) * self.image_value_range[0]

        # age
        for i, label in enumerate(sample_labels):
            # label = int(str(batch_files[i]).split('/')[-1].split('_')[0])
            label = sample_labels[i, 3] #eta'
            if 31 <= label <= 50:
                label = 0
            elif 51 <= label <= 60:
                label = 1
            elif 61 <= label <= 65:
                label = 2
            elif 66 <= label <= 70:
                label = 3
            elif 71 <= label <= 75:
                label = 4
            elif 76 <= label <= 80:
                label = 5
            elif 81 <= label <= 85:
                label = 6
            elif 86 <= label <= 90:
                label = 7
            elif 91 <= label <= 100:
                label = 8
            else:
                label = 9

            # tra tutti i -1 dell'inizializzazione, ora assegno 1 solo a quelli corrispondenti al vero label
            sample_label_age[i, label] = self.image_value_range[-1]  # ultimo elemento, cioe' 1
            classes = int(sample_labels[i,0]) #0 e' l'indice a cui trovo la classificazione
            #print("\n\n\nCLASSES ", str(classes))
            sample_label_classes[i, classes] = self.image_value_range[-1]  # ultimo elemento cioe' 1
        return sample_images, sample_label_age, sample_label_classes

    def encoder(self, image, reuse_variables=False):
        print("\n\n\nENCODER\n")
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        min_side = min(self.img_x, self.img_y, self.img_z)
        num_layers = int(np.log2(min_side)) - int(self.size_kernel / 2)  # resta due perchè ad ogni layer raddoppio
        size_mini_map = int(min_side / 2 ** num_layers)
        current = image
        # conv layers with stride 2
        for i in range(num_layers):
            name = 'E_conv' + str(i)
            current = conv3d(
                    input_map=current,
                    num_output_channels=self.num_encoder_channels * (2 ** i),
                    size_kernel=self.size_kernel,
                    add_bias=False,
                    name=name,
                    batchnorm=False,
                )
            print(str(current.name) + " - " + str(current.get_shape()))
            current = tf.nn.relu(current)
            print(str(current.name) + " - " + str(current.get_shape()))

        # fully connection layer
        name = 'E_fc'
        current = fc(
            input_vector=tf.reshape(current, [self.size_batch, -1]),
            num_output_length=self.num_z_channels,
            name=name,
            batchnorm=False,
        )
        print(str(current.name) + " - " + str(current.get_shape()))
        self.z_before_tanh= current
        # output
        current = tf.nn.tanh(current)
        print(str(current.name) + " - " + str(current.get_shape()))
        return current

    def generator(self, z, y, classes, reuse_variables=False, enable_tile_label=True, tile_ratio=1.0):
        print("\n\n\nGENERATOR\n")
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        min_side = min(self.img_x, self.img_y, self.img_z)
        num_layers = int(np.log2(min_side)) - int(self.size_kernel / 2)  # resta due perchè ad ogni layer raddoppio
        size_mini_map = int(min_side / 2 ** num_layers)
        if enable_tile_label:
            duplicate = int(self.num_z_channels * tile_ratio / self.num_categories)
        else:
            duplicate = 1
        z = concat_label(z, y, duplicate=duplicate)
        if enable_tile_label:
            duplicate = int(self.num_z_channels * tile_ratio / 2)
        else:
            duplicate = 1
        z = concat_label(z, classes, duplicate=duplicate)
        # fc layer
        name = 'G_fc'
        current = fc(
            input_vector=z,
            num_output_length=self.num_gen_channels * size_mini_map * size_mini_map*size_mini_map,
            name=name,
            batchnorm=False,
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
            current = deconv3d(
                    input_map=current,
                    output_shape=[self.size_batch,
                                  size_mini_map * 2 ** (i + 1),
                                  size_mini_map * 2 ** (i + 1),
                                  size_mini_map * 2 ** (i + 1),
                                  int(self.num_gen_channels / 2 ** (i + 1))],
                    size_kernel=self.size_kernel,
                    add_bias=False,
                    name=name,
                    batchnorm=False
                )
            print(str(current.name) + " - " + str(current.get_shape()))
            current = tf.nn.relu(current)
            print(str(current.name) + " - " + str(current.get_shape()))
        # name = 'G_deconv' + str(i+1)
        # current = deconv3d(
        #     input_map=current,
        #     output_shape=[self.size_batch,
        #                   self.img_x,
        #                   self.img_y,
        #                   self.img_z,
        #                   int(self.num_gen_channels / 2 ** (i + 2))],
        #     size_kernel=self.size_kernel,
        #     stride=1,
        #     add_bias=False,
        #     name=name,
        #     batchnorm=False
        # )
        # print(str(current.name) + " - " + str(current.get_shape()))
        # current = tf.nn.relu(current)
        # print(str(current.name) + " - " + str(current.get_shape()))

        if self.labelId==16:
            complete_stride_for_last_layer = [1, 2, 2, 1, 1]
        elif self.labelId==26:
            complete_stride_for_last_layer = [1, 1, 1, 2, 1]
        elif self.labelId==19 or self.labelId==27:
            complete_stride_for_last_layer = [1, 1, 1, 1, 1]
        elif self.labelId==6  or self.labelId==19:
            complete_stride_for_last_layer = [1, 1, 1, 2, 1]

        name = 'G_deconv' + str(i + 1)

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
        current = tf.nn.relu(current)
        print(str(current.name) + " - " + str(current.get_shape()))
        return current

    def discriminator_img(self, image, y, classes, is_training=True, reuse_variables=False, num_hidden_layer_channels=(16, 32, 64, 128), enable_bn=False):
        print("\n\n\nDISCRIMINATOR_IMG\n")
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        num_layers = len(num_hidden_layer_channels)
        current = image
        # conv layers with stride 2
        for i in range(num_layers):
            name = 'D_img_conv' + str(i)
            current = conv3d(
                    input_map=current,
                    num_output_channels=num_hidden_layer_channels[i],
                    size_kernel=self.size_kernel,
                    add_bias=False,
                    name=name,
                    batchnorm=False
                )
            print(str(current.name) + " - " + str(current.get_shape()))
            current = tf.nn.relu(current)
            print(str(current.name) + " - " + str(current.get_shape()))
            if i == 0:
                current = concat_label(current, y)
                current = concat_label(current, classes, int(self.num_categories / 2))
        # fully connection layer
        name = 'D_img_fc1'
        current = fc(
            input_vector=tf.reshape(current, [self.size_batch, -1]),
            num_output_length=1024,
            name=name,
            batchnorm=False,
        )
        print(str(current.name) + " - " + str(current.get_shape()))
        current = lrelu(current)
        print(str(current.name) + " - " + str(current.get_shape()))
        name = 'D_img_fc2'
        current = fc(
            input_vector=current,
            num_output_length=1,
            name=name,
            batchnorm=False
        )
        # output
        print(str(current.name) + " - " + str(current.get_shape()))
        print(str(tf.nn.sigmoid(current).name) + " - " + str(tf.nn.sigmoid(current).get_shape()))
        return tf.nn.sigmoid(current), current

    def save_checkpoint(self):
        checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        for file in glob(os.path.join(checkpoint_dir, "*")):
            os.remove(file)

        self.saver.save(
            sess=self.session,
            save_path=os.path.join(checkpoint_dir, 'model'),
            global_step=self.EG_global_step.eval()
        )

    def load_checkpoint(self):
        print("\n\tLoading pre-trained model ...")
        checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')
        checkpoints = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoints and checkpoints.model_checkpoint_path:
            checkpoints_name = os.path.basename(checkpoints.model_checkpoint_path)
            self.saver.restore(self.session, os.path.join(checkpoint_dir, checkpoints_name))
            return True
        else:
            return False

    def sample(self, images, labels, classes, name, epoch, original=False):

        sample_dir = os.path.join(self.save_dir, 'samples')
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)

        z, G = self.session.run(
            [self.z, self.G],
            feed_dict={
                self.input_image: images,
                self.age: labels,
                self.classes: classes
            }
        )
        test_index = np.random.randint(0, images.shape[0])
        test_image = images[test_index]
        test_reconstruct=G[test_index]

        fig = plt.figure(figsize=[40, 20])
        self.plot_different_views(fig, test_image, epoch=epoch, title="original")
        self.plot_different_views(fig, test_reconstruct, epoch=epoch, title="reconstruct", reference_image=test_image, start=5)
        plt.savefig(os.path.join(sample_dir, str(epoch) + ".png"))
        plt.close()

    def plotting_3d(self, image, epoch, ax, title="", reference_image=None, color=None):
        fontdict= {'fontsize': 30}
        print("807 ", image.shape)
        image = np.reshape(image, newshape=[image.shape[0], image.shape[1], image.shape[2]])
        x,y,z=image.nonzero() #I use this only for the next print
        print("before round. Nonzero: "+str(x.shape)+" Min: " + str(np.min(image))+ ", Max: "+str(np.max(image)))
        image=np.round(image)

        print("after round. Min: " + str(np.min(image))+ ", Max: "+str(np.max(image)))

        x, y, z = image.nonzero()

        for i in zip(x,y,z):
            print(i)
        print("VOGLIO VEDERE LA LUNGHEZZA: ", len(x))
        if (title!=""):
            ax.set_title(title+": #nonzero is "+str(x.shape), fontdict=fontdict)
        total_points=image.shape[0]*image.shape[1]*image.shape[2]
        print("\n\n#nonzero elements: "+str(x.shape)+"/"+ str(total_points)+"\n\n")
        print("image shape: ",image.shape)

        # plt.scatter(range(0,128), range(0,128), range(0,128), )
        if (reference_image is None): #se è l'originale la plotto rossa
            if (color is None):
                ax.scatter(x, y, z, s=1000, c="r", marker=".")
            else: #sono nel caso custom test
                ax.scatter(x, y, z, s=1000, c=color, marker=".")
        else: #altrimenti la plotto blu
            ax.scatter(x, y, z, s=1000, c="b", marker=".")

        #ax.plot(x,y,z) #sharp lines

    def plot_different_views(self, fig, test_image, epoch, title, reference_image=None, start=1, rows=2, cols=4, color=None):
        if (reference_image is None):

            ax = fig.add_subplot(rows,cols,start+0, projection='3d')
            self.plotting_3d(test_image, epoch=epoch, ax=ax, title=title, color=color)

            ax = fig.add_subplot(rows,cols, start+1, projection='3d')
            self.plotting_3d(test_image, epoch=epoch, ax=ax, color=color)
            ax.view_init(30, 90)

            ax = fig.add_subplot(rows,cols, start+2, projection='3d')
            self.plotting_3d(test_image, epoch=epoch, ax=ax, color=color)
            ax.view_init(30, 180)

            ax = fig.add_subplot(rows,cols,start+3, projection='3d')
            self.plotting_3d(test_image, epoch=epoch, ax=ax, color=color)
            ax.view_init(30, 270)
            print("RELATIVE TO ORIGINAL")
        else:

            dice=self.dice_score(reference_image, test_image)
            ax = fig.add_subplot(rows,cols, start + 0, projection='3d')
            self.plotting_3d(test_image, epoch=epoch, ax=ax, title=title+" dice="+str(dice), reference_image=reference_image, color=color)

            ax = fig.add_subplot(rows,cols, start + 1, projection='3d')
            self.plotting_3d(test_image, epoch=epoch, ax=ax, reference_image=reference_image, color=color)
            ax.view_init(30, 90)

            ax = fig.add_subplot(rows,cols, start + 2, projection='3d')
            self.plotting_3d(test_image, epoch=epoch, ax=ax, reference_image=reference_image, color=color)
            ax.view_init(30, 180)

            ax = fig.add_subplot(rows,cols, start + 3, projection='3d')
            self.plotting_3d(test_image, epoch=epoch, ax=ax, reference_image=reference_image, color=color)
            ax.view_init(30, 270)
            print("RELATIVE TO RECONSTRUCTED")

    def save_latent(self, to_save):

        if to_save=='train':
            num_batches = self.x_train.shape[0] // self.size_batch
            x=self.x_train
            y=self.y_train

        else:
            num_batches = self.x_test.shape[0] // self.size_batch
            x = self.x_test
            y = self.y_test

        latents = self.encode_entire_dataset(x, y)
        print(latents)
        print("done with save_latents!")
        np.save(os.path.join(self.save_dir, "encoder_mean_"+ str(to_save)+".npy"), latents)
        print("latents "+str(to_save)+" saved.")

    def test_progression(self, images, classes, name, epoch, start_index=0, plot=False): #initial_age is None for custom_test
        #start index mi serve per scegliere l'immagine a caso che voglio ricostruire
        max_images_to_test=1
        test_dir = os.path.join(self.save_dir, 'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        print("\n\n\n" +str(self.size_batch)+"\n\n\n")
        images = images[start_index:start_index+max_images_to_test, :, :, :, :]; print("\n806: images.shape " + str(images.shape) + "\n")
        classes = classes[start_index:start_index+max_images_to_test, :]; print("\n807: classes.shape " + str(classes.shape) + "\n")
        size_sample = images.shape[0]
        labels = np.arange(self.num_categories); print("\n809:size_sample " + str(size_sample) + "\n")
        labels = np.repeat(labels, size_sample); print("\n810: labels.shape " + str(labels.shape) + "\n")
        query_labels = np.ones(
            shape=(self.num_categories, self.num_categories),
            dtype=np.float32
        ) * self.image_value_range[0]

        # con questo loop scelgo, una dopo l'altra, tutte le fasce di eta'e gli assegno il valore 1.
        # in questo modo cerco di generare un MRI per ogni fascia d'eta'.
        for i in range(query_labels.shape[0]):
            query_labels[i, labels[i]] = self.image_value_range[-1]
        print("\n820: query_labels.shape " + str(query_labels.shape) + "\n")

        query_images = np.tile(images, [self.num_categories, 1, 1, 1, 1]); print("\n822: query_images.shape " + str(query_images.shape) + "\n")
        query_classes = np.tile(classes, [self.num_categories, 1]); print("\n823: query_classes.shape " + str(query_classes.shape) + "\n")
        z, G = self.session.run(
            [self.z, self.G],
            feed_dict={
                self.input_image: query_images,
                self.classes: query_classes,
                self.age: query_labels,
            }
        )

        if (plot):
            fig = plt.figure(figsize=[40, 40])
            self.plot_different_views( fig, G[0], epoch, title="reconstruction", reference_image=None, start=1, rows=5, cols=4, color="b")
            self.plot_different_views( fig, G[2], epoch, title="reconstruction", reference_image=None, start=5, rows=5, cols=4, color="b")
            self.plot_different_views( fig, G[4], epoch, title="reconstruction", reference_image=None, start=9, rows=5, cols=4, color="b")
            self.plot_different_views( fig, G[6], epoch, title="reconstruction", reference_image=None, start=13, rows=5, cols=4, color="b")
            self.plot_different_views( fig, G[8], epoch, title="reconstruction", reference_image=None, start=17, rows=5, cols=4, color="b")
            # class_for_title= classes[start_index]
            # plt.set_title("progressing/regressing from patient  classes = classes[start_index")
            if (str(epoch)!="test"):
                plt.savefig(os.path.join(test_dir,str(epoch)+".png"))
            else:
                plt.savefig(os.path.join(test_dir, name))
            plt.close()
            
        return query_images, G

    def test_and_t_test(self, name ,epoch, class_id_for_custom=None):
        volumes = np.zeros([self.x_test.shape[0], 2]) #complessivo
        volumes_batch = np.zeros([1,2]) # la sovrascrivo ad ogni iterazioni
        decrease_percentage=np.zeros([self.x_test.shape[0]])
        #test(sample_images, sample_label_classes, "test"+str(test_num)+"_"+str(name), start_index=np.random.randint(low=0, high=sample_images.shape[0]), epoch=epoch)
        for ind_batch in range(self.x_test.shape[0]//self.size_batch):
            sample_images, sample_label_age, sample_label_classes = self.sample_batches(start_index=ind_batch*self.size_batch)

            if class_id_for_custom is not None: #sono in custom test
                sample_label_classes = np.ones(
                             shape=(sample_images.shape[0], 3),
                             dtype=np.float32
                         ) * self.image_value_range[0]
                for i in range(sample_label_classes.shape[0]):
                        sample_label_classes[i, class_id_for_custom] = self.image_value_range[-1]

            # ripeto per tutte le immagini in quel batch
            for k in range(self.size_batch):
                if ind_batch == 0 and k == 0:
                    plot = True
                else:
                    plot = False
                    #ho cambiato lo start index qui sotto
                    print("1001: ", str(ind_batch*self.size_batch+k))
                query_images, G = self.test_progression(sample_images, sample_label_classes, name=name, epoch=epoch, start_index=k, plot=plot)
                #ind_batch*self.size_batch+k
                print("G SHAPE: ", G.shape)
                print("query images: ", query_images.shape)
                volumes_batch[0,0] = self.calculate_volume(G[0]) #volume of the image at the start_progression
                volumes_batch[0,1] = self.calculate_volume(G[self.size_batch-1]) #volume of the image at the end_progression
                current_percentage=np.divide(np.multiply(np.subtract(volumes_batch[0,0],volumes_batch[0,1]),100),volumes_batch[0,0])

                if ind_batch == 0 and k == 0:
                    decrease_percentage = current_percentage
                    volumes = volumes_batch
                else:
                    decrease_percentage = np.append(decrease_percentage, current_percentage)
                    volumes = np.append(volumes, volumes_batch, axis=0)

        t_test_file = os.path.join(self.save_dir, 't_test.txt')
        sys.stdout = open(t_test_file, 'w+')

        print("VOLUMES:")
        print(volumes.shape)
        print(volumes)
        print("DECREASE PERCENTAGE")
        print(decrease_percentage.shape)
        print(decrease_percentage)
        decrease_percentage=np.array(decrease_percentage)

        print("percentage mean: "+str(np.mean(decrease_percentage)))
        print("percentage std: " + str(np.std(decrease_percentage)))
        if "custom" in name:
            np.save(os.path.join(self.save_dir, 'volumes_'+name.split("_")[1]+'.npy'), volumes)
            np.save(os.path.join(self.save_dir, 'decrease_percentage_'+name.split("_")[1]+'.npy'), decrease_percentage)
        else:
            np.save(os.path.join(self.save_dir, 'volumes.npy'), volumes)
            np.save(os.path.join(self.save_dir, 'decrease_percentage.npy'), decrease_percentage)

        #riporto sys.stdout dov'era prima
        training_steps = os.path.join(self.save_dir, 'epochs.txt')
        sys.stdout = open(training_steps, 'a+')
        self.calculate_dice_loss(self.x_test, self.y_test, epoch, "test")
        self.calculate_dice_loss(self.x_train, self.y_train, epoch, "train")

    def custom_test(self):
        self.test_and_t_test(name="custom_nc",epoch="test", class_id_for_custom=0)
        self.test_and_t_test(name="custom_mci",epoch="test", class_id_for_custom=1)
        self.test_and_t_test(name="custom_ad",epoch="test", class_id_for_custom=2)

    def dice_score(self, original_image, reconstructed_image):
        original_flattened=original_image.flatten()
        reconstructed_flattened=reconstructed_image.flatten()

        dice= 2*np.sum(np.multiply(original_flattened, reconstructed_flattened)) / (np.sum(original_flattened)+np.sum(reconstructed_flattened))
        print("dice: " + str(dice) +" of type: "+str(type(dice)))
        dice_format = float("%.2f" % dice)
        #print("dice_format: " + str(dice_format) + " of type: " + str(type(dice_format)))
        return dice_format

    def calculate_dice_loss(self, x, y, epoch, test_or_train):

        num_batches = x.shape[0] // self.size_batch
        dice_score=np.zeros(shape=[x.shape[0]], dtype=np.float32)

        for ind_batch in range(num_batches):
            sample_images, sample_label_age, sample_label_classes = self.sample_batches(x=x, y=y, start_index=ind_batch * self.size_batch)
            if sample_images.shape[0]<10:
                break;
            z, G = self.session.run(
                [self.z, self.G],
                feed_dict={
                    self.input_image: sample_images,
                    self.classes: sample_label_classes,
                    self.age: sample_label_age,
                }
            )

            print(test_or_train)
            for i in range(self.size_batch):
                print("i ", i)
                print("index: ", ind_batch*self.size_batch+i)
                dice= self.dice_score(sample_images[i], G[i])
                #print("DICE ", dice)
                dice_score[ind_batch*self.size_batch+i] = dice
                if "train" in test_or_train:
                    print("dapaRICE: ", dice)
        if "train" in test_or_train:
            print("FINISHED: ", ind_batch+self.size_batch+i)
        print("total dice", dice_score)
        mean=float(np.mean(dice_score))
        print("mean", mean)

        if (test_or_train == "test"):
            self.mean_dice_test.append(mean)
        else:
            self.mean_dice_train.append(mean)

        print("dice plot")

        if (test_or_train == "test"):
            y = self.mean_dice_test
        else:
            y = self.mean_dice_train
        print("y ", y)
        print("1101: ", len(y))
        x = np.arange(0, len(y));
        print("x : ", x)
        plt.plot(x, y)
        plt.ylim(0,1)
        plt.savefig(os.path.join(self.save_dir, "mean_dice_"+str(test_or_train)+".png"))
        plt.close()

    def send_email(self, epoch, time_left):

        fromaddr = "teslak40@gmail.com"
        frompsw = "munichbangkokLAX"
        toaddr = "milana.diletta@gmail.com"
        msg = MIMEMultipart()
        msg['From'] = fromaddr
        msg['To'] = toaddr
        msg['Subject'] = '[{}/{}] Tesla {} Progress Report'.format(epoch, self.epochs, self.save_dir.split("/")[-1])

        body = "TIME LEFT: {} hours and {} minutes". format(int(time_left / 3600), int(time_left % 3600 / 60))
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(fromaddr, frompsw)
        text = msg.as_string()
        server.sendmail(fromaddr, toaddr, text)
        server.quit()

        print("done. email sent")

    def encode_entire_dataset(self, x, y):

        num_batches = x.shape[0] // self.size_batch
        latents = np.zeros([x.shape[0], self.num_z_channels])
        print("LATENTS STILL 0 SHAPE: ", latents.shape)

        for ind_batch in range(num_batches):
            # non ci vuole controllo per il limite dell'array! Niente index out of bound exception! Che figata!!!
            batch_images = x[ind_batch * self.size_batch:(ind_batch + 1) * self.size_batch]
            batch_images=np.reshape(batch_images, newshape=[self.size_batch, self.img_x, self.img_y, self.img_z, 1])
            # potrei avere una batch eccezionalmente piu corta qui
            # current_batch_size = (ind_batch + 1) * self.size_batch - ind_batch * self.size_batch
            batch_images = np.array(batch_images).astype(np.float32)

            z = self.session.run(
                self.z_before_tanh,
                feed_dict={
                self.input_image:batch_images}
            )
            if (ind_batch == 0):
                latents = z
            else:
                latents = np.append(latents, z, axis=0)
        print("DONE WITH ENCODING! SHAPE:", latents.shape)
        return latents

    def calculate_volume(self, image):
        image=np.reshape(image, newshape=[self.img_x, self.img_y, self.img_z])
        print("KUDURO: image.shape", image.shape)
        x, y, z = image.nonzero()  # I use this only for the next print
        print(str(x.shape[0]))
        return int(x.shape[0])

    def visualise_supervised_finetuning(self):
        supervised_finetuning_path_test = self.save_dir + "\\supervised_finetuning\\test\\"
        if not os.path.exists(supervised_finetuning_path_test):
            os.makedirs(supervised_finetuning_path_test)
            print("CREATED: ", supervised_finetuning_path_test)

        self.visualiser.directory = supervised_finetuning_path_test
        self.visualiser.ScatterOfLatentSpace(dataset_images=self.x_test, dataset_encoded_images=self.encode_entire_dataset(self.x_test, self.y_test), dataset_labels=self.y_test)

        supervised_finetuning_path_train = self.save_dir + "\\supervised_finetuning\\train\\"
        if not os.path.exists(supervised_finetuning_path_train):
            os.makedirs(supervised_finetuning_path_train)
            print("CREATED: ", supervised_finetuning_path_train)

        self.visualiser.directory = supervised_finetuning_path_train
        self.visualiser.ScatterOfLatentSpace(dataset_images=self.x_train, dataset_encoded_images=self.encode_entire_dataset(self.x_train,self.y_train), dataset_labels=self.y_train)

    def different_perplexities(self, dir, dataset_images, dataset_labels, tsne_perplexity):

        #os.makedirs(dir)
        self.visualiser.directory = dir
        self.visualiser.ScatterOfLatentSpace(dataset_images=dataset_images, dataset_labels=dataset_labels, dataset_encoded_images=self.encode_entire_dataset(dataset_images,dataset_labels),
                                         tsne_perplexity=tsne_perplexity)
        # # visualiser.computeTSNEProjectionOfLatentSpace(dataset_images=dataset_images, dataset_labels=dataset_labels, n_components=3, _3d=True)
        # visualiser.computeTSNEProjectionOfPixelSpace(dataset_images=dataset_images, dataset_labels=dataset_labels, n_components=3,_3d=True)
        # visualiser.computeTSNEProjectionOfLatentSpace(dataset_images=dataset_images, dataset_labels=dataset_labels,
        #                                               tsne_perplexity=tsne_perplexity, n_components=2, _3d=False)
        # visualiser.computeTSNEProjectionOfPixelSpace(dataset_images=dataset_images, dataset_labels=dataset_labels,
        #                                              tsne_perplexity=tsne_perplexity, n_components=2, _3d=False)
        #

    def visualise(self, epoch):
        print("637 VISUALISING")
        test_path=self.save_dir + "\\latent_space_test_set\\"
        if not os.path.exists(test_path):
            os.makedirs(test_path)
            print("CREATED: ", test_path)

        train_path=self.save_dir+"\\latent_space_train_set\\"
        if not os.path.exists(train_path):
            os.makedirs(train_path)
            print("CREATED: ", train_path)

        progression_path = self.save_dir + "\\progression\\"
        if not os.path.exists(progression_path):
            os.makedirs(progression_path)
            print("CREATED: ", progression_path)

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

        #writing in the train path directory
        self.visualiser.directory = train_path
        self.different_perplexities(dir=train_path + "perplexity_3_", dataset_images=self.x_train, dataset_labels=self.y_train, tsne_perplexity=3)
        self.different_perplexities(dir=train_path + "perplexity_5_", dataset_images=self.x_train, dataset_labels=self.y_train, tsne_perplexity=5)
        self.different_perplexities(dir=train_path + "perplexity_10_", dataset_images=self.x_train, dataset_labels=self.y_train, tsne_perplexity=10)
        self.different_perplexities(dir=train_path + "perplexity_20_", dataset_images=self.x_train, dataset_labels=self.y_train, tsne_perplexity=20)
        self.different_perplexities(dir=train_path + "perplexity_30_", dataset_images=self.x_train, dataset_labels=self.y_train, tsne_perplexity=30)
        self.different_perplexities(dir=train_path + "perplexity_40_", dataset_images=self.x_train, dataset_labels=self.y_train, tsne_perplexity=40)
        self.different_perplexities(dir=train_path + "perplexity_50_", dataset_images=self.x_train, dataset_labels=self.y_train, tsne_perplexity=50)
        self.different_perplexities(dir=train_path + "perplexity_60_", dataset_images=self.x_train, dataset_labels=self.y_train, tsne_perplexity=60)