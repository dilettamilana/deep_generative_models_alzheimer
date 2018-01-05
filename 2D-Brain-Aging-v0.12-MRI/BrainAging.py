from __future__ import division
from scipy.misc import imread, imresize, imsave
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from scipy.io import savemat
from ops import *
import training_notes_helper as tn
import sys
from os.path import abspath
import collections
import matplotlib.pyplot as plt

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
                 num_encoder_channels=64,  # number of channels of the first conv layer of encoder
                 num_z_channels=120,  # number of channels of the layer z (noise or code)
                 num_categories=10,  # number of categories (age segments) in the training dataset
                 num_gen_channels=1024,  # number of channels of the first deconv layer of generator
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
        self.visualise_latent_frequency=5

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        print("Let's start!")

        # ************************************* input to graph ********************************************************
        self.input_image = tf.placeholder(
            tf.float32,
            [self.size_batch, self.size_image, self.size_image, self.num_input_channels],
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

        # ************************************* loss functions *******************************************************
        # loss function of encoder + generator
        # tf.nn.l2_loss( penalising_factor*(binary_map+1) * (input_image-G) )
        self.EG_loss = tf.nn.l2_loss(self.input_image - self.G) / self.size_batch  # L2 loss
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
        self.D_img_variables = [var for var in trainable_variables if 'D_' in var.name]

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

    def train(self,
              beta1=0.5,  # parameter for Adam optimizer
              decay_rate=1,  # learning rate decay (0, 1], 1 means no decay
              enable_shuffle=True,  # enable shuffle of the dataset
              use_trained_model=True,  # used the saved checkpoint to initialize the model
              ):

        #set stdout:
        training_steps = os.path.join(self.save_dir, 'epochs.txt')
        sys.stdout = open(training_steps, 'a+')

        # *************************** load file names of images ******************************************************
        # file_names = glob(os.path.join('./data', self.dataset_name, '*.jpg'))
        # size_data = len(file_names)

        # Now, seed makes random numbers predictable:
        # With the seed reset (every time), the same set of numbers will appear every time.
        # If the random seed is not reset, different numbers appear with every invocation:

        # np.random.seed(seed=2017)
        # if enable_shuffle:
        #     np.random.shuffle(file_names)

        ######  DATASET   #####
        self.dataset_list = ["AIBL", "OASIS", "Harp"]
        self.nc_only = False
        self.max_images=5000

        ######  IMAGES   #####
        self.image_info = "norm_coronal_4centerslices_flattened"
        self.conv3d = False
        # 2 coronal
        # 1 sagittal
        # 0 axial
        self.selected_slice=None

        ######  LABELS   #####
        self.label_info = "axial_4centerslices_flattened" #tanto qui non importa la prospettiva
        #fields = 6
        self.allFields = True
        self.oneHot = False

        self.x_train, self.y_train, self.x_test, self.y_test = import_mri_dataset(self.dataset_list, self.image_info, self.label_info,
                                                                     self.selected_slice, self.oneHot, self.allFields, self.conv3d,
                                                                     self.max_images, self.nc_only)

        size_data = self.x_train.shape[0] + self.x_test.shape[0]
        self.visualiser = visualising.Visualiser(self, _3d=False, n_z=self.num_z_channels, size_batch=self.size_batch, save_dir=self.save_dir, img_x=self.size_image, img_y=self.size_image, img_z=1, x_train=self.x_train, x_test=self.x_test, y_train=self.y_train, y_test=self.y_test)


        # *********************************** optimizer **************************************************************

        # over all, there are three loss functions, weights may differ from the paper because of different datasets
        #self.loss_EG = self.EG_loss + self.G_img_param * self.G_img_loss + self.E_z_param * self.E_z_loss + self.tv_param * self.tv_loss # slightly increase the params
        self.loss_EG = 10*self.EG_loss + self.G_img_loss
        self.loss_Di = self.D_img_loss_input + self.D_img_loss_G

        #self.loss_Di = tf.reduce_mean(logits_real - logits_fake)
        #self.loss_EG = tf.reduce_mean(logits_fake)

        # set learning rate decay
        self.EG_global_step = tf.Variable(0, trainable=False, name='global_step')
        EG_learning_rate = tf.train.exponential_decay(
            learning_rate=self.learning_rate,
            global_step=self.EG_global_step,
            decay_steps=size_data / self.size_batch * 2,
            decay_rate=decay_rate,
            staircase=True
        )

        # optimizer for encoder + generator
        self.EG_optimizer = tf.train.AdamOptimizer(
            learning_rate=EG_learning_rate,
            beta1=beta1
        ).minimize(
            loss=self.loss_EG,
            global_step=self.EG_global_step,
            var_list=self.E_variables + self.G_variables
        )

        # optimizer for encoder + generator
        # self.G_img_optimizer = tf.train.AdamOptimizer(
        #     learning_rate=EG_learning_rate,
        #     beta1=beta1
        # ).minimize(
        #     loss=self.loss_G_img,
        #     global_step=self.EG_global_step,
        #     var_list=self.E_variables + self.G_variables
        # )

        # optimizer for discriminator on image
        self.D_img_optimizer = tf.train.AdamOptimizer(
            learning_rate=EG_learning_rate,
            beta1=beta1
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

        sample_images = self.x_test[0:self.size_batch]
        #print("SAMPLE IMAGES: ",sample_images.shape)
        sample_labels = self.y_test[0:self.size_batch]
        #print("SAMPLE LABELS: ",sample_labels.shape)

        if self.num_input_channels == 1:
            sample_images = np.array(sample_images).astype(np.float32)[:, :, :, None]
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
            label = sample_labels[i, 3]
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
                    batch_images = np.array(batch_images).astype(np.float32)[:, :, :, None]
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

                updating_disc=False
                updating_gen=False
                #update the generator only when mod=0
                if (np.mod(epoch, self.disc_updates)==0):
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

                if (np.mod(epoch, self.gen_updates)==0):
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

            # estimate left run time
            elapse = time.time() - start_time
            time_left = (self.epochs - epoch - 1) * elapse
            print("\tTime left: %02d:%02d:%02d" %
                  (int(time_left / 3600), int(time_left % 3600 / 60), time_left % 60))

            print("\nUpdating [%s] | Epoch [%3d/%3d]\n\tEG_err=%.4f" %
                  (turn, epoch + 1, self.epochs, EG_err))
            print("\tGi=%.4f\tDi=%.4f\tDiG=%.4f" % (Gi_err, Di_err, DiG_err))

            name = '{:02d}'.format(epoch + 1)
            if (epoch%1==0):
                if (epoch==0):
                    np.save(os.path.join(self.save_dir, "labels_train.npy"), self.y_train)
                    np.save(os.path.join(self.save_dir, "labels_test.npy"), self.y_test)

                    self.sample(sample_images, sample_label_age, sample_label_classes, name, original=True)
                else:
                    self.sample(sample_images, sample_label_age, sample_label_classes, name)

                for test_num in range(self.num_age_progression_tests):
                    self.test(sample_images, sample_label_classes, "test"+str(test_num)+"_"+str(name), start_index=np.random.randint(low=0, high=sample_images.shape[0]))

            # save checkpoint for each 10 epoch
            if ((np.mod(epoch, self.epochs-1) == 0 and epoch>0) or np.mod(epoch, 5)):
                self.save_checkpoint()

            if (np.mod(epoch,self.visualise_custom_test_frequency)==0):
                self.custom_test()

            self.save_latent(to_save="train")
            self.save_latent(to_save="test")
            if (np.mod(epoch, self.visualise_latent_frequency) == 0):
                self.visualise(epoch)

            if np.mod(epoch, int(self.epochs/self.send_email_report_frequency))==0 or epoch==(self.epochs-1) or epoch==0:
                self.send_email(epoch, time_left)

        self.writer.close()

    def encoder(self, image, reuse_variables=False):
        print("\n\n\nENCODER\n")
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        num_layers = int(np.log2(self.size_image)) - int(self.size_kernel / 2)
        current = image
        # conv layers with stride 2
        for i in range(num_layers):
            name = 'E_conv' + str(i)
            with tf.name_scope(name) as scope:
                current = conv2d(
                        input_map=current,
                        num_output_channels=self.num_encoder_channels * (2 ** i),
                        size_kernel=self.size_kernel,
                        name=name
                    )
                print(str(current.name) + " - " + str(current.get_shape()))
                current = tf.nn.relu(current)
                print(str(current.name) + " - " + str(current.get_shape()))

        # fully connection layer
        name = 'E_fc'
        with tf.name_scope(name) as scope:
            current = fc(
                input_vector=tf.reshape(current, [self.size_batch, -1]),
                num_output_length=self.num_z_channels,
                name=name
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
        num_layers = int(np.log2(self.size_image)) - int(self.size_kernel / 2)
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
        size_mini_map = int(self.size_image / 2 ** num_layers)
        # fc layer
        name = 'G_fc'
        with tf.name_scope(name) as scope:
            current = fc(
                input_vector=z,
                num_output_length=self.num_gen_channels * size_mini_map * size_mini_map,
                name=name
            )
            print(str(current.name) + " - " + str(current.get_shape()))
            # reshape to cube for deconv
            current = tf.reshape(current, [-1, size_mini_map, size_mini_map, self.num_gen_channels])
            print(str(current.name) + " - " + str(current.get_shape()))
            current = tf.nn.relu(current)
            print(str(current.name) + " - " + str(current.get_shape()))
        # deconv layers with stride 2
        for i in range(num_layers):
            name = 'G_deconv' + str(i)
            with tf.name_scope(name) as scope:
                current = deconv2d(
                        input_map=current,
                        output_shape=[self.size_batch,
                                      size_mini_map * 2 ** (i + 1),
                                      size_mini_map * 2 ** (i + 1),
                                      int(self.num_gen_channels / 2 ** (i + 1))],
                        size_kernel=self.size_kernel,
                        name=name
                    )
                print(str(current.name) + " - " + str(current.get_shape()))
                current = tf.nn.relu(current)
                print(str(current.name) + " - " + str(current.get_shape()))
        name = 'G_deconv' + str(i+1)
        with tf.name_scope(name) as scope:
            current = deconv2d(
                input_map=current,
                output_shape=[self.size_batch,
                              self.size_image,
                              self.size_image,
                              int(self.num_gen_channels / 2 ** (i + 2))],
                size_kernel=self.size_kernel,
                stride=1,
                name=name
            )
            print(str(current.name) + " - " + str(current.get_shape()))
            current = tf.nn.relu(current)
            print(str(current.name) + " - " + str(current.get_shape()))
        name = 'G_deconv' + str(i + 2)
        with tf.name_scope(name) as scope:
            current = deconv2d(
                input_map=current,
                output_shape=[self.size_batch,
                              self.size_image,
                              self.size_image,
                              self.num_input_channels],
                size_kernel=self.size_kernel,
                stride=1,
                name=name
            )
            print(str(current.name) + " - " + str(current.get_shape()))

            # output
            current = tf.nn.tanh(current)
            print(str(current.name) + " - " + str(current.get_shape()))
        return current

    def discriminator_img(self, image, y, classes, is_training=True, reuse_variables=False, num_hidden_layer_channels=(16, 32, 64, 128), enable_bn=True):
        print("\n\n\nDISCRIMINATOR_IMG\n")
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        num_layers = len(num_hidden_layer_channels)
        current = image
        # conv layers with stride 2
        for i in range(num_layers):
            name = 'D_conv' + str(i)
            with tf.name_scope(name) as scope:
                current = conv2d(
                        input_map=current,
                        num_output_channels=num_hidden_layer_channels[i],
                        size_kernel=self.size_kernel,
                        name=name
                    )
                print(str(current.name) + " - " + str(current.get_shape()))
                if enable_bn:
                    name = 'D_bn' + str(i)
                    current = tf.contrib.layers.batch_norm(
                        current,
                        scale=False,
                        is_training=is_training,
                        scope=name,
                        reuse=reuse_variables
                    )
                    print(str(current.name) + " - " + str(current.get_shape()))
                current = tf.nn.relu(current)
                print(str(current.name) + " - " + str(current.get_shape()))
                if i == 0:
                    current = concat_label(current, y)
                    current = concat_label(current, classes, int(self.num_categories / 2))
        # fully connection layer
        name = 'D_fc1'
        with tf.name_scope(name) as scope:
            current = fc(
                input_vector=tf.reshape(current, [self.size_batch, -1]),
                num_output_length=1024,
                name=name
            )
            print(str(current.name) + " - " + str(current.get_shape()))
            current = lrelu(current)
            print(str(current.name) + " - " + str(current.get_shape()))

        name = 'D_fc2'
        with tf.name_scope(name) as scope:
            current = fc(
                input_vector=current,
                num_output_length=1,
                name=name
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

    def sample(self, images, labels, classes, name, original=False):

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
        print("zzzzzzzz shape: ", z.shape)
        size_frame = int(np.sqrt(self.size_batch))
        print("SSSSSSSHAPE: ", images.shape)

        if original==True:
            save_batch_images(
                batch_images=images,
                save_path=os.path.join(sample_dir, "original.png"),
                image_value_range=self.image_value_range
            )

        save_batch_images(
            batch_images=G,
            save_path=os.path.join(sample_dir, str(name)+".png"),
            image_value_range=self.image_value_range
        )

        #sending a blank image

        z, G = self.session.run(
            [self.z, self.G],
            feed_dict={
                self.input_image: np.zeros(shape=[images.shape[0], images.shape[1], images.shape[2], images.shape[3]]),
                self.age: labels,
                self.classes: classes
            }
        )
        save_batch_images(
            batch_images=G,
            save_path=os.path.join(sample_dir, str(name)+"_blank.png"),
            image_value_range=self.image_value_range
        )

        # self.z=np.random.rand(z.shape[0],z.shape[1])
        # G=self.session.run(
        #     self.G
        # )
        #
        # save_batch_images(
        #     batch_images=G,
        #     save_path=os.path.join(sample_dir, str(name)+"_random.png"),
        #     image_value_range=self.image_value_range
        # )

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

    def test(self, images, classes, name, start_index=0):
        max_images_to_test=1
        test_dir = os.path.join(self.save_dir, 'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        print("\n\n\n" +str(self.size_batch)+"\n\n\n")
        #images = images[:int(np.sqrt(self.size_batch)), :, :, :]
        images = images[start_index:start_index+max_images_to_test, :, :, :]; print("\n806: images.shape " + str(images.shape) + "\n")
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

        query_images = np.tile(images, [self.num_categories, 1, 1, 1]); print("\n822: query_images.shape " + str(query_images.shape) + "\n")
        query_classes = np.tile(classes, [self.num_categories, 1]); print("\n823: query_classes.shape " + str(query_classes.shape) + "\n")
        z, G = self.session.run(
            [self.z, self.G],
            feed_dict={
                self.input_image: query_images,
                self.classes: query_classes,
                self.age: query_labels,
            }
        )
        # if "norm_coronal" in self.image_info:
        #     query_images=rotate_coronal_images(query_images)
        save_batch_images(
            batch_images=query_images,
            save_path=os.path.join(test_dir, str(name.split("_")[0])+'_input.png'),
            image_value_range=self.image_value_range,
            size_frame=[int(query_images.shape[0]/2), 1],
            test_phase=True
        )
        #if "norm_coronal" in self.image_info:
            #query_images=rotate_coronal_images(G)
        save_batch_images(
            batch_images=G,
            save_path=os.path.join(test_dir, str(name)+".png"),
            image_value_range=self.image_value_range,
            size_frame=[int(query_images.shape[0]/2), 1],
            test_phase=True,
        )

    def custom_test(self):
        # if not self.load_checkpoint():
        #     print("\tFAILED >_<!")
        #     exit(0)
        # else:
        #     print("\tSUCCESS ^_^")

        num_samples = int(np.sqrt(self.size_batch))
        sample = self.x_test[0:num_samples]
        if self.num_input_channels == 1:
            images = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            images = np.array(sample).astype(np.float32)
        classes_nc = np.ones(
            shape=(num_samples, 3),
            dtype=np.float32
        ) * self.image_value_range[0]
        classes_mci = np.ones(
            shape=(num_samples, 3),
            dtype=np.float32
        ) * self.image_value_range[0]
        classes_ad = np.ones(
            shape=(num_samples, 3),
            dtype=np.float32
        ) * self.image_value_range[0]
        for i in range(classes_nc.shape[0]):
            classes_nc[i, 0] = self.image_value_range[-1]
            classes_mci[i, 1] = self.image_value_range[-1]
            classes_nc[i, 2] = self.image_value_range[-1]

        self.test(images, classes_nc, 'test_as_nc.png')
        self.test(images, classes_mci, 'test_as_mci.png')
        self.test(images, classes_ad, 'test_as_ad.png')

        print('\n\tDone! Results are saved as %s\n' % os.path.join(self.save_dir, 'test', 'test_as_xxx.png'))

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
            batch_images=np.reshape(batch_images, newshape=[self.size_batch, self.size_image, self.size_image, 1])
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

    def different_perplexities(dir, dataset_images, dataset_labels, tsne_perplexity):
        os.makedirs(dir)
        visualiser.directory = dir
        visualiser.ScatterOfLatentSpace(dataset_images=dataset_images, dataset_labels=dataset_labels,
                                        tsne_perplexity=tsne_perplexity)
        # visualiser.computeTSNEProjectionOfLatentSpace(dataset_images=dataset_images, dataset_labels=dataset_labels, n_components=3, _3d=True)
        # visualiser.computeTSNEProjectionOfPixelSpace(dataset_images=dataset_images, dataset_labels=dataset_labels, n_components=3,_3d=True)
        visualiser.computeTSNEProjectionOfLatentSpace(dataset_images=dataset_images, dataset_labels=dataset_labels,
                                                      tsne_perplexity=tsne_perplexity, n_components=2, _3d=False)
        visualiser.computeTSNEProjectionOfPixelSpace(dataset_images=dataset_images, dataset_labels=dataset_labels,
                                                     tsne_perplexity=tsne_perplexity, n_components=2, _3d=False)


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
        different_perplexities(dir=test_path + "perplexity_3_", dataset_images=self.x_test, dataset_labels=self.y_test, tsne_perplexity=3)
        different_perplexities(dir=test_path + "perplexity_5_", dataset_images=self.x_test, dataset_labels=self.y_test, tsne_perplexity=5)
        different_perplexities(dir=test_path + "perplexity_10_", dataset_images=self.x_test, dataset_labels=self.y_test, tsne_perplexity=10)
        different_perplexities(dir=test_path + "perplexity_20_", dataset_images=self.x_test, dataset_labels=self.y_test, tsne_perplexity=20)
        different_perplexities(dir=test_path + "perplexity_30_", dataset_images=self.x_test, dataset_labels=self.y_test, tsne_perplexity=30)
        different_perplexities(dir=test_path + "perplexity_40_", dataset_images=self.x_test, dataset_labels=self.y_test, tsne_perplexity=40)
        different_perplexities(dir=test_path + "perplexity_50_", dataset_images=self.x_test, dataset_labels=self.y_test, tsne_perplexity=50)
        different_perplexities(dir=test_path + "perplexity_60_", dataset_images=self.x_test, dataset_labels=self.y_test, tsne_perplexity=60)

        #writing in the train path directory
        self.visualiser.directory = train_path
        different_perplexities(dir=train_path + "perplexity_3_", dataset_images=self.x_train, dataset_labels=self.y_train, tsne_perplexity=3)
        different_perplexities(dir=train_path + "perplexity_5_", dataset_images=self.x_train, dataset_labels=self.y_train, tsne_perplexity=5)
        different_perplexities(dir=train_path + "perplexity_10_", dataset_images=self.x_train, dataset_labels=self.y_train, tsne_perplexity=10)
        different_perplexities(dir=train_path + "perplexity_20_", dataset_images=self.x_train, dataset_labels=self.y_train, tsne_perplexity=20)
        different_perplexities(dir=train_path + "perplexity_30_", dataset_images=self.x_train, dataset_labels=self.y_train, tsne_perplexity=30)
        different_perplexities(dir=train_path + "perplexity_40_", dataset_images=self.x_train, dataset_labels=self.y_train, tsne_perplexity=40)
        different_perplexities(dir=train_path + "perplexity_50_", dataset_images=self.x_train, dataset_labels=self.y_train, tsne_perplexity=50)
        different_perplexities(dir=train_path + "perplexity_60_", dataset_images=self.x_train, dataset_labels=self.y_train, tsne_perplexity=60)