from __future__ import division
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize, imsave
import load_path as lp
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.morphology import binary_dilation

def conv3d(input_map, num_output_channels, size_kernel=5, stride=2, add_bias=True, name='conv2d', batchnorm=False):

    with tf.variable_scope(name) as scope:
        stddev = np.sqrt(2.0 / (np.sqrt(input_map.get_shape()[-1].value * num_output_channels) * size_kernel ** 2))
        kernel = tf.get_variable(
            name='w',
            shape=[size_kernel, size_kernel, size_kernel, input_map.get_shape()[-1], num_output_channels],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=stddev)
        )
        #print("\n\n\n 21:",kernel.name)
        biases = tf.get_variable(
            name='b',
            shape=[num_output_channels],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0)
        )
        #print("\n\n\n 28: ",biases.name)
        conv = tf.nn.conv3d(input_map, kernel, strides=[1, stride, stride, stride, 1], padding='SAME')
        if (add_bias):
            conv= tf.nn.bias_add(conv, biases)
        if (batchnorm):
            conv=tf.contrib.layers.batch_norm(conv)
        return conv

def fc(input_vector, num_output_length, name='fc', batchnorm=False):
    print("\n\n\n num output length ", num_output_length)
    print(input_vector.get_shape()[1].value)
    with tf.variable_scope(name):
        stddev = np.sqrt(1.0 / (np.sqrt(input_vector.get_shape()[-1].value * num_output_length)))
        w = tf.get_variable(
            name='w',
            shape=[input_vector.get_shape()[1], num_output_length],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(stddev=stddev)
        )
        b = tf.get_variable(
            name='b',
            shape=[num_output_length],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0)
        )
    fc=tf.matmul(input_vector, w) + b
    if (batchnorm):
        fc = tf.contrib.layers.batch_norm(fc)
    return fc

def deconv3d(input_map, output_shape, size_kernel=5, stride=2, complete_stride_for_last_layer=None,stddev=0.02, last_deconv=False, add_bias=True, name='deconv2d', batchnorm=False):
    with tf.variable_scope(name):
        stddev = np.sqrt(1.0 / (np.sqrt(input_map.get_shape()[-1].value * output_shape[-1]) * size_kernel ** 2))
        # filter : [height, width, output_channels, in_channels]
        kernel = tf.get_variable(
            name='w',
            shape=[size_kernel, size_kernel, size_kernel, output_shape[-1], input_map.get_shape()[-1]],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(stddev=stddev)
        )
        biases = tf.get_variable(
            name='b',
            shape=[output_shape[-1]],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0)
        )
        if last_deconv:
            assert complete_stride_for_last_layer is not None, "in the last layer the stride has to be !=None"
            deconv = tf.nn.conv3d_transpose(input_map, kernel, strides=complete_stride_for_last_layer,
                                            output_shape=output_shape)
            print("strides: ",complete_stride_for_last_layer)
            print("output_shape ", output_shape)

        else:
            deconv = tf.nn.conv3d_transpose(input_map, kernel, strides=[1, stride, stride, stride, 1], output_shape=output_shape)

        if (add_bias):
            deconv= tf.nn.bias_add(deconv, biases)
        if (batchnorm):
            deconv=tf.contrib.layers.batch_norm(deconv)
        return deconv

def lrelu(logits, leak=0.2):
    return tf.maximum(logits, leak * logits)

def concat_label(x, label, duplicate=1):
    x_shape = x.get_shape().as_list()
    if duplicate < 1:
        return x
    # duplicate the label to enhance its effect, does it really affect the result?
    label = tf.tile(label, [1, duplicate])
    label_shape = label.get_shape().as_list()
    if len(x_shape) == 2:
        return tf.concat(1, [x, label])
    elif len(x_shape) == 5:
        label = tf.reshape(label, [x_shape[0], 1, 1, 1, label_shape[-1]])
        print(label.get_shape())
        return tf.concat(values=[x, label*tf.ones([x_shape[0], x_shape[1], x_shape[2],x_shape[3], label_shape[-1]])], concat_dim=4)

def save_batch_images(
        batch_images,   # a batch of images
        save_path,  # path to save the images
        image_value_range=(0,1),   # value range of the input batch images
        size_frame=None,     # size of the image matrix, number of images in each row and column
        test_phase=False
):
    print("batch_images shape", batch_images.shape)
    print("size_frame", size_frame)
    # transform the pixcel value to 0~1
    # images = (batch_images - image_value_range[0]) / (image_value_range[0] - image_value_range[0])
    images=batch_images
    if size_frame is None:
        auto_size = int(np.ceil(np.sqrt(images.shape[0])))
        size_frame = [auto_size, auto_size] # vuole creare un riquadro in cui salvare tutte le immagini generate
        print("size_frame: ", size_frame)
    img_h, img_w, img_z = batch_images.shape[1], batch_images.shape[2], batch_images.shape[3]
    frame = np.zeros([img_h * size_frame[0], img_w * size_frame[1],  3])
    print("frame: ", frame.shape)

    if (test_phase):
        saved_index=np.arange(batch_images.shape[0]) #if I want to save all images
        saved_index = [0, 2, 4, 6, 8]
        print("TESTING")
        count=0
        for ind, image in enumerate(images):
            print("IND: ", ind)
            if (ind in saved_index):
                frame[(count * img_h):(count * img_h + img_h), (0 * img_w):(0 * img_w + img_w),:] = image
                print("in frame: " + str((count * img_h)) + ":" + str(count * img_h + img_h) + "," + str(
                    0 * img_w) + ":" + str(0 * img_w + img_w) + ", :")
                count+=1
    else:
        print("TRAINING")
        for ind, image in enumerate(images):
            print("\n\n\n\n"+str(image.shape)+"\n\n\n\n")
            #plt.imshow(np.reshape(image[:,:,0], newshape=[image.shape[0],image.shape[1]]))
            #plt.show()
            ind_col = ind % size_frame[1]
            ind_row = ind // size_frame[1]
            frame[(ind_row * img_h):(ind_row * img_h + img_h), (ind_col * img_w):(ind_col * img_w + img_w), :] = image
            print("ind_col: ", ind_col)
            print("ind_row: ", ind_row)
            print("in frame: "+str(ind_row * img_h)+":"+str(ind_row * img_h + img_h)+","+str(ind_col * img_w)+":"+str(ind_col * img_w + img_w)+", :")

    imsave(save_path, frame)

def import_mri_dataset(dataset_list, image_info, label_info,selected_slice, oneHot, allFields, conv3d, max_images, nc_only):
    complete_images = None
    complete_labels = None

    for dataset in dataset_list:
        print("\n")
        images_path=lp.load_image_path(dataset, conv3d, image_info )
        labels_path = lp.load_labels_path(dataset, oneHot, allFields, label_info )
        print("reading " + str(dataset) + " images...")
        print("images_path: ", images_path)
        print("labels_path: ", labels_path)
        images = np.load(images_path)
        print("14 ", images.shape)

        if (selected_slice!=None):
            #images=np.reshape(images[:,:,:,selected_slice], newshape=[images.shape[0], images.shape[1], images.shape[2],1])
            images = images[:, :, :, selected_slice]

        print("done", images.shape)
        print("reading " + str(dataset) + " labels...")
        labels = np.load(labels_path)
        print("done", labels.shape)

        assert labels.shape[0] == images.shape[0], "Error: labels" + str(labels.shape[0]) + "and images" + str(
            images.shape[0]) + " sizes do not match"

        if (complete_images==None):
            print("complete_images is None")
        if (complete_labels==None):
            print("complete_labels is None")
        # assert x, makes sure x is true!
        #prints if x is False
        #assert not(complete_images==None and not complete_labels==None), "Something is wrong: images is None while labels is not"
        #assert not(complete_labels==None and not complete_images==None), "Something is wrong: labels is None while images is not"

        if (complete_images is None):
            complete_images = images
            complete_labels = labels
        else:
            complete_images = np.append(complete_images, images, axis=0)
            complete_labels = np.append(complete_labels, labels, axis=0)

        print("some news from complete_images")
        print(complete_images.shape)
        print(complete_labels.shape)

    #this is a huge mistake I corrected! 08/29 I wasn't really shuffling them because this
    #shuffle op is not in-place, and I was not assigning it to complete_images and complete_labels again
    #this was probably the reason why the images looked so similar in the sampling face: since in this case
    #I am taking many similar images (slices one right after the other) it might be that they all looked the same.
    # I shuffle before splitting into train and test
    complete_images, complete_labels= shuffle(complete_images, complete_labels)

    complete_images=complete_images[:max_images,]
    complete_labels=complete_labels[:max_images,]

    n_samples=complete_images.shape[0]

    #labels=ld.read_labels(n_samples, 3, labels_path= labels_path, oneHotLabels=False)
    print("\n\n\ncomplete_images: ", complete_images.shape)
    train_percentage = 0.9

    idx_nc = np.where(complete_labels[:,0]==0)[0]
    idx_mci = np.where(complete_labels[:,0]==1)[0]
    idx_ad = np.where(complete_labels[:,0]==2)[0]
    print("\n\n we have {} nc patients, {} mci patients and {} ad patients".format(len(idx_nc), len(idx_mci), len(idx_ad)))

    if (nc_only):
        print("keeping normal controls only!")
        complete_labels=complete_labels[idx_nc]
        #print("labels shape: ",complete_labels.shape)
        complete_images=complete_images[idx_nc]
        #print("images shape: ", complete_images.shape)
        n_samples=complete_images.shape[0]

    train_index =int((n_samples+1)*train_percentage)

    y_train=complete_labels[0:train_index]
    y_test=complete_labels[train_index:]

    x_train = complete_images[0:train_index]
    x_test= complete_images[train_index:]

    #x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    #x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    print("\n\n\nMAX x_train: ", str(np.max(x_train)))
    print("MIN x_train: ", str(np.min(x_train)))

    print("MAX x_test: ", str(np.max(x_test)))
    print("MIN x_test: ", str(np.min(x_test)))

    #print("new dim I want to create: ", ((x_test.shape[0],) + original_img_size))
    #x_test = x_test.reshape((x_test.shape[0],) + original_img_size)

    #guaranteeing always 4D. Not needed for now
    # if (np.ndim(x_train)==3):
    #     x_train = np.reshape(x_train, newshape=(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    #     x_test = np.reshape(x_test, newshape=(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

    #non ho lo stesso problema con i labels
    #y_train = np.reshape(y_train, newshape=(y_train.shape[0], y_train.shape[1], y_train.shape[2], 1))
    #y_test = np.reshape(y_test, newshape=(y_test.shape[0], y_test.shape[1], y_test.shape[2], 1))

    print('x_train.shape:', x_train.shape)
    print('x_test.shape:', x_test.shape)
    print('y_train.shape:', y_train.shape)
    print('y_test.shape:', y_test.shape)
    #print_images(x_train, "complete_images")
    return x_train, y_train, x_test,y_test


def apply_gradients_dilation(x):
        print("267: ", x.shape)
        if len(x.shape)==5:
            shape_5=True
            x=np.reshape(x, newshape=[x.shape[0], x.shape[1], x.shape[2], x.shape[3]])
        else:
            shape_5=False
        print ("applying gradients and dilation for every element from 0 to : ",x.shape[0])
        for i in range(x.shape[0]):

            volume = np.gradient(x[i])
            #print("248", x[i].shape)
            #print(volume[0].shape)
            #print("min: ", np.min(volume[0]))
            #print("max: ", np.max(volume[0]))
            #print("counting occurrencies in volume[0]")
            #unique, counts = np.unique(volume[0], return_counts=True)
            #print(dict(zip(unique, counts)))
            #print("NOW THEY SHOULD BECOME EITHER 0 OR 1")
            for k in np.nditer(volume[0], op_flags=['readwrite']):
                if (k[...]!=0.0):
                    k[...]=1
            # print("counting occurrencies in volume[0]")
            # unique, counts = np.unique(volume[0], return_counts=True)
            # print(dict(zip(unique, counts)))

            #print(volume[1].shape)
            # print("min: ", np.min(volume[1]))
            # print("max: ", np.max(volume[1]))
            # print("counting occurrencies in volume[1]")
            # unique, counts = np.unique(volume[1], return_counts=True)
            # print(dict(zip(unique, counts)))
            # print("NOW THEY SHOULD BECOME EITHER 0 OR 1")
            for k in np.nditer(volume[1], op_flags=['readwrite']):
                if (k[...] != 0.0):
                    k[...] = 1
            # print("counting occurrencies in volume[1]")
            # unique, counts = np.unique(volume[1], return_counts=True)
            # print(dict(zip(unique, counts)))

            #print(volume[2].shape)
            # print("min: ", np.min(volume[2]))
            # print("max: ", np.max(volume[2]))
            # print("counting occurrencies in volume[2]")
            # unique, counts = np.unique(volume[2], return_counts=True)
            # print(dict(zip(unique, counts)))
            # print("NOW THEY SHOULD BECOME EITHER 0 OR 1")
            for k in np.nditer(volume[2], op_flags=['readwrite']):
                if (k[...] != 0.0):
                    k[...] = 1
            # print("counting occurrencies in volume[2]")
            # unique, counts = np.unique(volume[2], return_counts=True)
            # print(dict(zip(unique, counts)))

            volume = volume[0] + volume[1] + volume[2]
            # print("counting occurrencies in gradient sum")
            # unique, counts = np.unique(volume, return_counts=True)
            # print(dict(zip(unique, counts)))
            volume=np.clip(volume, a_min=0, a_max=1)
            # print("counting occurrencies in gradient sum CLIPPED")
            # unique, counts = np.unique(volume, return_counts=True)
            # print(dict(zip(unique, counts)))
            _, y, z = volume.nonzero()
            # print("sum gradients, min: ", np.min(volume))
            # print("sum gradients, max: ", np.max(volume))
            # print("x length: ", len(y))

            dilated_volume = binary_dilation(volume).astype(volume.dtype)
            #print("dilated min: ", np.min(dilated_volume))
            #print("dilated max: ", np.max(dilated_volume))
            #print("counting occurrencies in dilated vol")
            #unique, counts = np.unique(dilated_volume, return_counts=True)
            #print(dict(zip(unique, counts)))

            #print(x[i].shape)
            x[i, :, :, :]=dilated_volume
        if shape_5:
            x=np.reshape(x, newshape=[x.shape[0], x.shape[1], x.shape[2], x.shape[3], 1])

        print("DONE\n\n\n: ", x.shape)
        print("min: ", np.min(x))
        print("max: ", np.max(x))
        print("counting occurrencies before returning x")
        unique, counts = np.unique(x, return_counts=True)
        print(dict(zip(unique, counts)))

        return x

# def print_images(x, title):
#     max_images_per_frame=60
#     count_frames = int(np.ceil(x.shape[0] / max_images_per_frame))
#     max_images_sqrt = int(np.sqrt(max_images_per_frame))
#     count = 0  # to iterate through the lists
#
#     for i in range(count_frames):
#         f, axarr = plt.subplots(max_images_sqrt, max_images_sqrt, figsize=(25, 25))
#         for z in range(max_images_sqrt):
#             for j in range(max_images_sqrt):  # per ogni riga righe
#                 if (count < x.shape[0]):
#                     axarr[z, j].imshow(x[count])
#                     axarr[z, j].set_title(title)
#                     count += 1
#                 else:
#                     break;
#         plt.show()
#         plt.close()

def shuffle(images, labels):
        n_samples=images.shape[0]
        perm = np.arange(n_samples)
        np.random.shuffle(perm)
        return images[perm], labels[perm]

def oneHot(old_batch):
    # these 3 are just for the prints within this method, they are not the global ones
    nc_count = 0
    mci_count = 0
    ad_count = 0
    new_batch = np.zeros([old_batch.shape[0], 3])
    for i in range(old_batch.shape[0]):
        label_code = old_batch[i]
        # print("label code ", label_code)
        # print(type(label_code))
        # print(label_code.shape)
        if int(label_code) == 0:  # NC
            new_batch[i] = [1, 0, 0]
            nc_count += 1
        elif int(label_code) == 1:  # MCI
            new_batch[i] = [0, 1, 0]
            mci_count += 1
        else:  # AD
            new_batch[i] = [0, 0, 1]
            ad_count += 1
    print("COUNTING CLASSES IN ONEHOT BATCH")
    print("nc_count: ", nc_count)
    print("mci_count: ", mci_count)
    print("ad_count: ", ad_count)
    return new_batch
#
# def rotate_coronal_images(images):
#     print("IMAGES SHAPE: ", images.shape)
#
#     for i in range(images.shape[0]):
#         images[i,:,:,:]=np.rot90(images[i,:,:,:], axes=[1,2]);
#
#     return images