from __future__ import division
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize, imsave
import load_path as lp
import matplotlib.pyplot as plt

def conv2d(input_map, num_output_channels, size_kernel=5, stride=2, name='conv2d'):
    with tf.variable_scope(name):
        stddev = np.sqrt(2.0 / (np.sqrt(input_map.get_shape()[-1].value * num_output_channels) * size_kernel ** 2))
        kernel = tf.get_variable(
            name='w',
            shape=[size_kernel, size_kernel, input_map.get_shape()[-1], num_output_channels],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=stddev)
        )
        biases = tf.get_variable(
            name='b',
            shape=[num_output_channels],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0)
        )
        conv = tf.nn.conv2d(input_map, kernel, strides=[1, stride, stride, 1], padding='SAME')
        return tf.nn.bias_add(conv, biases)


def fc(input_vector, num_output_length, name='fc'):
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
        return tf.matmul(input_vector, w) + b


def deconv2d(input_map, output_shape, size_kernel=5, stride=2, stddev=0.02, name='deconv2d'):
    with tf.variable_scope(name):
        stddev = np.sqrt(1.0 / (np.sqrt(input_map.get_shape()[-1].value * output_shape[-1]) * size_kernel ** 2))
        # filter : [height, width, output_channels, in_channels]
        kernel = tf.get_variable(
            name='w',
            shape=[size_kernel, size_kernel, output_shape[-1], input_map.get_shape()[-1]],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(stddev=stddev)
        )
        biases = tf.get_variable(
            name='b',
            shape=[output_shape[-1]],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0)
        )
        deconv = tf.nn.conv2d_transpose(input_map, kernel, strides=[1, stride, stride, 1], output_shape=output_shape)
        return tf.nn.bias_add(deconv, biases)
       

def lrelu(logits, leak=0.2):
    return tf.maximum(logits, leak*logits)


def concat_label(x, label, duplicate=1):
    x_shape = x.get_shape().as_list()
    if duplicate < 1:
        return x
    # duplicate the label to enhance its effect, does it really affect the result?
    label = tf.tile(label, [1, duplicate])
    label_shape = label.get_shape().as_list()
    if len(x_shape) == 2:
        return tf.concat(1, [x, label])
    elif len(x_shape) == 4:
        label = tf.reshape(label, [x_shape[0], 1, 1, label_shape[-1]])
        return tf.concat(3, [x, label*tf.ones([x_shape[0], x_shape[1], x_shape[2], label_shape[-1]])])


# def load_image(
#         image_path,  # path of a image
#         image_size=64,  # expected size of the image
#         image_value_range=(-1, 1),  # expected pixel value range of the image
#         is_gray=False,  # gray scale or color image
# ):
#     if is_gray:
#         image = imread(image_path, flatten=True).astype(np.float32)
#     else:
#         image = imread(image_path).astype(np.float32)
#     image = imresize(image, [image_size, image_size])
#     image = image.astype(np.float32) * (image_value_range[-1] - image_value_range[0]) / 255.0 + image_value_range[0]
#     return image


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
    img_h, img_w = batch_images.shape[1], batch_images.shape[2]
    print("img_h ", img_h)
    print("img_w ", img_w)
    frame = np.zeros([img_h * size_frame[0], img_w * size_frame[1], 3])
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
    train_percentage = 0.90

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

    x_train = x_train.astype('float32') / 255.

    x_test = x_test.astype('float32') / 255.

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


def print_images(x, title):
    max_images_per_frame=60
    count_frames = int(np.ceil(x.shape[0] / max_images_per_frame))
    max_images_sqrt = int(np.sqrt(max_images_per_frame))
    count = 0  # to iterate through the lists

    for i in range(count_frames):
        f, axarr = plt.subplots(max_images_sqrt, max_images_sqrt, figsize=(25, 25))
        for z in range(max_images_sqrt):
            for j in range(max_images_sqrt):  # per ogni riga righe
                if (count < x.shape[0]):
                    axarr[z, j].imshow(x[count])
                    axarr[z, j].set_title(title)
                    count += 1
                else:
                    break;
        plt.show()
        plt.close()

def shuffle(images, labels):
        n_samples=images.shape[0]
        perm = np.arange(n_samples)
        np.random.shuffle(perm)
        return images[perm], labels[perm]

def rotate_coronal_images(images):
    print("IMAGES SHAPE: ", images.shape)

    for i in range(images.shape[0]):
        images[i,:,:,:]=np.rot90(images[i,:,:,:], axes=[1,2]);

    return images