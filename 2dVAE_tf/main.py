import tensorflow as tf
from vae import VariationalAutoencoder
from autoencoder import Autoencoder
from tensorflow.python import debug as tf_debug

flags = tf.app.flags
flags.DEFINE_integer(flag_name='epochs', default_value=50, docstring='number of epochs')
flags.DEFINE_float(flag_name='learning_rate', default_value=0.0001, docstring='')
flags.DEFINE_boolean(flag_name='is_train', default_value=True, docstring='training mode')
flags.DEFINE_string(flag_name='testdir', default_value='None', docstring='dir for testing images')
flags.DEFINE_string(flag_name='training_notes', default_value=' ', docstring='training notes')
flags.DEFINE_string(flag_name='model', default_value='Autoencoder', docstring='model to use for training')


#flags.DEFINE_string(flag_name='dataset', default_value='UTKFace', docstring='dataset name')
#flags.DEFINE_string(flag_name='savedir', default_value='save', docstring='dir for saving training results')
#

FLAGS = flags.FLAGS

def main(_):

    # print settings
    import pprint
    pprint.pprint(FLAGS.__flags)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess=tf.Session(config=config)

    with tf_debug.LocalCLIDebugWrapperSession(sess) as session:
        #session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        model_class = globals()[FLAGS.model]
        model = model_class(
            session,  # TensorFlow session
            training_notes=FLAGS.training_notes,
            epochs=FLAGS.epochs,
            is_training=FLAGS.is_train,  # flag for training or testing mode
            learning_rate=FLAGS.learning_rate,  # learning rate of optimizer
        )
        if FLAGS.is_train:
            print('\n\tTraining Mode')
            model.train_vae()
            model.train_classifier()
        else:
            print('\n\tTesting Mode')
            model.custom_test(
                testing_samples_dir=FLAGS.testdir + '/*jpg'
            )

if __name__ == '__main__':

    tf.app.run()
