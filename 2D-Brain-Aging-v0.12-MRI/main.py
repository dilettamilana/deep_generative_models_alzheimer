import tensorflow as tf
from BrainAging import BrainAging
from WGAN import WGAN
flags = tf.app.flags
flags.DEFINE_integer(flag_name='epochs', default_value=100, docstring='number of epochs')
flags.DEFINE_float(flag_name='learning_rate', default_value=0.0007, docstring='')
flags.DEFINE_float(flag_name='G_img_param', default_value=1, docstring='')
flags.DEFINE_float(flag_name='E_z_param', default_value=0.00, docstring='')
flags.DEFINE_float(flag_name='tv_param', default_value=0.00, docstring='')
#flags.DEFINE_integer(flag_name='selected_slice', default_value=None, docstring='')
flags.DEFINE_boolean(flag_name='is_train', default_value=True, docstring='training mode')
flags.DEFINE_string(flag_name='testdir', default_value='None', docstring='dir for testing images')


FLAGS = flags.FLAGS

def main(_):

    # print settings
    import pprint
    pprint.pprint(FLAGS.__flags)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        model = BrainAging(
            session,  # TensorFlow session
            epochs=FLAGS.epochs,
            is_training=FLAGS.is_train,  # flag for training or testing mode
            learning_rate=FLAGS.learning_rate,  # learning rate of optimizer
            G_img_param=FLAGS.G_img_param,
            E_z_param=FLAGS.E_z_param,
            tv_param=FLAGS.tv_param,
        )

        #corretto e funziona, 11/11
        # model = BrainAging(
        #     session,  # TensorFlow session
        #     epochs=FLAGS.epochs,
        #     is_training=FLAGS.is_train,  # flag for training or testing mode
        #     learning_rate=FLAGS.learning_rate,  # learning rate of optimizer
        #     G_img_param=FLAGS.G_img_param,
        #     E_z_param=FLAGS.E_z_param,
        #     tv_param=FLAGS.tv_param,
        # )
        if FLAGS.is_train:
            print('\n\tTraining Mode')
            model.train()
        else:
            print('\n\tTesting Mode')
            model.custom_test(
                testing_samples_dir=FLAGS.testdir + '/*jpg'
            )

if __name__ == '__main__':

    tf.app.run()