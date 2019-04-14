import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from lib.data_loader.data_loader import Flowers102DataLoader
from lib.utils.config import ConfigReader, TrainNetConfig, DataConfig
from lib.googlenet.inception_v1 import InceptionV1

FLAGS = None

def plot_image_test(image_batch, label_batch, train_config):
    with tf.Session() as sess:
        i = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop() and i < 1:
                img, label = sess.run([image_batch, label_batch])
                # just test one batch
                for j in np.arange(train_config.batch_size):
                    print('label: %d' % label[j])
                    plt.imshow(img[j, :, :, :])
                    plt.show()
                    pass
                i += 1

        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
        coord.join(threads)


def train(_):
    # Configure distibuted machines
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    # Create and start a server
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    
    config_reader = ConfigReader('experiments/configs/inception_v1.yml')
    train_config = TrainNetConfig(config_reader.get_train_config())
    data_config = DataConfig(config_reader.get_train_config())

    train_log_dir = './logs/train/'
    val_log_dir = './logs/val/'

    if not os.path.exists(train_log_dir):
        os.makedirs(train_log_dir)
    if not os.path.exists(val_log_dir):
        os.makedirs(val_log_dir)

    if FLAGS.job_name == "ps":
        server.join()
    else:
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" %FLAGS.task_index,
            cluster = cluster)):
            net = InceptionV1(train_config, FLAGS.task_index)

            with tf.name_scope('input'):
                train_loader = Flowers102DataLoader(data_config, is_train=True, is_shuffle=True)
                train_image_batch, train_label_batch = train_loader.generate_batch()
                val_loader = Flowers102DataLoader(data_config, is_train=False, is_shuffle=False)
                val_image_batch, val_label_batch = val_loader.generate_batch()

            train_op = net.build_model()
            summaries = net.get_summary()

            saver = tf.train.Saver(tf.global_variables())
            summary_op = tf.summary.merge(summaries)

            init = tf.global_variables_initializer()
            #sess = tf.Session()
            sess = tf.train.MonitoredTrainingSession(
                master = server.target,
                is_chief=(FLAGS.task_index==0),
                hooks=net.sync_replicas_hook)
            #sess.run(init)

            #net.load_with_skip(train_config.pre_train_weight, sess, ['loss3_classifier'])

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
            val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)

            step = 0
            try:
                #for step in np.arange(train_config.max_step):
                while not sess.should_stop():
                    if coord.should_stop():
                        break

                    train_image, train_label = sess.run([train_image_batch, train_label_batch])
                    _, train_loss, train_acc = sess.run([train_op, net.loss, net.accuracy],
                                                        feed_dict={net.x: train_image, net.y: train_label})

                    if step % (50*train_config.update_delays) == 0 or step + 1 == train_config.max_step:
                        print('===TRAIN===: Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, train_loss, train_acc))
                        summary_str = sess.run(summary_op, feed_dict={net.x: train_image, net.y: train_label})
                        train_summary_writer.add_summary(summary_str, step/train_config.update_delays)
                    if step % (200*train_config.update_delays) == 0 or step + 1 == train_config.max_step:
                        pass
                        val_image, val_label = sess.run([val_image_batch, val_label_batch])
                        val_loss, val_acc = sess.run([net.loss, net.accuracy], feed_dict={net.x: val_image, net.y: val_label})
                        print('====VAL====: Step %d, val loss = %.4f, val accuracy = %.4f%%' % (step, val_loss, val_acc))
                        summary_str = sess.run(summary_op, feed_dict={net.x: train_image, net.y: train_label})
                        val_summary_writer.add_summary(summary_str, step/train_config.update_delays)
                    if step % 2000 == 0 or step + 1 == train_config.max_step:
                        checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                        #saver.save(sess, checkpoint_path, global_step=step)
                    step += 1

            except tf.errors.OutOfRangeError:
                print('===INFO====: Training completed, reaching the maximum number of steps')
            finally:
                coord.request_stop()

            coord.join(threads)
            sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ps_hosts", type=str, default="", help="Parameter server host:port")
    parser.add_argument("--worker_hosts", type=str, default="", help="Worker server host:port")
    parser.add_argument("--job_name", type=str, default="", help="ps or worker")
    parser.add_argument("--task_index", type=int, default=0, help="Index of task within the job")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=train)
