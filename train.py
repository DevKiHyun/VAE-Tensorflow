import tensorflow as tf
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

sys.path.append('.')
import vae

mnist = input_data.read_data_sets("./data/MNIST_data/", one_hot=True)

def train(config):
    '''
     SETTING HYPERPARAMETER (DEFAULT)
     '''
    training_epoch = config.training_epoch
    z_dim = config.z_dim
    batch_size = config.batch_size
    n_data = mnist.train.num_examples
    total_batch = int(mnist.train.num_examples / batch_size)
    total_iteration = training_epoch * total_batch

    # Build Network
    VAE = vae.VAE(config)
    VAE.build()
    # Optimize Network
    VAE.optimize(config)

    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    saver = tf.train.Saver()

    print("Total the number of Data : " + str(n_data))
    print("Total Step per 1 Epoch: {}".format(total_batch))
    print("The number of Iteration: {}".format(total_iteration))

    for epoch in range(training_epoch):
        avg_cost = 0
        avg_recons = 0
        avg_regular = 0
        for i in range(total_batch):
            batch_xs, _ = mnist.train.next_batch(batch_size)

            _cost, _, _recons, _regular = sess.run([VAE.cost, VAE.optimizer, VAE.recons, VAE.regular], feed_dict={VAE.X: batch_xs})
            avg_cost += _cost / total_batch
            avg_recons += _recons / total_batch
            avg_regular += _regular / total_batch

        if epoch % 10 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost),
                  'Recons_Loss =', '{:.9f}'.format(avg_recons),
                  'Regular_Loss =', '{:.9f}'.format(avg_regular))

    print("Training Complete!")

    save_dir = './mode_z_dim_{}/'.format(z_dim)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    save_path = '{}VAE.ckpt'.format(save_dir)
    saver.save(sess, save_path)
    print("Saved Model")

    return VAE, sess

def run_train(config):
    result_dir = './result'
    if not os.path.exists(result_dir): os.makedirs(result_dir)

    '''
    Train 20-dimension model
    '''
    VAE_20dim, sess = train(config)

    #Just reconstruction image
    n_sample = 8
    sampled_original_images = mnist.test.images[:n_sample]

    result_images = sess.run(VAE_20dim.output, feed_dict={VAE_20dim.X: sampled_original_images})
    images_list = [sampled_original_images, result_images]

    columns = 8
    rows = 2
    fig, axis = plt.subplots(rows, columns)
    for i in range(columns):
        for j in range(rows):
            axis[j, i].imshow(images_list[j][i].reshape(28, 28))
    plt.savefig('{}/reconstruction.png'.format(result_dir))

    #Generation from Z distribution
    z = np.random.normal(0, 1, size=[n_sample,VAE_20dim.z_dim])
    x_hat = sess.run(VAE_20dim.output, feed_dict={VAE_20dim.sampled_z: z})
    images_list = [x_hat]
    columns = 8
    rows = 1
    fig, axis = plt.subplots(rows, columns)
    for i in range(columns):
        for j in range(rows):
            axis[j+i].imshow(images_list[j][i].reshape(28, 28))
    plt.savefig('{}/generation.png'.format(result_dir))
    plt.close()

    tf.reset_default_graph()

    '''
    Train 2-dimension model
    '''
    config.z_dim = 2
    VAE_2dim, sess = train(config)

    # Manifold 2-dimension
    n_sample = 4000
    sampled_original_images, sampled_original_labels = mnist.test.next_batch(n_sample)
    z = sess.run(VAE_2dim.sampled_z, feed_dict={VAE_2dim.X:sampled_original_images})

    fig = plt.figure()
    plt.scatter(z[:, 0], z[:, 1], c=np.argmax(sampled_original_labels, 1))
    plt.colorbar()
    plt.grid()
    plt.savefig('{}/manifold.png'.format(result_dir))
    plt.close(fig)

    # Manifold 2-dimension walking
    x_space = np.linspace(start=-2, stop=2, num=20)
    y_space = np.linspace(start=-2, stop=2, num=20)
    result_size = [28*20, 28*20]
    result_image = np.empty(shape=result_size)
    interval = 28

    for x_index, x in enumerate(x_space):
        for y_index, y in enumerate(y_space):
            z = np.expand_dims([x,y], axis=0)
            x_hat = sess.run(VAE_2dim.output, feed_dict={VAE_2dim.sampled_z:z})
            x_hat = np.squeeze(x_hat, axis=0)

            height_start = y_index*interval
            height_end = (y_index+1)*interval
            width_start = x_index*interval
            width_end = (x_index+1)*interval

            result_image[height_start:height_end, width_start:width_end] = x_hat.reshape(28,28)

    fig = plt.figure()
    plt.imshow(result_image, cmap="gray")
    plt.savefig('{}/walking.png'.format(result_dir))
    plt.close(fig)