import numpy as np
import tensorflow as tf
import cv2

# from deepsense import neptune
import neptuner


channel_loss = neptuner.numeric_channel('loss')
channel_loss_style = neptuner.numeric_channel('loss_style')
channel_loss_content = neptuner.numeric_channel('loss_content')
channel_balance = neptuner.numeric_channel('content/style balance')

channel_image = neptuner.image_channel('channel_image')

content_style_balance = 1.0
def _change_content_style_balance_handler(csb):
    global content_style_balance
    content_style_balance = csb
    return True
neptuner.register_action(name='Change content/style balance', handler=_change_content_style_balance_handler)
# It would be nice to have a more specialized version of action registering:
# neptuner.register_action_change_value('Change content/style balance', content_style_balance)


def read_images(path_to_content, path_to_style, max_axis_size):
    content = cv2.imread(path_to_content).astype(np.float32)
    style = cv2.imread(path_to_style).astype(np.float32)
    shape = content.shape[0], content.shape[1]
    if max(shape) > max_axis_size:
        ratio = max_axis_size/float(max(shape))
        shape = int(round(shape[0]*ratio)), int(round(shape[1]*ratio))
        content = cv2.resize(content, shape[::-1])
    style = cv2.resize(style, shape[::-1])
    content = np.expand_dims(content, 0)
    style = np.expand_dims(style, 0)
    return content, style


def retrieve(img):
    img = np.squeeze(img, axis=(0,))
    img += [103.939, 116.779, 123.68]
    img = np.clip(img.round(), 0, 255).astype(np.uint8)
    return img


def conv2d(signal, num_outputs, kernel_size=[3, 3], stride=1, padding='SAME', activation_fn=tf.nn.relu, scope='Conv'):
    num_inputs = int(signal.get_shape()[-1])
    wshape = [kernel_size[0], kernel_size[1], num_inputs, num_outputs]
    stddev = np.sqrt(2./(num_inputs*kernel_size[0]*kernel_size[1]))
    with tf.variable_scope(scope):
        weights = tf.Variable(tf.truncated_normal(wshape, mean=0., stddev=stddev), dtype=tf.float32, name='weights')
        bias = tf.Variable(tf.constant(0., shape=[num_outputs]), dtype=tf.float32, name='bias')
        signal = tf.nn.conv2d(signal, weights, strides=[1, stride, stride, 1], padding=padding)
        signal = tf.nn.bias_add(signal, bias)
        signal = activation_fn(signal)
    return signal


def avg_pool(signal, kernel_size=[2, 2], stride=2, padding='SAME', name='AvgPool'):
    signal = tf.nn.avg_pool(signal, ksize=[1, kernel_size[0], kernel_size[1], 1],
                            strides=[1, stride, stride, 1], padding=padding, name=name)
    return signal


def gram_matrix(signal):
    _, h, w, d = map(int, signal.get_shape())
    V = tf.reshape(signal, shape=(h*w, d))
    G = tf.matmul(V, V, transpose_a=True)
    G /= 2*h*w*d
    return G


def get_stats(content, style):
    tf.reset_default_graph()

    assert content.shape == style.shape
    inputs = tf.placeholder(tf.float32, shape=content.shape, name='inputs')

    signal = inputs - [103.939, 116.779, 123.68]

    conv1_1 = conv2d(signal, 64, scope='conv1_1')
    conv1_2 = conv2d(conv1_1, 64, scope='conv1_2')
    pool1 = avg_pool(conv1_2, name='pool1')

    conv2_1 = conv2d(pool1, 128, scope='conv2_1')
    conv2_2 = conv2d(conv2_1, 128, scope='conv2_2')
    pool2 = avg_pool(conv2_2, name='pool2')

    conv3_1 = conv2d(pool2, 256, scope='conv3_1')
    conv3_2 = conv2d(conv3_1, 256, scope='conv3_2')
    conv3_3 = conv2d(conv3_2, 256, scope='conv3_3')
    pool3 = avg_pool(conv3_3, name='pool3')

    conv4_1 = conv2d(pool3, 512, scope='conv4_1')
    conv4_2 = conv2d(conv4_1, 512, scope='conv4_2')
    conv4_3 = conv2d(conv4_2, 512, scope='conv4_3')
    pool4 = avg_pool(conv4_3, name='pool4')

    conv5_1 = conv2d(pool4, 512, scope='conv5_1')
    conv5_2 = conv2d(conv5_1, 512, scope='conv5_2')
    conv5_3 = conv2d(conv5_2, 512, scope='conv5_3')
    pool5 = avg_pool(conv5_3, name='pool5')

    vgg16 = tf.train.Saver()

    stats = {}

    with tf.Session() as sess:
        vgg16.restore(sess, neptuner.params.path_to_model)
        stats['content_stats'] = sess.run(conv4_2, feed_dict={inputs: content})
        stats['style_stats1']  = sess.run(gram_matrix(conv1_1), feed_dict={inputs: style})
        stats['style_stats2']  = sess.run(gram_matrix(conv2_1), feed_dict={inputs: style})
        stats['style_stats3']  = sess.run(gram_matrix(conv3_1), feed_dict={inputs: style})
        stats['style_stats4']  = sess.run(gram_matrix(conv4_1), feed_dict={inputs: style})
        stats['style_stats5']  = sess.run(gram_matrix(conv5_1), feed_dict={inputs: style})

    return stats


def transfer_style(stats, img):
    tf.reset_default_graph()

    if neptuner.params.from_content:
        initial_image = img
    else:
        initial_image = tf.truncated_normal(img.shape, mean=0., stddev=1, dtype=tf.float32, seed=None)
    image = tf.Variable(initial_image, name='image')

    conv1_1 = conv2d(image, 64, scope='conv1_1')
    conv1_2 = conv2d(conv1_1, 64, scope='conv1_2')
    pool1 = avg_pool(conv1_2, name='pool1')

    conv2_1 = conv2d(pool1, 128, scope='conv2_1')
    conv2_2 = conv2d(conv2_1, 128, scope='conv2_2')
    pool2 = avg_pool(conv2_2, name='pool2')

    conv3_1 = conv2d(pool2, 256, scope='conv3_1')
    conv3_2 = conv2d(conv3_1, 256, scope='conv3_2')
    conv3_3 = conv2d(conv3_2, 256, scope='conv3_3')
    pool3 = avg_pool(conv3_3, name='pool3')

    conv4_1 = conv2d(pool3, 512, scope='conv4_1')
    conv4_2 = conv2d(conv4_1, 512, scope='conv4_2')
    conv4_3 = conv2d(conv4_2, 512, scope='conv4_3')
    pool4 = avg_pool(conv4_3, name='pool4')

    conv5_1 = conv2d(pool4, 512, scope='conv5_1')
    conv5_2 = conv2d(conv5_1, 512, scope='conv5_2')
    conv5_3 = conv2d(conv5_2, 512, scope='conv5_3')
    pool5 = avg_pool(conv5_3, name='pool5')

    content_style_balance_param = tf.placeholder(dtype=tf.float32, shape=[])

    loss_content = 3e-4 * tf.maximum(content_style_balance_param, 1e-3) * neptuner.params.content_intensity\
                   * tf.reduce_sum((stats['content_stats'] - conv4_2)**2)/2

    loss_style1 = tf.reduce_sum((stats['style_stats1'] - gram_matrix(conv1_1))**2)/2
    loss_style2 = tf.reduce_sum((stats['style_stats2'] - gram_matrix(conv2_1))**2)/2
    loss_style3 = tf.reduce_sum((stats['style_stats3'] - gram_matrix(conv3_1))**2)/2
    loss_style4 = tf.reduce_sum((stats['style_stats4'] - gram_matrix(conv4_1))**2)/2
    loss_style5 = tf.reduce_sum((stats['style_stats5'] - gram_matrix(conv5_1))**2)/2

    loss_style = (1.0 / tf.maximum(content_style_balance_param, 1e-3)) * neptuner.params.style_intensity\
                 * tf.add_n([loss_style1, loss_style2, loss_style3, loss_style4, loss_style5])/5

    loss = loss_content + loss_style

    train_op = tf.train.AdamOptimizer(neptuner.params.learning_rate).minimize(loss, var_list=[image])
    vgg16 = tf.train.Saver(tf.global_variables()[1:27])

    cfg = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    with tf.Session(config=cfg) as sess:
        sess.run(tf.global_variables_initializer())
        vgg16.restore(sess, neptuner.params.path_to_model)
        for step in xrange(neptuner.params.number_of_iterations):
            _, l, ls, lc = sess.run([train_op, loss, loss_style, loss_content],
                                    feed_dict={content_style_balance_param: content_style_balance})
            if step % 10 == 0:
                channel_loss.send(step, l)
                channel_loss_style.send(step, ls)
                channel_loss_content.send(step, lc)
                channel_balance.send(step, content_style_balance)
                if step % 100 == 0:
                    img = retrieve(sess.run(image))
                    channel_image.send_cv2(x=step, name=step, description=step, img=img)
        img = retrieve(sess.run(image))
        cv2.imwrite('final.jpg', img)
        channel_image.send_cv2(x=neptuner.params.number_of_iterations, name='final', description='final', img=img)


def main():
    content, style = read_images(neptuner.params.path_to_images + neptuner.params.content_image_name,
                                 neptuner.params.path_to_images + neptuner.params.style_image_name,
                                 neptuner.params.max_img_size)
    stats = get_stats(content, style)
    transfer_style(stats, content)


if __name__ == "__main__":
    main()
