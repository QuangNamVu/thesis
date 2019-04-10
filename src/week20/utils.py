import numpy as np
import tensorflow as tf

logc = np.log(2. * np.pi)
c = - 0.5 * np.log(2 * np.pi)


def tf_normal_logpdf(x, mu, log_sigma_sq):
    return (- 0.5 * logc - log_sigma_sq / 2. - tf.div(tf.square(tf.subtract(x, mu)), 2 * tf.exp(log_sigma_sq)))


def tf_stdnormal_logpdf(x):
    return (- 0.5 * (logc + tf.square(x)))


def tf_gaussian_ent(log_sigma_sq):
    return (- 0.5 * (logc + 1.0 + log_sigma_sq))


def tf_gaussian_marg(mu, log_sigma_sq):
    return (- 0.5 * (logc + (tf.square(mu) + tf.exp(log_sigma_sq))))


def tf_binary_xentropy(x, y, const=1e-10):
    return - (x * tf.log(tf.clip_by_value(y, const, 1.0)) + \
              (1.0 - x) * tf.log(tf.clip_by_value(1.0 - y, const, 1.0)))


def feed_numpy_semisupervised(num_lab_batch, num_ulab_batch, x_lab, y, x_ulab):
    size = x_lab.shape[0] + x_ulab.shape[0]
    batch_size = num_lab_batch + num_ulab_batch
    count = int(size / batch_size)

    dim = x_lab.shape[1]

    for i in range(count):
        start_lab = i * num_lab_batch
        end_lab = start_lab + num_lab_batch
        start_ulab = i * num_ulab_batch
        end_ulab = start_ulab + num_ulab_batch

        yield [x_lab[start_lab:end_lab, :dim / 2], x_lab[start_lab:end_lab, dim / 2:dim], y[start_lab:end_lab],
               x_ulab[start_ulab:end_ulab, :dim / 2], x_ulab[start_ulab:end_ulab, dim / 2:dim]]


def feed_numpy(x, x_con, batch_size, y=None):
    size = x.shape[0]
    count = size // batch_size

    if y is None:
        for i in range(count):
            start = i * batch_size
            end = start + batch_size
            yield x[start:end], x_con[start:end]
    else:
        for i in range(count):
            start = i * batch_size
            end = start + batch_size

            yield x[start:end], x_con[start:end], y[start:end]


def print_metrics(epoch, *metrics):
    # print(25 * '-')
    print('%3d,' % (epoch), end=""),
    for metric in metrics:
        print(' %s:, %.4f,' % (metric[0], metric[1]), end=""),
    print()
    # print(25 * '-')