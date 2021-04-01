import tensorflow as tf
import tensorflow.keras

 
from tensorflow.keras import backend as K
from scipy import stats
from sklearn.metrics import cohen_kappa_score
from tensorflow.python.ops import math_ops

from tensorflow.python.keras import backend 
 

tf.config.experimental_run_functions_eagerly(True)


def KendallTau(y_true, y_pred): 
    return stats.kendalltau(y_true.numpy(), y_pred.numpy())[0]    
 

#def CohenKappa(y_true, y_pred, y_pow=2, eps=1e-10, N=NUM_CLASSES, bsize=BATCH_SIZE, name='kappa'):
def CohenKappa(y_true, y_pred, N, bsize, y_pow=2, eps=1e-10, name='kappa'):
    """A continuous differentiable approximation of discrete kappa loss.
        Args:
            y_pred: 2D tensor or array, [batch_size, num_classes]
            y_true: 2D tensor or array,[batch_size, num_classes]
            y_pow: int,  e.g. y_pow=2
            N: typically num_classes of the model
            bsize: batch_size of the training or validation ops
            eps: a float, prevents divide by zero
            name: Optional scope/name for op_scope.
        Returns:
            A tensor with the kappa loss."""

    with tf.name_scope(name):
        
        y_true = tf.cast(y_true,dtype='float') 
        repeat_op = tf.cast(tf.tile(tf.reshape(tf.range(0, N), [N, 1]), [1, N]), dtype='float')
        repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))
        weights = repeat_op_sq / tf.cast((N - 1) ** 2, dtype='float')

        pred_ = y_pred ** y_pow
        try:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [-1, 1]))
        except Exception:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [bsize, 1]))

        hist_rater_a = tf.reduce_sum(pred_norm, 0)
        hist_rater_b = tf.reduce_sum(y_true, 0)

        conf_mat = tf.matmul(tf.transpose(pred_norm), y_true)

        nom = tf.reduce_sum(weights * conf_mat)
        denom = tf.reduce_sum(weights * tf.matmul(
            tf.reshape(hist_rater_a, [N, 1]), 
            tf.reshape(hist_rater_b, [1, N])) / tf.cast(bsize, dtype='float'))

        return nom / (denom + eps)
    


#Pearson correlation coefficlent : $r_{xy} = \frac{\sum\left((x-\overline{x})(y-\overline{y})\right)}{\sqrt{\sum(x-\overline{x})^2\sum(y-\overline{y})^2}} $

def Pearson(y_true, y_pred, axis=-2):
    """Metric returning the Pearson correlation coefficient of two tensors over some axis, default -2."""
    x = tf.convert_to_tensor(y_true)
    y = math_ops.cast(y_pred, x.dtype)
    #print(type(x))#<class 'tensorflow.python.framework.ops.Tensor'>
    #print(type(y))#<class 'tensorflow.python.framework.ops.Tensor'>

    n = tf.cast(tf.shape(x)[axis], x.dtype)
    xsum = tf.reduce_sum(x, axis=axis)
    ysum = tf.reduce_sum(y, axis=axis)
    xmean = xsum / n
    ymean = ysum / n 
    xvar = tf.reduce_sum( tf.math.squared_difference(x, xmean), axis=axis)
    yvar = tf.reduce_sum( tf.math.squared_difference(y, ymean), axis=axis)
    
    cov = tf.reduce_sum( (x - xmean) * (y - ymean), axis=axis)
    corr = cov / tf.sqrt(xvar * yvar)
    return tf.constant(1.0, dtype=x.dtype) - corr

