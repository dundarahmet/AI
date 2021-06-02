import tensorflow as tf

class Macro_Averaged_F1_Score (tf.keras.metrics.Metric):
  def __init__(self, name="macro_averaged_f1_score", **kwargs):
    tf.keras.metrics.Metric.__init__(self, name=name, **kwargs)
    self.macro_f1_score = self.add_weight(name="macro_f1", initializer="zeros")
    self.counter = self.add_weight(name="counter", initializer='zeros')
  
  def update_state(self, y_true, y_pred, sample_weight=None):
    self.counter.assign_add(1)

    false_positive = tf.math.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True)), tf.float32), axis=0)
    false_negative = tf.math.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False)), tf.float32), axis=0)
    true_positive = tf.math.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True)), tf.float32), axis=0)

    macro_precision = true_positive / ((true_positive + false_positive) + 1e-8)
    macro_recall = true_positive / ((true_positive + false_negative) + 1e-8)
    self.macro_f1_score.assign_add(tf.reduce_sum((2 * macro_precision * macro_recall / (macro_recall + macro_precision + 1e-8))) / y_true.shape[-1])
  
  def result(self):
    return self.macro_f1_score / self.counter
