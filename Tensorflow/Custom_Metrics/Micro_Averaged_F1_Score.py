import tensorflow as tf

class Micro_Averaged_F1_Score (tf.keras.metrics.Metric):
  def __init__(self, name="micro_averaged_f1_score", **kwargs):
    tf.keras.metrics.Metric.__init__(self, name=name, **kwargs)
    self.false_positive = self.add_weight(name="fn", initializer="zeros")
    self.false_negative = self.add_weight(name="fn", initializer="zeros")
    self.true_positive = self.add_weight(name="fn", initializer="zeros")
  
  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.cast(y_pred, tf.bool)

    false_positive = tf.math.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True)), tf.float32))
    false_negative = tf.math.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False)), tf.float32))
    true_positive = tf.math.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True)), tf.float32))

    self.false_positive.assign_add(false_positive)
    self.false_negative.assign_add(false_negative)
    self.true_positive.assign_add(true_positive)
  
  def result(self):
    recall = (self.true_positive / (self.true_positive + self.false_negative))
    precision = (self.true_positive / (self.true_positive + self.false_positive))
    micro_f1_score = (2 * recall * precision) / (recall + precision + 1e-7)
    return micro_f1_score
