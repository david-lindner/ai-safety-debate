import tensorflow as tf
import numpy as np
from .judge import Judge

tf.logging.set_verbosity(tf.logging.INFO)

class MNISTJudge(Judge):
    """
    Sparse MNIST classifier, based on
    https://www.tensorflow.org/tutorials/estimators/cnn#building_the_cnn_mnist_classifier
    """

    def __init__(self, N_pixels):
        self.N_pixels = N_pixels
        self.batch_size = 128
        self.shape = [None,28,28,2]

        # Load training and eval data
        (
            (train_data, train_labels),
            (eval_data, eval_labels),
        ) = tf.keras.datasets.mnist.load_data()

        self.train_data = train_data / np.float32(255)
        self.train_labels = train_labels.astype(np.int32)  # not required

        self.eval_data = eval_data / np.float32(255)
        self.eval_labels = eval_labels.astype(np.int32)  # not required

        # Create the Estimator
        self.estimator = tf.estimator.Estimator(
            model_fn=self.cnn_model_fn
        )  # , model_dir="/tmp/mnist_convnet_model")

        # Set up logging for predictions
        tensors_to_log = {"probabilities": "softmax_tensor"}

        self.logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50
        )

    def mask_image_batch(self, image_batch):
        shape = tf.shape(image_batch)
        batch_flat = tf.reshape(image_batch, (shape[0], shape[1] * shape[2]))
        mask_flat = self.mask_batch(batch_flat)
        return tf.reshape(mask_flat, (shape[0], shape[1], shape[2], 2))

    def cnn_model_fn(self, features, labels, mode):
        """Model function for CNN."""
        # Input Layer
        if "masked_x" in features:
            input_layer = features["masked_x"]
        else:
            input_layer = self.mask_image_batch(features["x"])

        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
        )

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
        )

        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Dense Layer
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
            inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN
        )

        # Logits Layer
        logits = tf.layers.dense(inputs=dropout, units=10)

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
            train_op = optimizer.minimize(
                loss=loss, global_step=tf.train.get_global_step()
            )
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"]
            )
        }
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
        )

    # overrides Judge
    def evaluate_debate(self, input, answers):
        input = np.reshape(input, (1, 28, 28, 2))
        return super().evaluate_debate(input, answers)
