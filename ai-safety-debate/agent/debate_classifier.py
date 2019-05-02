import tensorflow as tf


class DebateClassifier:
    def __init__(self, sample_shape=[28, 28], model_dir=None, log_dir=None):
        self.sample_shape = sample_shape
        self.estimator = tf.estimator.Estimator(
            model_fn=self.model_fn, model_dir=model_dir
        )

    def train(self, np_batch, labels, loss_weights):
        batch_size = np_batch.shape[0]
        # Train for one batch
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np_batch, "loss_weights": loss_weights},
            y=labels,
            batch_size=batch_size,
            num_epochs=1,
            shuffle=True,
        )
        self.estimator.train(input_fn=train_input_fn)

    def evaluate_accuracy(self, eval_data, eval_labels):
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False
        )
        eval_results = self.estimator.evaluate(input_fn=eval_input_fn)
        return eval_results

    def predict(self, sample):
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": sample}, y=None, num_epochs=1, shuffle=False
        )
        eval_results = self.estimator.predict(input_fn=eval_input_fn)
        return eval_results

    def model_fn(self, features, labels, mode):
        # Input Layer
        input_layer = tf.reshape(features["x"], [-1] + self.sample_shape + [1])

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

        softmax_tensor = tf.nn.softmax(logits, name="softmax_tensor")
        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": softmax_tensor,
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        if mode == tf.estimator.ModeKeys.TRAIN:
            loss_weights = features["loss_weights"]
            loss = tf.losses.sparse_softmax_cross_entropy(
                labels=labels, logits=logits, weights=loss_weights
            )
        else:
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
