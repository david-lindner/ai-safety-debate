import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

class Judge:
    def __init__(self, N_to_mask, restore_model_from, save_model_as):
        self.N_to_mask = N_to_mask

        # Create the Estimator
        self.estimator = tf.estimator.Estimator(
            model_fn = self.model_fn,
            model_dir = save_model_as,
            warm_start_from = restore_model_from
        )

        # Subclasses need to implement a model_fn with a "softmax_tensor"

        # Set up logging for predictions
        tensors_to_log = {"probabilities": "softmax_tensor"}

        self.logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50
        )

        self.update_predictor()

    def update_predictor(self):
        self.predictor = tf.contrib.predictor.from_estimator(
            self.estimator,
            tf.estimator.export.build_raw_serving_input_receiver_fn(
                {"masked_x":tf.placeholder(
                    'float32', 
                    shape=[None]+self.shape[1:])}
            )
        )

    def mask_image_batch(self, image_batch):
        shape = tf.shape(image_batch)
        batch_flat = tf.reshape(image_batch, (shape[0], shape[1] * shape[2]))
        mask_flat = self.mask_batch(batch_flat)
        return tf.reshape(mask_flat, (shape[0], shape[1], shape[2], 2))

    def mask_batch(self, batch):
        """
        Create mask for each image in a batch, that contains N_to_mask nonzero pixels
        of the image. Combine this with the image to create the input for the DNN.
        """
        shape = tf.shape(batch)
        p = tf.random_uniform(shape, 0, 1)
        nonzero_p = tf.where(batch > 0, p, tf.zeros_like(p))
        _, indices = tf.nn.top_k(nonzero_p, self.N_to_mask)
        mask = tf.one_hot(indices, shape[1], axis=1)
        mask = tf.reduce_sum(mask, axis=2)
        return tf.stack((mask, mask * batch), 2)

    def train(self, n_steps):
        # Train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": self.train_data},
            y=self.train_labels,
            batch_size=self.batch_size,
            num_epochs=None,
            shuffle=True,
        )

        self.estimator.train(input_fn=train_input_fn, steps=n_steps)

        self.update_predictor()

    def evaluate_accuracy(self):
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": self.eval_data}, y=self.eval_labels, num_epochs=1, shuffle=False
        )
        eval_results = self.estimator.evaluate(input_fn=eval_input_fn)
        return eval_results

    def evaluate_debate(self, input, answers):
        assert len(answers) == 2
        input = np.reshape(input, self.shape) # needed for images
        prediction = self.predictor({"masked_x": input})
        probs = prediction["probabilities"][0]
        if probs[answers[0]] > probs[answers[1]]:
            return 0
        else:
            return 1
