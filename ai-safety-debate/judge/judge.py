import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.WARN)


class Judge:
    def __init__(self, N_to_mask, model_dir):
        self.N_to_mask = N_to_mask

        # Create the Estimator
        self.estimator = tf.estimator.Estimator(
            model_fn=self.model_fn, model_dir=model_dir
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
                {"masked_x": tf.placeholder(shape=[None, 28, 28, 2], dtype=tf.float32)}
            ),
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

    def evaluate_accuracy_using_predictor(self):
        """
        Evaluates the test set accuracy using the tensorflow predictor instead
        of the estimator. Can be useful for debugging.
        """
        correct = 0
        count = 0
        for i in range(len(self.eval_labels)):
            # print(i)
            image = self.eval_data[i].flat
            mask = np.zeros_like(image)
            while mask.sum() < self.N_to_mask:
                a = np.random.randint(mask.shape[0])
                if image[a] > 0:
                    mask[a] = 1
            input = np.stack((mask, image * mask), axis=1)
            input = np.reshape(input, self.shape)
            prediction = self.predictor({"masked_x": input})
            probs = prediction["probabilities"][0]
            pred_label = np.argmax(probs)
            count += 1
            if pred_label == self.eval_labels[i]:
                correct += 1
            # print(correct / count)
        return correct / count

    def evaluate_debate(self, input, answers):
        assert len(answers) == 2
        input = np.reshape(input, self.shape)  # needed for images
        prediction = self.predictor({"masked_x": input})
        probs = prediction["probabilities"][0]
        # print("probs", probs)
        if probs[answers[0]] > probs[answers[1]]:
            return 0
        else:
            return 1
