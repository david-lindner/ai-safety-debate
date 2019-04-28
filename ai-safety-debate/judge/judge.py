import tensorflow as tf
import numpy as np

class Judge:

    def mask_batch(self, batch):
        """
        Create mask for each image in a batch, that contains N_pixels nonzero pixels
        of the image. Combine this with the image to create the input for the DNN.
        """
        shape = tf.shape(batch)
        p = tf.random_uniform(shape, 0, 1)
        nonzero_p = tf.where(batch > 0, p, tf.zeros_like(p))
        _, indices = tf.nn.top_k(nonzero_p, self.N_pixels)
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

        # train one step and display the probabilties
        self.classifier.train(
            input_fn=train_input_fn, steps=1, hooks=[self.logging_hook]
        )

        self.classifier.train(input_fn=train_input_fn, steps=n_steps)

    def evaluate_accuracy(self):
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": self.eval_data}, 
            y=self.eval_labels, 
            num_epochs=1, 
            shuffle=False
        )
        eval_results = self.classifier.evaluate(input_fn=eval_input_fn)
        return eval_results

    def evaluate_debate(self, input, answers):
        assert len(answers) == 2
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"masked_x":input}, shuffle=False
        )
        output = self.classifier.predict(eval_input_fn)
        prediction = next(output)
        probs = prediction["probabilities"]
        if probs[answers[0]] > probs[answers[1]]:
            return 0
        else:
            return 1