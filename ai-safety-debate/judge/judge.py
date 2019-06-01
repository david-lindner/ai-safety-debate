import time
import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)


class Judge:
    def __init__(
        self,
        N_to_mask,
        estimator_model_dir=None,
        predictor_saved_model_dir=None,
        binary_rewards=True,
    ):
        self.N_to_mask = N_to_mask
        self.binary_rewards = binary_rewards
        self.evaluate_debate_time = 0
        self.evaluate_debate_counter = 0

        if estimator_model_dir is not None or predictor_saved_model_dir is None:
            # Create the Estimator
            try:
                self.estimator = tf.estimator.Estimator(
                    model_fn=self.model_fn,
                    model_dir=estimator_model_dir,  # directory to restore model from and save model to
                    # Only the latest checkpoint is saved, so you don't have to upload/download as much data
                    config=tf.estimator.RunConfig(keep_checkpoint_max=1),
                )
            except AttributeError:
                raise Exception("Subclass needs to define a model_fn")

            # Create the predictor from the present model. Important when restoring a model.
            self.update_predictor()
        else:
            print(
                "Warning: Created judge without initializing the estimator. "
                "The judge will not be able to train."
            )
            self.predictor = tf.contrib.predictor.from_saved_model(
                predictor_saved_model_dir
            )
            self.estimator = None

    def reset_timing(self):
        self.evaluate_debate_time = 0
        self.evaluate_debate_counter = 0

    def _check_estimator_available(self):
        if not self.estimator:
            raise Exception(
                "Estimator not available. Make sure you specify "
                "a path to load the estimator from or you do not specify a path "
                "to load the predictor from."
            )

    def make_serving_input_receiver_fn(self):
        # The serving input receiver fn is witchcraft, which I don't quite understand.
        # It's supposed to set up the data in a way that tensorflow can handle.
        return tf.estimator.export.build_raw_serving_input_receiver_fn(
            # The input is a dictionary that corresponds to the data we feed the predictor.
            # Each key stores a tensor that is replaced by data when the predictor is used.
            {
                "masked_x": tf.placeholder(
                    shape=[None, 28, 28, 2], dtype=tf.float32, name="masked_x"
                )
            }
        )

    def update_predictor(self):
        self._check_estimator_available()
        # Predictors are used to get predictions fast once the model has been trained.
        # We create it from an estimator.
        self.predictor = tf.contrib.predictor.from_estimator(
            self.estimator, self.make_serving_input_receiver_fn()
        )

    def mask_image_batch(self, image_batch):
        """
        Takes a batch of two-dimensional images, reshapes them, runs them through
        mask_batch, reshapes them back to images, and returns them.
        """
        shape = tf.shape(image_batch)
        batch_flat = tf.reshape(image_batch, (shape[0], shape[1] * shape[2]))
        mask_flat = self.mask_batch(batch_flat)
        return tf.reshape(mask_flat, (shape[0], shape[1], shape[2], 2))

    def mask_batch(self, batch):
        """
        Create mask for each feature-vector in a batch, that contains N_to_mask nonzero features
        of the input vector. Combine this with the vector to create the input for the DNN.
        """
        shape = tf.shape(batch)
        p = tf.random_uniform(shape, 0, 1)
        nonzero_p = tf.where(batch > 0, p, tf.zeros_like(p))
        _, indices = tf.nn.top_k(nonzero_p, self.N_to_mask)
        mask = tf.one_hot(indices, shape[1], axis=1)
        mask = tf.reduce_sum(mask, axis=2)
        return tf.stack((mask, mask * batch), 2)

    def train(self, n_steps):
        self._check_estimator_available()
        # Train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": self.train_data},
            y=self.train_labels,
            batch_size=self.batch_size,
            num_epochs=None,
            shuffle=True,
        )
        self.estimator.train(input_fn=train_input_fn, steps=n_steps)

        # Replace the old predictor with one created from the new estimator
        self.update_predictor()

    def evaluate_accuracy(self):
        if not self.estimator:
            return self.evaluate_accuracy_using_predictor()
        # Evaluate the accuracy on all the eval_data
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

    def evaluate_debate(self, input, initial_statements):
        """
        Returns the utility of player 0.
        If self.binary_rewards is true, it returns 1 if they win, -1 if they lose.
        Otherwise, it returns the difference between the probability assigned to the
        first players label and the second players label.
        """
        t = time.time()
        assert len(initial_statements) == 2
        input = np.reshape(
            input, self.shape
        )  # reshapes vectors into images, if appropriate
        prediction = self.predictor({"masked_x": input})
        probs = prediction["probabilities"][0]
        # print("probs", probs)

        # Initial statement of None corresponds to agents that have not precommited.
        # The unrestricted ( = non-precommited) player gets the probability of the best non-taken label
        # this is weird and unintuitive, you should instead either run all 9 debates and pick the best one,
        # or give the unrestricted player the sum of the non-taken labels. The latter is too hard for the pre-commited
        # player, the former takes too long. So we do this weird thing as a cheaper approximation of the former.
        # But beware: it is weird!
        if initial_statements == [None, None]:
            raise Exception("At least one agent has to make a claim!")
        elif initial_statements[0] is None:
            prob_pl_1 = probs[initial_statements[1]]
            probs[initial_statements[1]] = 0  # set to 0 to get max of the other labels
            utility = max(probs) - prob_pl_1
        elif initial_statements[1] is None:
            prob_pl_0 = probs[initial_statements[0]]
            probs[initial_statements[0]] = 0  # set to 0 to get max of the other labels
            utility = prob_pl_0 - max(probs)
        else:
            utility = probs[initial_statements[0]] - probs[initial_statements[1]]

        # convert to binary rewards, breaking ties in favor of player 1 (because whatever)
        if self.binary_rewards:
            if utility >= 0:
                utility = 1
            else:
                utility = -1

        self.evaluate_debate_time += time.time() - t
        self.evaluate_debate_counter += 1
        if (self.evaluate_debate_counter + 1) % 100 == 0:
            print(
                "Avg evaluate debate time:",
                self.evaluate_debate_time / self.evaluate_debate_counter,
            )

        return utility

    def full_report(self, input):
        input = np.reshape(
            input, self.shape
        )  # reshapes vectors into images, if appropriate
        prediction = self.predictor({"masked_x": input})
        probabilities = prediction["probabilities"][0]
        return probabilities

    def export_as_saved_model(self, export_dir):
        self._check_estimator_available()
        self.estimator.export_savedmodel(
            export_dir_base=export_dir,
            serving_input_receiver_fn=self.make_serving_input_receiver_fn(),
        )
