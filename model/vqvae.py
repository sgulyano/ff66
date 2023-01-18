import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class VectorQuantizer(layers.Layer):
    """ Implement a custom layer for the vector quantizer """
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        # Reshape the quantized values back to the original input shape
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices


class VQVAETrainer(keras.models.Model):
    """ Wrapping up the training loop inside VQVAETrainer """
    def __init__(self, get_vqvae, train_variance, encoder_input_shape, latent_dim=32, num_embeddings=128, **kwargs):
        super(VQVAETrainer, self).__init__(**kwargs)
        self.encoder_input_shape = encoder_input_shape
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        self.vqvae = get_vqvae(self.encoder_input_shape, self.latent_dim, self.num_embeddings)
        self.vqvae.summary()

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)

            # Calculate the losses.
            reconstruction_loss = (
                tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
            )
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return {
            "e_loss": self.total_loss_tracker.result(),
            "e_reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "e_vqvae_loss": self.vq_loss_tracker.result(),
        }


class VQSVAETrainer(keras.models.Model):
    def __init__(self, get_vqsvae, vqvae, train_variance, encoder_input_shape, num_class, latent_dim=32, num_embeddings=128, alpha=10, **kwargs):
        super(VQSVAETrainer, self).__init__(**kwargs)
        self.encoder_input_shape = encoder_input_shape
        self.train_variance = train_variance
        self.num_class = num_class
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.alpha = alpha

        self.vqsvae = get_vqsvae(vqvae, self.encoder_input_shape, self.num_class, self.latent_dim, self.num_embeddings)
        self.vqsvae.summary()

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")
        self.clf_loss_tracker = keras.metrics.Mean(name="clf_loss")
        self.clf_bac_tracker = keras.metrics.Mean(name="clf_bac")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
            self.clf_loss_tracker,
            self.clf_bac_tracker
        ]

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions, prediction = self.vqsvae(x)

            # Calculate the losses.
            reconstruction_loss = (
                tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
            )
            prediction_loss = tf.reduce_mean(
                tf.keras.metrics.binary_crossentropy(y, prediction)
            )
            
            prediction_bac = tf.reduce_mean(
                tf.keras.metrics.binary_accuracy(y, prediction)
            )
            
            total_loss = reconstruction_loss + sum(self.vqsvae.losses) + (self.alpha * prediction_loss)

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqsvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqsvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqsvae.losses))
        self.clf_loss_tracker.update_state(prediction_loss)
        self.clf_bac_tracker.update_state(prediction_bac)

        # Log results.
        return {
            "s_loss": self.total_loss_tracker.result(),
            "s_reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "s_vqsvae_loss": self.vq_loss_tracker.result(),
            "s_clf_loss": self.clf_loss_tracker.result(),
            "s_clf_bac": self.clf_bac_tracker.result(),
        }


def get_encoder_4conv3(input_shape, latent_dim=16):
    encoder_inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(16, 3, activation="relu", strides=2, padding="same")(
        encoder_inputs
    )
    x = layers.Conv1D(32, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv1D(48, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv1D(64, 3, activation="relu", strides=2, padding="same")(x)
    encoder_outputs = layers.Conv1D(latent_dim, 1, padding="same")(x)
    return keras.Model(encoder_inputs, encoder_outputs, name="encoder")

encoder_4conv3_str = """#### Encoder 4conv3
4xConv1; kernel=3, filter=[16, 32, 48, 64]"""


def get_decoder_4conv3(input_shape, latent_dim=16):
    latent_inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(64, 1, activation="relu", strides=1, padding="same")(
        latent_inputs
    )
    x = layers.Conv1DTranspose(48, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv1DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv1DTranspose(16, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv1DTranspose(1, 3, activation="sigmoid", strides=2, padding="same")(x)
    # decoder_outputs = layers.Conv1D(1, 1, padding="valid", activation="sigmoid")(x)
    return keras.Model(latent_inputs, decoder_outputs, name="decoder")

decoder_4conv3_str = """#### Decoder 4conv3
Conv1; kernel=1, filter=64
3xConv1T; kernel=3, filter=[48, 32, 16,]
Output=Conv1T; kernel=3"""

def get_classifier_tanh(input_shape, num_class, latent_dim=16):
    latent_inputs = keras.Input(shape=input_shape)
    x = layers.Dense(1, activation="relu")(latent_inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(100, activation='tanh')(x)
    x = layers.Dense(100, activation='tanh')(x) 
    clf_output = layers.Dense(num_class, activation='sigmoid')(x)
    return keras.Model(latent_inputs, clf_output, name="classifier")

classifier_tanh_str = """#### Classifier tanh
Dense; unit=1, activation=relu
Flatten
2*Dense; unit=[100, 100], activation=tanh"""