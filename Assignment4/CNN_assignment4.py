import os
import random
import numpy as np
import tensorflow as tf
import matplotlib as plt

# Set random seed for reproducibility
SEED = 2701
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.astype(np.float32) / 255.0
train_images = np.expand_dims(train_images, axis=-1)  # (batch, 28, 28, 1)
test_images = test_images.astype(np.float32) / 255.0
test_images = np.expand_dims(test_images, axis=-1)
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

batch_size = 64
hidden_size = 100
learning_rate = 0.01
output_size = 10

def batch_norm_custom(x, gamma, beta, epsilon=1e-5):
    mean = tf.reduce_mean(x, axis=0)
    variance = tf.reduce_mean(tf.square(x - mean), axis=0)
    x_hat = (x - mean) / tf.sqrt(variance + epsilon)
    return gamma * x_hat + beta

def layer_norm_custom(x, gamma, beta, epsilon=1e-5):
    mean = tf.reduce_mean(x, axis=1, keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=1, keepdims=True)
    x_hat = (x - mean) / tf.sqrt(variance + epsilon)
    return gamma * x_hat + beta

def weight_norm_custom(W, g):
    norm = tf.norm(W, axis=1, keepdims=True)  # normalize across input dim
    W_normalized = W / (norm + 1e-8)
    return W_normalized * tf.reshape(g, [1, -1])

class CNN(tf.Module):
    def __init__(self, hidden_size, output_size, norm_type='none', use_custom=True):
        super().__init__()
        self.norm_type = norm_type
        self.use_custom = use_custom
        self.W1 = tf.Variable(tf.random.normal([5, 5, 1, 30], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([30]))
        self.W2 = tf.Variable(tf.random.normal([14*14*30, hidden_size], stddev=0.1))
        self.b2 = tf.Variable(tf.zeros([hidden_size]))
        self.W3 = tf.Variable(tf.random.normal([hidden_size, output_size], stddev=0.1))
        self.b3 = tf.Variable(tf.zeros([output_size]))

        self.gamma1 = tf.Variable(tf.ones([hidden_size]))
        self.beta1 = tf.Variable(tf.zeros([hidden_size]))
        self.gamma2 = tf.Variable(tf.ones([output_size]))
        self.beta2 = tf.Variable(tf.zeros([output_size]))
        self.g_W3 = tf.Variable(tf.ones([output_size]))

        if not use_custom:
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.ln_hidden = tf.keras.layers.LayerNormalization()
            self.ln_output = tf.keras.layers.LayerNormalization()

    def __call__(self, x):
        x = tf.nn.conv2d(x, self.W1, strides=1, padding='SAME') + self.b1
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='VALID')
        x = tf.reshape(x, [x.shape[0], -1])
        x = tf.matmul(x, self.W2) + self.b2

        if self.norm_type == 'batchnorm':
            x = batch_norm_custom(x, self.gamma1, self.beta1) if self.use_custom else self.bn1(x)
        elif self.norm_type == 'layernorm':
            x = layer_norm_custom(x, self.gamma1, self.beta1) if self.use_custom else self.ln_hidden(x)

        x = tf.nn.relu(x)

        if self.norm_type == 'weightnorm':
            W3 = weight_norm_custom(self.W3, self.g_W3) if self.use_custom else self.W3
        else:
            W3 = self.W3

        x = tf.matmul(x, W3) + self.b3

        if self.norm_type == 'layernorm':
            x = layer_norm_custom(x, self.gamma2, self.beta2) if self.use_custom else self.ln_output(x)

        return x

def compute_loss(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

def compute_accuracy(logits, labels):
    pred = tf.argmax(logits, axis=1)
    true = tf.argmax(labels, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(pred, true), tf.float32))

def train_model(norm_type, use_custom, epochs=20):
    model = CNN(hidden_size, output_size, norm_type, use_custom)
    optimizer = tf.keras.optimizers.SGD(learning_rate)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).batch(batch_size)
    history = {'loss': [], 'accuracy': []}

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        for step, (x_batch, y_batch) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch)
                loss = compute_loss(logits, y_batch)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            acc = compute_accuracy(logits, y_batch)
            epoch_loss += loss.numpy()
            epoch_accuracy += acc.numpy()
        history['loss'].append(epoch_loss / (step+1))
        history['accuracy'].append(epoch_accuracy / (step+1))
        print(f"{norm_type.upper()} ({'Custom' if use_custom else 'Built-in'}) - Epoch {epoch+1}, Loss: {epoch_loss / (step+1):.4f}, Accuracy: {epoch_accuracy / (step+1):.4f}")
    return history

# Run all variants
histories = {}
settings = [
    ('none', True),
    ('batchnorm', True),
    ('batchnorm', False),
    ('layernorm', True),
    ('layernorm', False),
    ('weightnorm', True),
]
for norm_type, use_custom in settings:
    key = f"{norm_type}_{'custom' if use_custom else 'builtin'}"
    histories[key] = train_model(norm_type, use_custom)

# Plot
plt.figure(figsize=(14,6))
for key in histories:
    plt.plot(histories[key]['accuracy'], label=f"{key} accuracy")
plt.title("Training Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14,6))
for key in histories:
    plt.plot(histories[key]['loss'], label=f"{key} loss")
plt.title("Training Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
