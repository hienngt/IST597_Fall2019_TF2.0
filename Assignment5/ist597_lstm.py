# !wget http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz
# !wget http://yaroslavvb.com/upload/notMNIST/notMNIST_large.tar.gz
# !tar -xvzf notMNIST_small.tar.gz
# !tar -xvzf notMNIST_large.tar.gz


# Modified version for Lab Problem 1: GRU vs MGU Comparison
import os
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from matplotlib.image import imread
import glob

# Set seeds for reproducibility
tf.random.set_seed(2701)
np.random.seed(2701)

# Constants
num_classes = 10
batch_size = 1000
n_epochs = 50
learning_rate = 0.001
pixel_depth = 255
buffer_size = 10000

# Data Loading
BASE_PATH = './notMNIST_small/'
end = '/*.png'
data, label = [], []

for idx, fn in enumerate(['A','B','C','D','E','F','G','H','I','J']):
    for path in glob.glob(BASE_PATH + fn + end):
        try:
            image = (imread(path).astype(float)) / pixel_depth
        except:
            continue
        data.append(image)
        label.append(idx)

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3)
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).shuffle(buffer_size).batch(batch_size)

# Custom RNN Cells (GRU and MGU)
class BasicGRU(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.kernel = tf.keras.layers.Dense(2 * units, use_bias=False)
        self.recurrent_kernel = tf.keras.layers.Dense(2 * units)
        self.unique_kernel = tf.keras.layers.Dense(units, use_bias=False)
        self.unique_recurrent_kernel = tf.keras.layers.Dense(units)

    def call(self, inputs):
        h_state = tf.zeros((inputs.shape[0], self.units))
        h_list = []
        for t in range(inputs.shape[1]):
            ip = inputs[:, t, :]
            z = self.kernel(ip) + self.recurrent_kernel(h_state)
            z0, z1 = z[:, :self.units], z[:, self.units:]
            r_t = tf.sigmoid(z0)
            z_t = tf.sigmoid(z1)
            h_t = tf.tanh(self.unique_kernel(ip) + self.unique_recurrent_kernel(h_state * r_t))
            h_state = (1 - z_t) * h_state + z_t * h_t
            h_list.append(h_state)
        return tf.stack(h_list, axis=1)[:, -1, :]

class BasicMGU(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.kernel = tf.keras.layers.Dense(units, use_bias=False)
        self.recurrent_kernel = tf.keras.layers.Dense(units)
        self.unique_kernel = tf.keras.layers.Dense(units, use_bias=False)
        self.unique_recurrent_kernel = tf.keras.layers.Dense(units)

    def call(self, inputs):
        h_state = tf.zeros((inputs.shape[0], self.units))
        h_list = []
        for t in range(inputs.shape[1]):
            ip = inputs[:, t, :]
            z = self.kernel(ip) + self.recurrent_kernel(h_state)
            f_t = tf.sigmoid(z)
            h_t = tf.tanh(self.unique_kernel(ip) + self.unique_recurrent_kernel(h_state * f_t))
            h_state = (1 - f_t) * h_state + f_t * h_t
            h_list.append(h_state)
        return tf.stack(h_list, axis=1)[:, -1, :]

# Final Model Wrapper
class RNNModel(tf.keras.Model):
    def __init__(self, rnn_type, units):
        super().__init__()
        self.rnn = BasicGRU(units) if rnn_type == 'gru' else BasicMGU(units)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.out = tf.keras.layers.Dense(num_classes)

    def call(self, inputs):
        x = self.rnn(inputs)
        x = self.norm(x)
        return self.out(x)


# Training Functions
def predict(logits):
    logits = tf.cast(logits, tf.float32)
    if len(logits.shape) == 1:
        return logits  # already class indices
    return tf.argmax(tf.nn.softmax(logits), axis=-1)

from sklearn.metrics import accuracy_score

def cal_acc(predicted, actual):
    predicted = np.array(predicted)
    actual = np.array(actual)
    return accuracy_score(actual, predicted)

def cal_loss(logits, labels):
    logits = tf.cast(logits, tf.float32)
    return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

def test(mymodel):
    logitsall = []
    yall = []
    for ids, (_x, _y) in test_dataset.enumerate():
        logits = mymodel(_x, training=False)
        logits = tf.cast(logits, tf.float32)
        logitsall.extend(logits.numpy())  # store raw logits
        yall.extend(_y.numpy())           # store ground-truth

    preds = predict(np.array(logitsall))  # convert logits to predicted labels
    acc = cal_acc(preds, yall)
    loss = tf.reduce_sum(cal_loss(tf.convert_to_tensor(logitsall), tf.convert_to_tensor(yall)))
    return acc, loss


def plot_metrics(values, title):
    plt.plot(range(1, len(values)+1), values)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.savefig(f"{title.replace(' ', '_')}.pdf")
    plt.clf()

def train(rnn_type, units=128):
    model = RNNModel(rnn_type, units)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    train_accs, test_accs = [], []

    for epoch in range(n_epochs):
        acc = tf.keras.metrics.Accuracy()
        for x_batch, y_batch in train_dataset:
            with tf.GradientTape() as tape:
                logits = model(x_batch)
                loss = cal_loss(logits, y_batch)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            acc.update_state(y_batch, predict(logits))
        train_acc = acc.result().numpy()
        test_acc = test(model)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        # print(f"{rnn_type.upper()} Epoch {epoch+1}: Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")

    plot_metrics(train_accs, f"{rnn_type} train accuracy")
    plot_metrics(test_accs, f"{rnn_type} test accuracy")
    return test_accs[-1]

def multiple_trials(rnn_type, trials=3, units=128):
    results = []
    for t in range(trials):
        print(f"\nTrial {t+1}/{trials} for {rnn_type.upper()}:")
        acc = train(rnn_type, units)
        results.append(acc)
    print(f"\n{rnn_type.upper()} - Mean Accuracy: {np.mean(results):.4f}, Std: {np.std(results):.4f}")


multiple_trials('gru', trials=3, units=128)

multiple_trials('mgu')
