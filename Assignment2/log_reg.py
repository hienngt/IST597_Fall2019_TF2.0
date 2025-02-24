import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# Load Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize pixel values to [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Split into training and validation sets
val_split = 0.1
num_val = int(train_images.shape[0] * val_split)
val_images = train_images[:num_val]
val_labels = train_labels[:num_val]
train_images_split = train_images[num_val:]
train_labels_split = train_labels[num_val:]

img_shape = (28, 28)


# Logistic Regression Model
class LogisticRegressionModel(tf.Module):
    def __init__(self):
        super().__init__()
        self.W = tf.Variable(tf.zeros([28 * 28, 10]), name="weights")
        self.b = tf.Variable(tf.zeros([10]), name="biases")

    def __call__(self, x):
        x = tf.cast(x, tf.float32)
        x = tf.reshape(x, [-1, 28 * 28])  # Flatten the input
        logits = tf.matmul(x, self.W) + self.b
        return tf.nn.softmax(logits)


# Compute loss with optional L2 regularization
def compute_loss(model, images, labels, lambda_reg=0.0):
    predictions = model(images)
    ce_loss = tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(labels, predictions))
    l2_loss = tf.nn.l2_loss(model.W)  # L2 regularization on weights
    return ce_loss + lambda_reg * l2_loss


# Training step for one batch
def train_step(model, images, labels, optimizer, lambda_reg=0.0):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, images, labels, lambda_reg)
    grads = tape.gradient(loss, [model.W, model.b])
    optimizer.apply_gradients(zip(grads, [model.W, model.b]))
    return loss


# Training function for multiple epochs
def train_model(optimizer, lambda_reg=0.0, num_epochs=20, batch_size=128):
    # Create a new model instance
    model = LogisticRegressionModel()

    # Prepare datasets for training and validation
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images_split, train_labels_split))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    val_dataset = val_dataset.batch(batch_size)

    # History to store metrics over epochs
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }

    # Metrics for accuracy computation
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    for epoch in range(1, num_epochs + 1):
        # Reset metrics at the start of each epoch
        train_acc_metric.reset_state()
        val_acc_metric.reset_state()

        # Training loop
        epoch_losses = []
        for batch_images, batch_labels in train_dataset:
            loss = train_step(model, batch_images, batch_labels, optimizer, lambda_reg)
            epoch_losses.append(loss.numpy())
            predictions = model(batch_images)
            train_acc_metric.update_state(batch_labels, predictions)

        # Compute average training loss and accuracy over epoch
        train_loss = np.mean(epoch_losses)
        train_accuracy = train_acc_metric.result().numpy()

        # Validation loop
        val_losses = []
        for batch_images, batch_labels in val_dataset:
            loss = compute_loss(model, batch_images, batch_labels, lambda_reg)
            val_losses.append(loss.numpy())
            predictions = model(batch_images)
            val_acc_metric.update_state(batch_labels, predictions)

        val_loss = np.mean(val_losses)
        val_accuracy = val_acc_metric.result().numpy()

        # Save metrics to history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        print(f"Epoch {epoch:02d}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_accuracy:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.4f}")

    return model, history


# Plotting function to visualize metrics over epochs
def plot_metrics(history_dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    for opt_name, history in history_dict.items():
        epochs = range(1, len(history['train_accuracy']) + 1)

        ax1.plot(epochs, history['train_accuracy'], label=f'{opt_name} Train')
        ax1.plot(epochs, history['val_accuracy'], label=f'{opt_name} Val')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        ax2.plot(epochs, history['train_loss'], label=f'{opt_name} Train')
        ax2.plot(epochs, history['val_loss'], label=f'{opt_name} Val')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_images(images, y, yhat=None):
    assert len(images) == len(y) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if yhat is None:
            xlabel = "True: {0}".format(y[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(y[i], yhat[i])

        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def plot_weights(w):
    # Get the lowest and highest values for the weights.
    w_min = np.min(w)
    w_max = np.max(w)

    # Create figure with 3x4 sub-plots,
    # where the last 2 sub-plots are unused.
    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots.
        if i < 10:
            # Get the weights for the i'th digit and reshape it.
            image = w[:, i].reshape(img_shape)

            # Set the label for the sub-plot.
            ax.set_xlabel("Weights: {0}".format(i))

            # Plot the image.
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

        # Remove ticks from each sub-plot.
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


# Experiment with different optimizers
optimizers_to_try = {
    "RMSprop": tf.optimizers.RMSprop(learning_rate=0.001)
}

lambda_reg = 0.001  # Regularization parameter (L2 penalty)
num_epochs = 100  # Number of epochs to train

history_dict = {}

for opt_name, opt in optimizers_to_try.items():
    print(f"\nTraining with {opt_name} optimizer (lambda_reg={lambda_reg})")
    model, history = train_model(opt, lambda_reg=lambda_reg,
                                 num_epochs=num_epochs,
                                 batch_size=128)
    history_dict[opt_name] = history

# Plot metrics after training all optimizers
plot_metrics(history_dict)

# Get image from test set
images = test_images[0:9]

# Get the true classes for those images.
y = test_labels[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, y=y)

# Plot weights
plot_weights(model.W.numpy())


def compare_with_svm_and_forest():
    # Flatten images for Random Forest and SVM
    train_images_flat = train_images.reshape(train_images.shape[0], -1)
    test_images_flat = test_images.reshape(test_images.shape[0], -1)

    # Random Forest Classifier
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(train_images_flat, train_labels)
    rf_predictions = rf_clf.predict(test_images_flat)
    rf_accuracy = accuracy_score(test_labels, rf_predictions)

    # SVM Classifier
    svm_clf = SVC(kernel='linear', random_state=42)
    svm_clf.fit(train_images_flat, train_labels)
    svm_predictions = svm_clf.predict(test_images_flat)
    svm_accuracy = accuracy_score(test_labels, svm_predictions)

    # Train logistic regression model
    model, history = train_model(tf.optimizers.RMSprop(learning_rate=0.001), lambda_reg=0.001, num_epochs=100,
                                 batch_size=128)

    # Evaluate logistic regression model
    logistic_predictions = tf.argmax(model(test_images), axis=1).numpy()
    logistic_accuracy = accuracy_score(test_labels, logistic_predictions)

    # Print accuracies
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    print(f"SVM Accuracy: {svm_accuracy:.4f}")
    print(f"Logistic Regression Accuracy: {logistic_accuracy:.4f}")

    # Cluster weights using K-means
    weights = model.W.numpy().T
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(weights)

    # Visualize clusters using t-SNE
    tsne = TSNE(n_components=2, perplexity=2, n_iter=300, random_state=42)
    weights_2d = tsne.fit_transform(weights)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(weights_2d[:, 0], weights_2d[:, 1], c=kmeans.labels_, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('t-SNE visualization of weight clusters')
    plt.show()