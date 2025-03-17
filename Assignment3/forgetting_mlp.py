import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import hashlib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

# Constants
NUM_TASKS = 10
EPOCHS_FIRST_TASK = 50
EPOCHS_PER_TASK = 20
LEARNING_RATE = 0.0005
BATCH_SIZE = 5000
IMAGE_SIZE = 28
INPUT_SIZE = IMAGE_SIZE * IMAGE_SIZE
HIDDEN_SIZE = 256
OUTPUT_SIZE = 10
random_seed = 1500
np.random.seed(random_seed)
tf.random.set_seed(random_seed)


# Load MNIST dataset
def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train.reshape(-1, INPUT_SIZE)
    x_test = x_test.reshape(-1, INPUT_SIZE)
    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_mnist()


# Generate permuted tasks
def generate_permuted_tasks(num_tasks=NUM_TASKS):
    return [np.random.permutation(INPUT_SIZE) for _ in range(num_tasks)]


task_permutations = generate_permuted_tasks(NUM_TASKS)


# Apply permutation to images
def apply_permutation(images, permutation):
    return images[:, permutation]


# Create MLP model
def create_mlp_model(depth=2, dropout_rate=0.5, optimizer_name="adam", loss_function="nll"):
    model = Sequential()
    model.add(Dense(HIDDEN_SIZE, activation="relu", input_shape=(INPUT_SIZE,)))
    for _ in range(depth - 1):
        model.add(Dense(HIDDEN_SIZE, activation="relu"))
        model.add(Dropout(dropout_rate))
    model.add(Dense(OUTPUT_SIZE, activation="softmax"))

    optimizers = {"sgd": SGD(LEARNING_RATE), "adam": Adam(LEARNING_RATE), "rmsprop": RMSprop(LEARNING_RATE)}
    optimizer = optimizers.get(optimizer_name.lower(), Adam(LEARNING_RATE))

    if loss_function == "nll":
        loss = SparseCategoricalCrossentropy()
    elif loss_function == "l1":
        loss = "mean_absolute_error"
    elif loss_function == "l2":
        loss = "mean_squared_error"
    elif loss_function == "l1+l2":
        loss = lambda y_true, y_pred: tf.reduce_mean(tf.abs(y_true - y_pred)) + tf.reduce_mean(
            tf.square(y_true - y_pred))
    else:
        raise ValueError("Invalid loss function. Choose from: 'nll', 'l1', 'l2', 'l1+l2'.")

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model


R = np.zeros((NUM_TASKS, NUM_TASKS))  # Results matrix for task accuracies
G = np.zeros((NUM_TASKS, NUM_TASKS))  # Immediate post-learning performance matrix
loss_history = []
train_loss_history = []  # Store training loss for all tasks
val_loss_history = []  # Store validation loss for all tasks
train_acc_history = []  # Store training accuracy for all tasks
val_acc_history = []  # Store validation accuracy for all tasks

# Train on Task A
task_A_perm = task_permutations[0]
x_train_A = apply_permutation(x_train, task_A_perm)
x_test_A = apply_permutation(x_test, task_A_perm)

mlp_model = create_mlp_model(depth=2, dropout_rate=0.5, optimizer_name="adam", loss_function="nll")

print("\nTraining on Task A...")
history_A = mlp_model.fit(x_train_A, y_train, epochs=EPOCHS_FIRST_TASK, batch_size=BATCH_SIZE,
                          validation_data=(x_test_A, y_test), verbose=1)
loss_history.append(history_A.history['val_loss'])
# Append Task A metrics
train_loss_history.extend(history_A.history['loss'])
val_loss_history.extend(history_A.history['val_loss'])
train_acc_history.extend(history_A.history['accuracy'])
val_acc_history.extend(history_A.history['val_accuracy'])

task_A_eval = mlp_model.evaluate(x_test_A, y_test, verbose=1)
task_A_accuracy = task_A_eval[1]
R[0, 0] = task_A_accuracy
G[0, 0] = task_A_accuracy

print(f"\nTask A Final Accuracy: {task_A_accuracy:.4f}")

print("\nCurrent Results Matrix R (After Task A Training):")
print(R)

# Train on subsequent tasks
for task_id in range(1, NUM_TASKS):
    print(f"\nTraining on Task {task_id + 1}...")

    x_train_task = apply_permutation(x_train, task_permutations[task_id])
    x_test_task = apply_permutation(x_test, task_permutations[task_id])

    history = mlp_model.fit(x_train_task, y_train, epochs=EPOCHS_PER_TASK, batch_size=BATCH_SIZE,
                            validation_data=(x_test_task, y_test), verbose=1)
    loss_history.append(history.history['val_loss'])
    # Append metrics for each task
    train_loss_history.extend(history.history['loss'])
    val_loss_history.extend(history.history['val_loss'])
    train_acc_history.extend(history.history['accuracy'])
    val_acc_history.extend(history.history['val_accuracy'])

    for test_task in range(task_id + 1):
        x_test_eval = apply_permutation(x_test, task_permutations[test_task])
        _, acc = mlp_model.evaluate(x_test_eval, y_test, verbose=0)
        R[task_id, test_task] = acc
    G[task_id][task_id] = R[task_id][task_id]  # Record immediate post-learning accuracy
    print(f"\nUpdated Results Matrix R (After Task {task_id + 1} Training):")
    print(R)

# Calculate forgetting metrics
ACC = np.mean(R[-1])
BWT = np.mean(R[-1, :-1] - np.diag(R[:-1]))

# Temporal Backward Transfer (TBWT)
TBWT = np.mean([R[-1, i] - G[i, i] for i in range(NUM_TASKS - 1)])  # Performance change over time

# Cumulative Backward Transfer (CBWT)
CBWT = np.mean([R[j, i] - R[i, i] for i in range(NUM_TASKS - 1) for j in range(i + 1, NUM_TASKS)])

print(f"\nForgetting Metrics:\n ACC: {ACC:.4f}, BWT: {BWT:.4f}, TBWT: {TBWT:.4f}, CBWT: {CBWT:.4f}")


# Plot Loss and Accuracy Curves
epochs = range(1, len(train_loss_history) + 1)

plt.figure(figsize=(14, 6))

# Plot Loss Curve
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss_history, label='Training Loss', color='blue')
plt.plot(epochs, val_loss_history, label='Validation Loss', color='orange')
plt.title("Loss Over Training")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Plot Accuracy Curve
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc_history, label='Training Accuracy', color='green')
plt.plot(epochs, val_acc_history, label='Validation Accuracy', color='red')
plt.title("Accuracy Over Training")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

def plot_model_architecture():
    # Plot model architecture
    mlp_model = create_mlp_model(depth=2, dropout_rate=0.5, optimizer_name="adam", loss_function="nll")
    plot_model(mlp_model, to_file="mlp_model.png", show_shapes=True, show_layer_names=True)


def get_permuted_image():
    image_idx = 0
    task_id = 0

    # Get the original image
    original_image = x_train[image_idx].reshape(28, 28)

    # Apply the permutation to get the permuted image
    permutation = task_permutations[task_id]
    permuted_image = x_train[image_idx, permutation].reshape(28, 28)

    # Display the original and permuted images
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(original_image, cmap='gray')
    ax[0].set_title('Original Image')

    ax[1].imshow(permuted_image, cmap='gray')
    ax[1].set_title('Permuted Image')

    plt.show()


def plot_grouped_bar_chart(categories, metric_values):
    metrics = ["ACC", "BWT", "TBWT", "CBWT"]
    bar_width = 0.2
    x = np.arange(len(categories))

    plt.figure(figsize=(10, 6))
    plt.bar(x, metric_values[:, 0], width=bar_width, label="ACC", color="#7f7f7f")
    plt.bar(x + bar_width, metric_values[:, 1], width=bar_width, label="BWT", color="#17becf")
    plt.bar(x + 2 * bar_width, metric_values[:, 2], width=bar_width, label="TBWT", color="teal")
    plt.bar(x + 3 * bar_width, metric_values[:, 3], width=bar_width, label="CBWT", color="#e377c2")

    # plt.xlabel(xlabel)
    plt.ylabel("Metric Value")
    # plt.title(title)
    plt.xticks(x + 1.5 * bar_width, categories)
    plt.axhline(0, color="black", linewidth=1)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


# Compute metrics for different settings dynamically
def compute_metrics(loss_functions=["nll"], dropout_rates=[0.5], depths=[2], optimizers=["adam"]):
    metric_values = []

    for loss_function in loss_functions:
        for dropout_rate in dropout_rates:
            for depth in depths:
                for optimizer in optimizers:
                    mlp_model = create_mlp_model(depth=depth, dropout_rate=dropout_rate, optimizer_name=optimizer,
                                                 loss_function=loss_function)

                    # Train on Task A
                    task_A_perm = task_permutations[0]
                    x_train_A = apply_permutation(x_train, task_A_perm)
                    x_test_A = apply_permutation(x_test, task_A_perm)

                    history_A = mlp_model.fit(x_train_A, y_train, epochs=EPOCHS_FIRST_TASK, batch_size=BATCH_SIZE,
                                              validation_data=(x_test_A, y_test), verbose=0)

                    # Calculate forgetting metrics for this setting
                    R = np.zeros((NUM_TASKS, NUM_TASKS))
                    for task_id in range(NUM_TASKS):
                        x_train_task = apply_permutation(x_train, task_permutations[task_id])
                        x_test_task = apply_permutation(x_test, task_permutations[task_id])
                        history = mlp_model.fit(x_train_task, y_train, epochs=EPOCHS_PER_TASK, batch_size=BATCH_SIZE,
                                                validation_data=(x_test_task, y_test), verbose=0)
                        for test_task in range(task_id + 1):
                            x_test_eval = apply_permutation(x_test, task_permutations[test_task])
                            _, acc = mlp_model.evaluate(x_test_eval, y_test, verbose=0)
                            R[task_id, test_task] = acc

                    ACC = np.mean(R[-1])
                    BWT = np.mean(R[-1, :-1] - np.diag(R[:-1]))
                    TBWT = np.mean(R[-1, :-1] - R[1:, :-1])
                    CBWT = np.mean([R[j, i] - R[i, i] for i in range(NUM_TASKS - 1) for j in range(i + 1, NUM_TASKS)])

                    metric_values.append([ACC, BWT, TBWT, CBWT])

    return np.array(metric_values)


# Compute and plot metrics for different loss functions
loss_functions = ["nll", "l1", "l2", "l1+l2"]
loss_values = compute_metrics(loss_functions=loss_functions)
plot_grouped_bar_chart(loss_functions, loss_values)


# Compute and plot metrics for different dropout rates
dropout_rates = [0.0, 0.2, 0.3, 0.4]
dropout_values = compute_metrics(dropout_rates=dropout_rates)
plot_grouped_bar_chart(dropout_rates, dropout_values)

# Compute and plot metrics for different model depths
depths = [2, 3, 4]
depth_values = compute_metrics(depths=depths)
plot_grouped_bar_chart(depths, depth_values)

# Compute and plot metrics for different optimizers
optimizers = ["adam", "sgd", "rmsprop"]
optimizer_values = compute_metrics(optimizers=optimizers)
plot_grouped_bar_chart(optimizers, optimizer_values)
