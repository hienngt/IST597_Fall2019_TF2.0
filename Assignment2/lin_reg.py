"""
author: aam35
"""
import time
import tensorflow as tf
import matplotlib.pyplot as plt

tf.random.set_seed(123)

# Create data
NUM_EXAMPLES = 500
X = tf.random.normal([NUM_EXAMPLES])  # Inputs
noise = tf.random.normal([NUM_EXAMPLES])  # Noise
y = X * 3 + 2 + noise  # True output

train_steps = 2500  # Number of training iterations
learning_rate = 0.001  # Step size

loss_type = 'Hybrid'
patience = 300  # Steps to wait before reducing LR
lr_decay_factor = 0.5  # Factor to reduce LR by (e.g., multiply by 0.5)
best_loss = float('inf')  # Initialize best loss as infinity
patience_counter = 0  # Counter to track how long loss has not improved

W = tf.Variable(3.)  # Initializing W
b = tf.Variable(2.)  # Initializing b

# Functions for adding noise
def add_gaussian_noise(data, mean=0.0, stddev=0.1):
    return data + tf.random.normal(tf.shape(data), mean=mean, stddev=stddev)

def add_noise_to_data(X, y, step, scheme="per_epoch", frequency=500, noise_level=0.1):
    if scheme == "per_epoch" and step % frequency == 0:
        X = add_gaussian_noise(X, stddev=noise_level)
        y = add_gaussian_noise(y, stddev=noise_level)
    return X, y

def add_noise_to_weights(W, b, step, scheme="per_epoch", frequency=500, noise_level=0.1):
    if scheme == "per_epoch" and step % frequency == 0:
        W.assign_add(tf.random.normal(tf.shape(W), mean=0.0, stddev=noise_level))
        b.assign_add(tf.random.normal(tf.shape(b), mean=0.0, stddev=noise_level))

def add_noise_to_learning_rate(lr_var, step, scheme="per_epoch", frequency=500, noise_level=0.1):
    if not isinstance(lr_var, tf.Variable):
        lr_var = tf.Variable(lr_var)

    if scheme == "per_epoch" and step % frequency == 0:
        noise = tf.random.uniform([], -lr_var * noise_level, lr_var * noise_level)
        lr_var.assign(lr_var + noise)

# Start timing the training process
start_time = time.time()

for i in range(train_steps):
    # Start the timer for the epoch
    epoch_start_time = time.time()

    # Add noise to data periodically
    X_noisy, y_noisy = add_noise_to_data(X, y, i, noise_level=0.1)

    with tf.GradientTape() as tape:
        # Forward pass: compute predicted y (yhat)
        yhat = X_noisy * W + b

        # Compute loss based on the selected loss function
        if loss_type == "MSE":
            loss = tf.reduce_mean(tf.square(yhat - y_noisy))
        elif loss_type == "MAE":
            loss = tf.reduce_mean(tf.abs(yhat - y_noisy))
        elif loss_type == "Hybrid":
            alpha = 0.5
            loss = tf.reduce_mean(alpha * tf.abs(yhat - y_noisy) + (1 - alpha) * tf.square(yhat - y_noisy))
        else:
            raise ValueError("Unknown loss type selected. Choose 'MSE', 'MAE', or 'Hybrid'.")

    # Compute gradients of loss with respect to W and b
    dW, db = tape.gradient(loss, [W, b])

    # Update parameters using gradient descent
    W.assign_sub(learning_rate * dW)
    b.assign_sub(learning_rate * db)

    # Add noise to weights periodically
    add_noise_to_weights(W, b, i)

    # Adjust learning rate with periodic noise
    add_noise_to_learning_rate(learning_rate, i)

    # Track loss history and implement patience-based LR decay
    current_loss = loss.numpy()

    if current_loss < best_loss:
        best_loss = current_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        learning_rate = tf.Variable(initial_value=learning_rate, trainable=True)
        learning_rate.assign(learning_rate * lr_decay_factor)
        print(f"Reducing learning rate to {learning_rate.numpy():.6f} at step {i}")
        patience_counter = 0

    # Print progress every 500 steps
    if i % 500 == 0:
        print(f"Step {i}, Loss: {current_loss:.4f}, W: {W.numpy():.4f}, b: {b.numpy():.4f}")

    # End the timer for the epoch and report the time taken
    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    if i % 500 == 0:
        print(f"Time for step {i}: {epoch_time:.2f} seconds")

# End timing the training process
end_time = time.time()
elapsed_time = end_time - start_time

# Final results
print(f"\nFinal Model: W = {W.numpy():.4f}, b = {b.numpy():.4f}, Final Loss: {loss.numpy():.4f}")
print(f"Total training time: {elapsed_time:.2f} seconds")

def check_effect_loss_function():
    def train_model(X, y, loss_type, train_steps, learning_rate, patience, lr_decay_factor):
        # Initialize parameters
        W = tf.Variable(tf.random.normal([1], dtype=tf.float32))
        b = tf.Variable(tf.zeros([1], dtype=tf.float32))

        best_loss = float('inf')
        patience_counter = 0
        loss_history = []

        for i in range(train_steps):
            with tf.GradientTape() as tape:
                # Forward pass: compute predicted y (yhat)
                yhat = X * W + b

                # Compute loss based on the selected loss function
                if loss_type == "MSE":
                    loss = tf.reduce_mean(tf.square(yhat - y))
                elif loss_type == "MAE":
                    loss = tf.reduce_mean(tf.abs(yhat - y))
                elif loss_type == "Hybrid":
                    alpha = 0.5
                    loss = tf.reduce_mean(alpha * tf.abs(yhat - y) + (1 - alpha) * tf.square(yhat - y))

            # Compute gradients and update parameters
            dW, db = tape.gradient(loss, [W, b])
            W.assign_sub(learning_rate * dW)
            b.assign_sub(learning_rate * db)

            # Learning Rate Scheduling
            current_loss = loss.numpy()
            loss_history.append(current_loss)
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                learning_rate *= lr_decay_factor
                patience_counter = 0

        return W.numpy(), b.numpy(), loss_history


    # Train models with different loss functions
    loss_types = ["MSE", "MAE", "Hybrid"]
    results = {}

    for loss_type in loss_types:
        W, b, loss_history = train_model(X, y, loss_type, train_steps, learning_rate, patience, lr_decay_factor)
        results[loss_type] = {"W": W, "b": b, "loss_history": loss_history}

    # Plot loss histories
    plt.figure(figsize=(10, 6))
    for loss_type in loss_types:
        plt.plot(results[loss_type]["loss_history"], label=loss_type)
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Loss Function Comparison")
    plt.legend()
    plt.yscale("log")
    plt.grid(True)
    plt.show()

def check_effect_w_b():
    def train_model(initial_W, initial_b, train_steps, learning_rate=0.001, best_loss=float('inf')):
        print(f"\nTraining with Initial W: {initial_W}, Initial b: {initial_b}")

        # Initialize W and b with custom values
        W = tf.Variable(initial_W)
        b = tf.Variable(initial_b)

        loss_history = []  # To store loss over time

        for i in range(train_steps):
            with tf.GradientTape() as tape:
                # Forward pass: compute predicted y (yhat)
                yhat = X * W + b

                # Compute loss based on the selected loss function
                if loss_type == "MSE":
                    loss = tf.reduce_mean(tf.square(yhat - y))
                elif loss_type == "MAE":
                    loss = tf.reduce_mean(tf.abs(yhat - y))
                elif loss_type == "Hybrid":
                    alpha = 0.5
                    loss = tf.reduce_mean(alpha * tf.abs(yhat - y) + (1 - alpha) * tf.square(yhat - y))
                else:
                    raise ValueError("Unknown loss type selected. Choose 'MSE', 'MAE', or 'Hybrid'.")

            # Compute gradients of loss with respect to W and b
            dW, db = tape.gradient(loss, [W, b])

            # Update parameters using gradient descent
            W.assign_sub(learning_rate * dW)
            b.assign_sub(learning_rate * db)

            # --- Learning Rate Scheduling ---
            current_loss = loss.numpy()
            loss_history.append(current_loss)
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                learning_rate *= lr_decay_factor
                print(f"Reducing learning rate to {learning_rate:.6f} at step {i}")
                patience_counter = 0

            # Print training progress every 1000 steps
            if i % 1000 == 0:
                print(f"Step {i}, Loss: {current_loss:.4f}, W: {W.numpy():.4f}, b: {b.numpy():.4f}")

        print(f"Final Model: W = {W.numpy():.4f}, b = {b.numpy():.4f}, Final Loss: {loss.numpy():.4f}")
        return W.numpy(), b.numpy(), loss_history

    # Experiment with different initializations and longer training duration
    initializations = [
        (0.0, 0.0),  # Original initialization (W=0, b=0)
        (1.5, -2.5),  # Farther from true values (W=1.5, b=-2.5)
        (3.0, 2.0),  # Close to true values (W=3, b=2)
    ]

    results = {}

    for init_W, init_b in initializations:
        W_final, b_final, loss_history = train_model(init_W, init_b, train_steps)
        results[(init_W, init_b)] = {"W": W_final, "b": b_final, "loss_history": loss_history}

    # Plotting the Loss Curves for Each Initialization
    plt.figure(figsize=(10, 6))
    for key in results:
        plt.plot(results[key]["loss_history"], label=f"Init W={key[0]}, Init b={key[1]}")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Loss Curves for Different Initializations")
    plt.legend()
    plt.grid(True)
    plt.show()

def check_effect_noise():
    initial_learning_rate = 0.001

    def add_gaussian_noise(data, mean=0.0, stddev=0.1):
        return data + tf.random.normal(tf.shape(data), mean=mean, stddev=stddev)

    def add_noise_to_data(X, y, step, scheme="per_epoch", frequency=500, noise_level=0.1):
        if scheme == "per_epoch" and step % frequency == 0:
            X = add_gaussian_noise(X, stddev=noise_level)
            y = add_gaussian_noise(y, stddev=noise_level)
        return X, y

    def add_noise_to_weights(W, b, step, scheme="per_epoch", frequency=500, noise_level=0.1):
        if scheme == "per_epoch" and step % frequency == 0:
            W.assign_add(tf.random.normal(tf.shape(W), mean=0.0, stddev=noise_level))
            b.assign_add(tf.random.normal(tf.shape(b), mean=0.0, stddev=noise_level))

    add_noise_to_weights(W, b)

    def add_noise_to_learning_rate(lr, step, scheme="per_epoch", frequency=500, noise_level=0.1):
        if scheme == "per_epoch" and step % frequency == 0:
            noise = tf.random.uniform([], -lr * noise_level, lr * noise_level)
            return lr + noise
        return lr

    learning_rate = tf.Variable(initial_learning_rate)
    loss_history = []

    for i in range(train_steps):
        X_noisy, y_noisy = add_noise_to_data(X, y, i, noise_level=0.1)

        with tf.GradientTape() as tape:
            yhat = X_noisy * W + b
            if loss_type == "MSE":
                loss = tf.reduce_mean(tf.square(yhat - y_noisy))
            elif loss_type == "MAE":
                loss = tf.reduce_mean(tf.abs(yhat - y_noisy))
            elif loss_type == "Hybrid":
                alpha = 0.5
                loss = tf.reduce_mean(alpha * tf.abs(yhat - y_noisy) + (1 - alpha) * tf.square(yhat - y_noisy))
            else:
                raise ValueError("Unknown loss type selected. Choose 'MSE', 'MAE', or 'Hybrid'.")

        dW, db = tape.gradient(loss, [W, b])
        W.assign_sub(learning_rate * dW)
        b.assign_sub(learning_rate * db)

        new_lr = add_noise_to_learning_rate(learning_rate, i, noise_level=0.05)
        learning_rate.assign(new_lr)

        current_loss = loss.numpy()
        loss_history.append(current_loss)
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            learning_rate.assign(learning_rate * lr_decay_factor)
            print(f"Reducing learning rate to {learning_rate.numpy():.6f} at step {i}")
            patience_counter = 0

        if i % 500 == 0:
            print(f"Step {i}, Loss: {current_loss:.4f}, W: {W.numpy():.4f}, b: {b.numpy():.4f}")

    print(f"\nFinal Model: W = {W.numpy():.4f}, b = {b.numpy():.4f}, Final Loss: {loss.numpy():.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Loss Curve with Noise Effects")
    plt.grid(True)
    plt.show()

