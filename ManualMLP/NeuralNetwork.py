import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import os
import random

# EXERCISE 1:

# Function for plotting data
def plot_data(vr_petal_length, vr_petal_width, va_petal_length, va_petal_width):

    plt.figure(figsize=(6, 4))

    plt.scatter(vr_petal_length, vr_petal_width, color='blue', marker='o', label='Versicolor')
    plt.scatter(va_petal_length, va_petal_width, color='green', marker='x', label='Virginica')


    plt.title('Iris Petal Length vs Width')
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')
    plt.grid(True)
    plt.legend()
    plt.show()

# Neural network functions

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Linear combination and activation
def neural_network(x, W):
    z = np.dot(x, W)
    return sigmoid(z)

# Calculate decision boundary points
def calculate_decision_boundary(vr_petal_length, va_petal_length, W):
    x_pl = np.linspace(0, max(vr_petal_length.max(), va_petal_length.max()), 100)


    x_pw = -(W[0] + W[1] * x_pl) / W[2]
    return x_pl, x_pw

# Plot decision boundary
def plot_decision_boundary(vr_petal_length, vr_petal_width, va_petal_length, va_petal_width, W, filepath = None):
    
    # Calculate decision boundary points
    x_pl, x_pw = calculate_decision_boundary(vr_petal_length, va_petal_length, W)

    plt.figure(figsize=(6, 4))
    
    plt.plot(x_pl, x_pw, color='red', label='Decision Boundary')
  

    plt.scatter(vr_petal_length, vr_petal_width, color='blue', marker='o', s=20, label='Versicolor')
    plt.scatter(va_petal_length, va_petal_width, color='green', marker='x', s=20, label='Virginica')

    plt.xlim(left=0)
    plt.ylim(bottom=0)

    plt.title('Iris Petal Length vs Width with Decision Boundary')
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')
    plt.grid(True)
    plt.legend()
    
    if filepath:
        plt.savefig(filepath)
        plt.close()
    else:
        plt.show()
        plt.close()

    return

# 3D plot of decision boundary surface
def plot_3d(x_pl, x_pw, W):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X1, X2 = np.meshgrid(x_pl, x_pw)
    Z = neural_network(np.c_[np.ones(X1.ravel().shape), X1.ravel(), X2.ravel()], W)
    Z = Z.reshape(X1.shape)

    ax.plot_surface(X1, X2, Z, alpha=0.5, rstride=100, cstride=100)

    ax.set_xlabel('Petal Length (cm)')
    ax.set_ylabel('Petal Width (cm)')
    ax.set_zlabel('Output Probability')
    ax.set_title('3D Decision Boundary Surface')
    plt.show()
    return

# Classify iris based on petal length and width
def classify_iris(petal_length, petal_width, W, threshold=0.5):
    print(f"Classifying iris with petal length: {petal_length}, petal width: {petal_width}")
    x = np.array([1, petal_length, petal_width])  # Add bias term
    prob = neural_network(x, W)
    print(f"Computed probability: {prob}")
    print("Classified as:", end=" ")

    # Classify based on threshold with 0 being versicolor and 1 being virginica
    print('versicolor' if prob <= threshold else 'virginica')
    return 'versicolor' if prob <= threshold else 'virginica'

# Load data
data = pd.read_csv(os.path.join(os.getcwd(), 'irisdata.csv'))
data = data[data["species"].isin(["versicolor", "virginica"])]

# Separate data by species
vr_petal_length = data[data["species"] == "versicolor"]['petal_length'].to_numpy()
vr_petal_width = data[data["species"] == "versicolor"]['petal_width'].to_numpy()

va_petal_length = data[data["species"] == "virginica"]['petal_length'].to_numpy()
va_petal_width = data[data["species"] == "virginica"]['petal_width'].to_numpy()


# Plot initial data
plot_data(vr_petal_length, vr_petal_width, va_petal_length, va_petal_width)

# Initial weights and plot decision boundary
print("Initial weights W = [-10, 1.5, 1.5]")
W = np.array([-10, 1.5, 1.5])
plot_decision_boundary(vr_petal_length, vr_petal_width, va_petal_length, va_petal_width, W)

# Create axes for 3D plot
pl_axis = np.linspace(0, data['petal_length'].max(), 100)
pw_axis = np.linspace(0, data['petal_width'].max(), 100)

# plot 3D decision boundary
plot_3d(pl_axis, pw_axis, W)

# Classify some sample irises
print("1e Simple Classifier:")
print("----------------------------------")
# Versicolor sample
classify_iris(data[data["species"] == "versicolor"]['petal_length'].iloc[0], data[data["species"] == "versicolor"]['petal_width'].iloc[0], W)

# Virginica sample
classify_iris(data[data["species"] == "virginica"]['petal_length'].iloc[0], data[data["species"] == "virginica"]['petal_width'].iloc[0], W)

# Versicolor sample that is close to boundary
classify_iris(data['petal_length'].iloc[20], data['petal_width'].iloc[20], W)

# Virginica sample that is close to boundary
classify_iris(data['petal_length'].iloc[73], data['petal_width'].iloc[73], W)



# EXERCISE 2:
# ----------------------------------------------------------

# Mean Squared Error calculation
def mean_squared_error(x, W, y_true):
    mse = np.mean((y_true - neural_network(x, W)) ** 2)
    return mse

print()
print()
print("2b MSE Calculations:")
print("----------------------------------")

# Prepare input data with bias term
x = np.c_[np.ones(data.shape[0]), data['petal_length'].to_numpy().ravel(), data['petal_width'].to_numpy().ravel()]

# True labels: 0 for versicolor, 1 for virginica
y_true = np.array([1 if species == 'virginica' else 0 for species in data['species']])

# Calculate and print MSE with initial weights
mse = mean_squared_error(x, W, y_true)
print("Current weights:")
print(f"basis = {W[0]}, w1 = {W[1]}, w2 = {W[2]}")
print(f"Initial MSE: {mse}")
plot_decision_boundary(vr_petal_length, vr_petal_width, va_petal_length, va_petal_width, W)

# Update weights to new values and calculate MSE
W = np.array([-11, 1, 2])

print("Updated weights to W =", W)

mse = mean_squared_error(x, W, y_true)

print(f"Updated MSE: {mse}")
plot_decision_boundary(vr_petal_length, vr_petal_width, va_petal_length, va_petal_width, W)



# Exercise 2c:
# Compute gradient of MSE with respect to weights
def compute_gradient(x, W, y_true):

    gradient_w1 = 2 * np.mean((neural_network(x, W) - y_true) * neural_network(x, W) * (1 - neural_network(x, W)) * x[:, 1])
    gradient_w2 = 2 * np.mean((neural_network(x, W) - y_true) * neural_network(x, W) * (1 - neural_network(x, W)) * x[:, 2])
    gradient_w0 = 2 * np.mean((neural_network(x, W) - y_true) * neural_network(x, W) * (1 - neural_network(x, W)) * x[:, 0])

    gradient = np.array([gradient_w0, gradient_w1, gradient_w2])
    
    return gradient


# Test gradient computation
gradient = compute_gradient(x, W, y_true)
print(f"Computed gradient: {gradient}")


x_pl, x_pw = calculate_decision_boundary(vr_petal_length, va_petal_length, W)

# Update weights using gradient
W = W - 0.1* gradient
print(f"Updated weights to W = {W}")


x_pl2, x_pw2 = calculate_decision_boundary(vr_petal_length, va_petal_length, W)

x_pl = np.c_[x_pl, x_pl2]
x_pw = np.c_[x_pw, x_pw2]

# Plot changes in decision boundary
def plot_changes_in_decision_boundary(vr_petal_length, vr_petal_width, va_petal_length, va_petal_width, x_pl, x_pw):
    
    plt.figure(figsize=(6, 4))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i in range(len(x_pl.T)):

        plt.plot(x_pl[:, i], x_pw[:, i], color=colors[i], label='Decision Boundary' if i == 0 else 'Updated Decision Boundary')


    plt.scatter(vr_petal_length, vr_petal_width, color='blue', marker='o', s=20, label='Versicolor')
    plt.scatter(va_petal_length, va_petal_width, color='green', marker='x', s=20, label='Virginica')

    plt.title('Iris Petal Length vs Width with Decision Boundary Changes')
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')
    plt.grid(True)
    plt.legend()
    plt.show()

    return

# Plot the changes in decision boundary after weight update
plot_changes_in_decision_boundary(vr_petal_length, vr_petal_width, va_petal_length, va_petal_width, x_pl, x_pw)



# EXERCISE 3:
# ----------------------------------------------------------

# Gradient Descent Implementation
def gradient_descent(x, W, y_true, learning_rate=0.01, iterations=100, convergence=False):
    
    # Store history for plotting
    W_history = [W.copy()]

    mse_history = [mean_squared_error(x, W, y_true)]

    # Gradient descent loop
    for i in range(iterations):
        # Compute gradient
        grad = compute_gradient(x, W, y_true)

        # Update weights
        W -= learning_rate * grad
        W_history.append(W.copy())
        mse_history.append(mean_squared_error(x, W, y_true))

        if convergence and i > 0 and abs(mse_history[-2] - mse_history[-1]) < 1e-6:
            break
    
    # Return final weights, history of weights, and MSE history
    return W, W_history, mse_history

# Single run of gradient descent
print()
print()
print("3b Single Run:")
print("----------------------------------")

print("Weights initialized to:")
print(f"basis: {W[0]}, w1: {W[1]}, w2: {W[2]}")

# Run gradient descent with specified learning rate and iterations
print("Learning Rate: 0.1, Iterations: 10")
print("initial MSE:", mean_squared_error(x, W, y_true))
iterations = 10
W, W_history, mse_history = gradient_descent(x, W, y_true, learning_rate=0.1, iterations=iterations)

# Print final weights and MSE
print("final weights after gradient descent:")
print(f"basis: {W[0]}, w1: {W[1]}, w2: {W[2]}")
print("final MSE:", mean_squared_error(x, W, y_true))

# Create directory for saving plots
threeb = os.path.join(os.getcwd(), '3b')
os.makedirs(threeb, exist_ok=True)

# Plot decision boundary at each iteration
for i in range(iterations + 1):
    filepath = os.path.join(threeb, f"decision_boundary_{i}.png")
    plot_decision_boundary(vr_petal_length, vr_petal_width, va_petal_length, va_petal_width, W_history[i], filepath=filepath)

# Plot MSE over iterations
def plot_mse_over_iterations(mse_history, filepath=None):
    plt.figure(figsize=(6, 4))
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(mse_history)), mse_history)
    plt.title('Mean Squared Error over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)

    if filepath:
        plt.savefig(filepath)
        plt.close()

    else:
        plt.show()
        plt.close()
    
    return

# Save MSE plot
mse_filepath = os.path.join(threeb, "mse_over_iterations.png")
plot_mse_over_iterations(mse_history, filepath=mse_filepath)

print("Plots for 3b saved in directory: ", threeb)

# Multiple runs of gradient descent with random initialization
print()
print()
print("3c Multiple Runs:")
print("----------------------------------")

# Set random seed for reproducibility
np.random.seed(42)

# Perform 5 runs with different random initializations
for i in range(5):
    print()
    print()

    # Randomly initialize weights
    W = np.array([
        np.random.uniform(-12, -8),   # bias
        np.random.uniform(0.1, 2),    # w1
        np.random.uniform(0.1, 2)     # w2
    ])

    # Print run information
    print(f"--- Run {i+1} ---")
    print("weights initialized to:")
    print(f"basis: {W[0]}, w1: {W[1]}, w2: {W[2]}")
    iterations = np.random.randint(100, 500)
    learningrate = np.random.uniform(0.01, 0.5)
    print(f"Learning Rate: {learningrate}, Iterations: {iterations}")
    print("initial MSE:", mean_squared_error(x, W, y_true))

    # Run gradient descent
    W, W_history, mse_history = gradient_descent(x, W, y_true, learning_rate=learningrate, iterations=iterations)

    print()
    print("final weights after gradient descent:")
    print(f"basis: {W[0]}, w1: {W[1]}, w2: {W[2]}")
    print("final MSE:", mean_squared_error(x, W, y_true))

    # Create directory for this run's plots
    threec = os.path.join(os.getcwd(), '3c', f'run_{i+1}')
    os.makedirs(threec, exist_ok=True)

    # Plot decision boundary at each iteration
    for j in range(iterations + 1):
        plot_decision_boundary(vr_petal_length, vr_petal_width, va_petal_length, va_petal_width, W_history[j], filepath=os.path.join(threec, f"decision_boundary_{j}.png"))

    # Plot MSE over iterations
    plot_mse_over_iterations(mse_history, filepath=os.path.join(threec, "mse_over_iterations.png"))

    print(f"Plots for Run {i+1} saved in directory: {threec}")

# Utilizing a Convergece Criterion
print()
print()
print("3d Convergence:")
print("----------------------------------")

# Randomly initialize weights
W = np.array([
        np.random.uniform(-12, -8),   # bias
        np.random.uniform(0.1, 2),    # w1
        np.random.uniform(0.1, 2)     # w2
    ])

# Print initial weights and settings
print("weights initialized to:")
print(f"basis: {W[0]}, w1: {W[1]}, w2: {W[2]}")
# Set learning rate and maximum iterations
iterations = 1000
learningrate = 0.01

# Print learning rate and iterations
print(f"Learning Rate: {learningrate}, Iterations: {iterations}")
print("initial MSE:", mean_squared_error(x, W, y_true))
plot_decision_boundary(vr_petal_length, vr_petal_width, va_petal_length, va_petal_width, W, filepath=os.path.join(os.getcwd(), '3d', f"decision_boundary_initial.png"))

# Run gradient descent with convergence criterion
W, W_history, mse_history = gradient_descent(x, W, y_true, learning_rate=learningrate, iterations=iterations, convergence = True)

# Print final results
print()
print(f"Converged in {len(mse_history)-1} iterations.")
print("final weights after gradient descent:")
print(f"basis: {W[0]}, w1: {W[1]}, w2: {W[2]}")
print("final MSE:", mean_squared_error(x, W, y_true))

# Check for convergence to less than 50% of initial MSE
for i in range(len(mse_history)):
    if mse_history[i]/mse_history[0] < 0.5:
        print()
        print(f"Converged to less than 50% of initial MSE at iteration {i}")
        print("Weights at this iteration:")
        print(f"basis: {W_history[i][0]}, w1: {W_history[i][1]}, w2: {W_history[i][2]}")
        print("MSE at this iteration:", mse_history[i])
        print()
        plot_decision_boundary(vr_petal_length, vr_petal_width, va_petal_length, va_petal_width, W_history[i], filepath=os.path.join(os.getcwd(), '3d', f"decision_boundary_convergence_half.png"))
        break 

# Create directory for saving plots
threed = os.path.join(os.getcwd(), '3d')

os.makedirs(threed, exist_ok=True)

# Plot decision boundary at final iteration
plot_decision_boundary(vr_petal_length, vr_petal_width, va_petal_length, va_petal_width, W, filepath=os.path.join(threed, f"decision_boundary_final.png"))

print("Plots for 3d saved in directory: ", threed)