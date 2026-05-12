Q4. What is the vanishing gradient problem? During backpropagation, gradients become very small, slowing learning.
Bias shifts decision boundary.
Bias helps improve classification accuracy.
Learning rate controls how much weights update during training.
BAM is a type of:

recurrent neural network
associative memory network

It stores-input-output pattern pairs
Adaptive Resonance Theory (ART) Neural Network
p8-ART is used for:unsupervised
pattern recognition
clustering
classification
without forgetting old patterns.
This is called:Stability–Plasticity Property
vigilance parameter? Threshold value that determines whether a pattern matches an existing cluster.
Bottom-Up Weights Meaning input → cluster comparison
Top-Down Weight Matri cluster → input matching
Choice Function similarity score
Match Function how closely input matches cluster
If similarity exceeds vigilance:resonance occurs
Pattern accepted by cluster.
Step 1 Initialize weights. step 2 input given patern s3-calculate similarity s4-perform vigilance test s5 if match-assign cluster,update wts else -search nother cluster
p9-Why Random Weights?To avoid:
symmetry problem
If all weights are same: all neurons learn same thing
“Forward  [pass ]propagation sends inputs through weighted connections and activation functions to generate predictions.”
p4 dta.csv
# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Load Dataset
data = pd.read_csv("data.csv")

print(data.head())

# Step 3: Separate Input and Output
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Step 4: Convert Labels into Binary
# Example:
# If label is 0 -> 0
# Else -> 1

y = np.where(y == 0, 0, 1)

# Step 5: Select only 2 features for visualization
X = X[:, :2]

# Step 6: Initialize Weights and Bias
w = np.zeros(X.shape[1])
b = 0

# Learning rate
lr = 0.1

# Number of epochs
epochs = 20

# Step 7: Prediction Function
def predict(x):
    z = np.dot(x, w) + b
    return np.where(z >= 0, 1, 0)

# Step 8: Training
for epoch in range(epochs):

    for i in range(len(X)):

        y_pred = predict(X[i])

        error = y[i] - y_pred

        # Update weights
        w = w + lr * error * X[i]

        # Update bias
        b = b + lr * error

# Step 9: Print Final Values
print("Training Completed")
print("Weights:", w)
print("Bias:", b)

# Step 10: Plot Decision Boundary

x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.01),
    np.arange(y_min, y_max, 0.01)
)

Z = predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)

plt.scatter(X[:,0], X[:,1], c=y, edgecolors='k')

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.title("Perceptron Decision Boundary")

plt.show()
