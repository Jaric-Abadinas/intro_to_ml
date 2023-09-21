import numpy as np

data = np.array([1, 2, 3, 4, 5])
weights = np.random.rand(5,1)


"""
# Slide 1: Array Shapes
print(data.shape)
print(data.dtype)

print(weights.shape)
print(weights.dtype)
"""


"""
# Slide 2: Array Indexing
print(weights.T)
print(weights[0])   # 1st index
print(weights[1])   # 2nd index
print(weights[1:3]) # 2nd and 3rd index
print (weights[-1]) # Last index

def weight_summary(weight_arry):
    for i in range(len(weight_arry)):
        print("Weight-Index-", i, ":", weight_arry[i])

weight_summary(weights)
"""


# Slide 3: Array Operations
# Calculate the standard deviation based on the data
noise_magnitude = 0.1 * np.mean(data)

# Generate Gaussian noise with the same shape as the data
noise = np.random.normal(0, noise_magnitude, data.shape)

# Add the noise to the data
noisy_data = data + noise_magnitude * noise

# Print the original and noisy data
print("Original-data:", data)
print("Noisy-data:", noisy_data)


# Slide 4: Advanced Indexing and Slicing
# Analyze negative noise
negative_noise_mask = data > noisy_data
negative_data = data[negative_noise_mask]
neg_magnitude = np.linalg.norm(negative_data, ord=2)

print("Negative-noise-condition:", negative_noise_mask)
print("Data-with-negative-noise:", negative_data)
print(f"Magnitude-of-negative-noise:-{neg_magnitude}")


# Slide 5: Array Manipulation
# Use weight to get a weighted sum of single input of data
weights_transposed = weights.T
output = np.dot(weights_transposed, data)

print("Data-shape:", data.shape)
print("Transposed-weights-shape:", weights_transposed.shape)
print("Output-shape:", output.shape)

# Expand Dimension
data = data.reshape(data.shape[0], 1)
output = np.dot(weights_transposed, data)

print("New-Data-shape:", data.shape)
print("Output-shape:", output.shape)

# Batch operation
data = np.random.rand(data.shape[0], 100)
output = np.dot(weights_transposed, data)

print("Batch-data-shape:", data.shape)
print("Batch-output-shape:", output.shape)


# Slide 6: Linear Algebra
# Calculate the covariance matrix on centered data
data_centered = data - np.mean(data, axis=0)
cov_matrix = np.cov(data_centered, rowvar=False)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sor eigenvalues and corresponding eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# Select the top k eigenvectors (e.g., k=2)
k = 2
top_k_eigenvectors = sorted_eigenvectors[:, :k]

# Transform the original data
principal_components = np.dot(data_centered, top_k_eigenvectors)

print("Principal-components-shape:", principal_components.shape)


# Slide 7: File I/O
# Reading and writing arrays to files
np.savetxt('array.txt', data)
arr_loaded = np.loadtxt('array.txt')