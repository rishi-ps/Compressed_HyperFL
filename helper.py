import torch

def gaussian_noise(data_shape, clip, sigma, device=None):
    return torch.normal(0, sigma * clip, data_shape).to(device)

# Define the tensor shape and noise parameters
data_shape = (3, 3)  # A 3x3 tensor
clip = 1.0           # Clip factor
sigma = 0.5          # Sigma factor

# Generate the tensor with Gaussian noise
noise_tensor = gaussian_noise(data_shape, clip, sigma)
print(noise_tensor)