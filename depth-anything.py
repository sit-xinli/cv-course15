from transformers import pipeline
from PIL import Image
import requests
import numpy as np
import matplotlib.pyplot as plt

# load pipe
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

# load image
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# inference
depth = pipe(image)["depth"]

# Convert to numpy array
depth_array = np.array(depth)


# Normalize for visualization
depth_normalized = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())

# Save depth image
plt.imsave("depth_image.png", depth_normalized, cmap='inferno')
