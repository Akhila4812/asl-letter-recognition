from PIL import Image
import numpy as np
img = np.zeros((28, 28), dtype=np.uint8)  # Black 28x28 image
Image.fromarray(img).save('test_image.png')