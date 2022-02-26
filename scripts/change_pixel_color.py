import cv2
import numpy as np
import matplotlib.pyplot as plt


blank = np.zeros((1000, 1000, 3), np.uint8)

# circle with black background
circle = cv2.circle(blank, (500,  500), 250, (0, 255, 0), -1)

# changing background to white
circle[(circle==[0, 0, 0]).all(2)] = (255, 255, 255)

plt.imshow(circle)
plt.show()
