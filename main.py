import matplotlib.pyplot as plt
import cv2
from src.correlacao_2d import *

def main ():
    img = cv2.imread("download.jpeg")
    kernel = carregar_matriz("filtros/sobel_x.txt")
    gx = correlacao_rgb(img, kernel)

    plt.imshow(np.clip(gx, 0, 255).astype(np.uint8))
    plt.show()

if(__name__ == "__main__"):
    main()