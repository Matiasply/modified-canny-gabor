import matplotlib.pyplot as plt
import cv2
from src.correlacao_2d import correlacao_gray 

def main ():
    img = cv2.imread("download.jpeg")

    gx = correlacao_gray("download.jpeg", "filtros/sobel_x.txt")

    plt.imshow(gx, cmap='gray')
    plt.show()

if(__name__ == "__main__"):
    main()