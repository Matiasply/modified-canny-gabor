import matplotlib.pyplot as plt
import cv2
from src.correlacao_2d import *
from src.gerador_filtros import *
from src.filtragem_orientada import *
from src.magnitude import *

def main ():
    img = cv2.imread("download.jpeg")
    kernel = carregar_matriz("filtros/sobel_x.txt")
    gx = correlacao_rgb(img, kernel)

    parametros_gabor = ler_parametros("config/gabor.json")

    filtros = gerador_parametrico(parametros_gabor)
    filtragem = filtro_orientado(img, filtros)
    magitudes = magnitude_combinada(filtragem)
    orientacao, magnitude_final = magnitude_maxima(magitudes)

if(__name__ == "__main__"):
    main()