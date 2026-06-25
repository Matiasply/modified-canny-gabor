import matplotlib.pyplot as plt
import cv2
from src.correlacao_2d import *
from src.gerador_filtros import *
from src.filtragem_orientada import *
from src.magnitude import *
from src.nms import *
from src.histerese import *

def main ():
    img = cv2.imread("Imagens/download.jpeg")

    parametros_gabor = ler_parametros("config/gabor.json")

    filtros = gerador_parametrico(parametros_gabor)
    filtragem = filtro_orientado(img, filtros)
    magnitudes = magnitude_combinada(filtragem)
    orientacao, magnitude_final = magnitude_maxima(magnitudes)
    nms = non_maximum_suppression(orientacao, magnitude_final)
    max_val = nms.max()

    Thigh = max_val * 0.2
    Tlow = max_val * 0.1

    histerese_resultado = histerese(nms, Tlow=Tlow, Thigh=Thigh)

    plt.imshow(histerese_resultado, cmap='gray')
    plt.title("Resultado final - Histerese (Canny)")
    plt.axis('off')
    plt.show()

    #for angulo, filtro in magnitudes.items():
        #plt.subplot(2, 4, int(angulo/22.5)+1)
        #plt.imshow(filtro, cmap='gray')
        #plt.title(f"{angulo}°")
        #plt.axis('off')

if(__name__ == "__main__"):
    main()