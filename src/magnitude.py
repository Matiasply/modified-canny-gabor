import numpy as np
import cv2

def magnitude_combinada(filtros_orientados):

    magnitude = {}

    for angulo, img in filtros_orientados.items():

        # separa canais
        B = img[:, :, 0].astype(np.float32)
        G = img[:, :, 1].astype(np.float32)
        R = img[:, :, 2].astype(np.float32)

        # Di Zenzo (norma L2)
        mag = np.sqrt(R**2 + G**2 + B**2)

        magnitude[angulo] = mag

    return magnitude

def magnitude_maxima(magnitudes):

    angulos = list(magnitudes.keys())
    matrizes = list(magnitudes.values())
    bloco_3d = np.array(matrizes)
    indices_maximos = np.argmax(bloco_3d, axis=0)
    angulos_array = np.array(angulos)
    orientacao_final = angulos_array[indices_maximos]
    magnitude_final = np.max(bloco_3d, axis=0)

    return orientacao_final, magnitude_final