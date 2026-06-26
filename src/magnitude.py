import numpy as np
import cv2

def magnitude_combinada(filtros_orientados):
    """
    Recebe um dicionário de imagens filtradas por Gabor {orientação: imagem}, 
    e devolve um dicionário {orientação: magnitude} com a magnitude de cada imagem
    """

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

def magnitude_maxima_gabor(magnitudes):
    """
    Recebe um dicionário de magnitudes {orientação: magnitude}, 
    e devolve a orientação e magnitude máxima para cada pixel.
    """

    # Cria uma lista de ângulos e uma lista de matrizes de magnitude
    angulos = list(magnitudes.keys())
    matrizes = list(magnitudes.values())

    # Empilha as matrizes de magnitude em um bloco 3D
    bloco_3d = np.array(matrizes)

    # Para cada pixel, encontra a orientação com a magnitude máxima
    indices_maximos = np.argmax(bloco_3d, axis=0)
    angulos_array = np.array(angulos)
    orientacao_final = angulos_array[indices_maximos]

    # Para cada pixel, encontra a magnitude máxima correspondente
    magnitude_final = np.max(bloco_3d, axis=0)

    return orientacao_final, magnitude_final
