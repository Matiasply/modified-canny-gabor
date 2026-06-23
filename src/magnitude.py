import numpy as np
import cv2

def magnitude_combinada(filtros_orientados):
    """
    Calcula a magnitude combinada a partir de um dicionário de imagens filtradas.
    
    Parâmetros:
    filtros_orientados (dict): Dicionário onde as chaves são os ângulos e os valores são as imagens filtradas.
    
    """
    # Inicializa a imagem de magnitude com zeros
    magnitude = {angulo: np.zeros_like(filtro, dtype=np.float32) for angulo, filtro in filtros_orientados.items()}
    
    # Soma os quadrados das imagens filtradas
    for angulo, filtro in filtros_orientados.items():
        magnitude[angulo] += np.square(filtro.astype(np.float32))
    
    # Calcula a raiz quadrada da soma dos quadrados
    for angulo in magnitude:
        magnitude[angulo] = np.sqrt(magnitude[angulo])
    
    # Normaliza a imagem para o intervalo [0, 255]
    for angulo in magnitude:
        magnitude[angulo] = cv2.normalize(magnitude[angulo], None, 0, 255, cv2.NORM_MINMAX)
    
    return {angulo: img.astype(np.uint8) for angulo, img in magnitude.items()}

def magnitude_maxima(magnitudes):

    angulos = list(magnitudes.keys())
    matrizes = list(magnitudes.values())
    bloco_3d = np.array(matrizes)
    indices_maximos = np.argmax(bloco_3d, axis=0)
    angulos_array = np.array(angulos)
    orientacao_final = angulos_array[indices_maximos]
    magnitude_final = np.max(bloco_3d, axis=0)

    return orientacao_final, magnitude_final.astype(np.uint8)