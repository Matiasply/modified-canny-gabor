import numpy as np

def histerese(magnitude_nms, Tlow, Thigh):
    """
    magnitude_nms: imagem após NMS
    Tlow: limiar baixo
    Thigh: limiar alto
    """

    comprimento, largura = magnitude_nms.shape

    # 1. classifica pixels
    forte = magnitude_nms >= Thigh
    fraco = (magnitude_nms >= Tlow) & (magnitude_nms < Thigh)

    # saída inicial
    resultado = np.zeros((comprimento, largura), dtype=np.uint8)

    # marca fortes como 255
    resultado[forte] = 255

    # 8 vizinhos
    vizinhos = [(-1,-1), (-1,0), (-1,1),
                (0,-1),         (0,1),
                (1,-1),  (1,0), (1,1)]

    # 2. propagação (histerese)
    # se um fraco encostar em forte, ele vira forte
    mudou = True

    while mudou:
        mudou = False

        for i in range(1, comprimento - 1):
            for j in range(1, largura - 1):

                if fraco[i, j] and resultado[i, j] == 0:

                    # verifica conexão com pixel forte
                    for di, dj in vizinhos:
                        if resultado[i + di, j + dj] == 255:
                            resultado[i, j] = 255
                            mudou = True
                            break

    return resultado