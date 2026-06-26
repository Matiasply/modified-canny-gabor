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

    # marca fortes como 255 e fracos como 100
    resultado[forte] = 255
    resultado[fraco] = 100

    # 8 vizinhos
    vizinhos = [(-1,-1), (-1,0), (-1,1),
                (0,-1),         (0,1),
                (1,-1),  (1,0), (1,1)]
    
    # 2. propagação (histerese)
    # se um fraco encostar em forte, ele vira forte
    pilha_forte = list(zip(*np.where(forte)))
    while pilha_forte:
        i, j = pilha_forte.pop()

        for di, dj in vizinhos:
            ni, nj = i + di, j + dj

            if 0 <= ni < comprimento and 0 <= nj < largura:
                if fraco[ni, nj] and resultado[ni, nj] == 100:
                    resultado[ni, nj] = 255
                    pilha_forte.append((ni, nj))

    return resultado
