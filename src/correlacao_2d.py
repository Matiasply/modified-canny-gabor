import json
import matplotlib.pyplot as plt
import cv2
import numpy as np

def carregar_matriz(arquivo_path):
    """
    Lê uma matriz salva em .txt ou .json e devolve um numpy.array.

    Formatos aceitos:
    - .json: [[1, 2], [3, 4]]
    - .txt: uma linha por linha da matriz, com valores separados por espaço ou vírgula
    """

    # Verifica se é json
    if (arquivo_path.lower().endswith(".json")):
        with open(arquivo_path, "r", encoding="utf-8") as arquivo:
            dados = json.load(arquivo)
        return np.array(dados, dtype=float)

    with open(arquivo_path, "r", encoding="utf-8") as arquivo:
        linhas = [linha.strip() for linha in arquivo if linha.strip()]

    matriz = []
    for linha in linhas:
        linha = linha.replace(",", " ")
        valores = [float(valor) for valor in linha.split()]
        matriz.append(valores)

    return np.array(matriz, dtype=float)

def correlacao_2d(img, kernel):
    """
    Recebe uma imagem 2D e um kernel 2D, e devolve a correlação com extensão por zero entre eles.
    """

    if img.ndim != 2:
        raise ValueError("correlacao_2d só aceita imagem 2D")
    
    if kernel.ndim != 2:
        raise ValueError("Kernel deve ser 2D")
    
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("Kernel deve ter dimensões ímpares")

    altura, largura = img.shape
    k_altura, k_largura = kernel.shape

    pad_h = k_altura // 2
    pad_w = k_largura // 2

    # adiciona borda de zeros
    img_padded = np.pad(
        img,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode='constant'
    )

    saida = np.zeros_like(img, dtype=np.float32)

    for i in range(altura):
        for j in range(largura):

            regiao = img_padded[
                i:i+k_altura,
                j:j+k_largura
            ]

            valor = np.sum(regiao * kernel)

            saida[i, j] = valor

    return saida

def correlacao_gray(img, kernel):
    """
    Recebe uma imagem colorida e um kernel 2D, converte a imagem para escala de cinza e 
    devolve a correlação com extensão por zero entre eles.
    Usado para Canny tradicional, que trabalha com imagens em escala de cinza.
    """
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)

    correlacao = correlacao_2d(gray, kernel)

    return correlacao

def correlacao_rgb(img, kernel):
    """
    Recebe uma imagem colorida e um kernel 2D, e devolve a correlação com extensão por zero entre eles.
    A correlação é feita separadamente para cada canal (R, G, B).
    """

    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]

    B_f = correlacao_2d(B, kernel)
    G_f = correlacao_2d(G, kernel)
    R_f = correlacao_2d(R, kernel)

    saida = np.stack([B_f, G_f, R_f], axis=2) # np.stack empilha as matrizes 2D formando uma matriz 3D

    return saida
