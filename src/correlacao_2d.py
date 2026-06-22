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


def correlacao_gray(img_path, kernel_path):
    """
    img_path: Caminho da imagem
    kernel_path: Caminho do arquivo .json ou .txt com a máscara
    """
    img = cv2.imread(img_path)

    if (img is None):
        raise FileNotFoundError(f"Não foi possível abrir a imagem: {img_path}")

    kernel = carregar_matriz(kernel_path).astype(float)

    # Converte imagem para tons de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)

    altura_img, largura_img = gray.shape

    hy, hx = kernel.shape

    # Kernel deve ter dimensões ímpares para centralizar corretamente
    if (hy % 2 == 0 or hx % 2 == 0):
        raise ValueError("Kernel deve ter dimensões ímpares (ex: 3x3, 5x5)")

    pad_y = hy // 2
    pad_x = hx // 2

    # Extensão por zeros com padding adequado para o kernel
    imagem_extendida = np.pad(gray, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=0)

    # Matriz resultante iniciada com zeros
    resultado = np.zeros((altura_img, largura_img), dtype=float)

    for y in range(altura_img):
        for x in range(largura_img):
            # Janela da imagem que a máscara irá operar
            janela = imagem_extendida[y: y + hy, x: x + hx]
            resultado[y, x] = np.sum(janela * kernel)

    return resultado


def main ():
    img = cv2.imread("download.jpeg")

    gx = correlacao_gray("download.jpeg", "sobel_x.txt")

    plt.imshow(gx, cmap='gray')
    plt.show()

if(__name__ == "__main__"):
    main()