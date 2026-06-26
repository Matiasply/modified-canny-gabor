from src.correlacao_2d import *

def magnitude_orientada_sobel(img):
    """
    Recebe uma imagem em tons de cinza e devolve a magnitude e orientação do gradiente usando o filtro de Sobel.
    """

    # Carregar os filtros de Sobel
    sobel_x = carregar_matriz("filtros/sobel_x.txt")
    sobel_y = carregar_matriz("filtros/sobel_y.txt")

    # Aplicar a correlação com os filtros de Sobel
    gradiente_x = correlacao_2d(img, sobel_x)
    gradiente_y = correlacao_2d(img, sobel_y)

    # Calcular a magnitude do gradiente
    magnitude = abs(gradiente_x) + abs(gradiente_y)

    # Calcular a orientação do gradiente
    orientacao = np.arctan2(gradiente_y, gradiente_x) * 180 / np.pi

    return magnitude, orientacao
