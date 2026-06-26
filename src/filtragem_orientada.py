from src.correlacao_2d import correlacao_rgb

def correlacao_orientada(img, banco_gabor):
    """
    Recebe uma imagem colorida e um dicionário de filtros Gabor {orientação: filtro}, 
    e devolve um dicionário {orientação: correlação} com a correlação da imagem com cada filtro.
    """

    resultado = {}
    for angulo, filtro in banco_gabor.items():
        resultado[angulo] = correlacao_rgb(img, filtro)
    
    return resultado