from src.correlacao_2d import correlacao_rgb

def filtro_orientado(img, banco_gabor):

    resultado = {}
    for angulo, filtro in banco_gabor.items():
        resultado[angulo] = correlacao_rgb(img, filtro)
    
    return resultado