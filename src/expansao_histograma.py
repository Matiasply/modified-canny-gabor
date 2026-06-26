import numpy as np

def expandir_histograma(imagem):
    """
    Expande o histograma da imagem.
    """
    # Calcula o valor mínimo e máximo da imagem
    min_val = np.min(imagem)
    max_val = np.max(imagem)

    # Aplica a fórmula de expansão do histograma
    imagem_expandida = (imagem - min_val) * (255 / (max_val - min_val))
    
    return imagem_expandida.astype(np.uint8)