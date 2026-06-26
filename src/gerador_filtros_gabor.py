import json
import numpy as np

def ler_parametros(caminho):
    """
    Lê um arquivo JSON e devolve um dicionário com os parâmetros.
    """

    with open(caminho, 'r', encoding="utf-8") as arquivo:
        parametros = json.load(arquivo)
    
    return parametros

def gerar_filtros_gabor_orientados(parametros):
    """
    Recebe um dicionário de parâmetros e devolve um dicionário de filtros Gabor {orientação: filtro}.
    """

    tamanho_mascara = parametros['tamanho_mascara']
    sigma = parametros['sigma']
    lambd = parametros['lambda']
    gama = parametros['gamma']
    psi = parametros['psi']

    resultado = {}

    for angulo in parametros['orientacoes_graus']:
        radiano = np.radians(angulo)
        resultado[angulo] = gerar_filtro_gabor(radiano, sigma, lambd, gama, psi, tamanho_mascara)
    
    return resultado


def gerar_filtro_gabor(theta, sigma, lambd, gama, psi, mascara_tam):
    """
    Gera um filtro Gabor 2D com os parâmetros fornecidos.
    """

    if mascara_tam % 2 == 0:
        raise ValueError("A máscara deve ter dimensões ímpares")

    half_size = mascara_tam // 2

    # Cria uma grade de coordenadas (x, y) para a máscara
    y, x = np.meshgrid(np.arange(-half_size, half_size + 1), np.arange(-half_size, half_size + 1))

    # Rotaciona as coordenadas
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    # Aplica a Gaussiana
    exp_factor = np.exp(-0.5 * (x_theta**2 + (gama**2) * (y_theta**2)) / (sigma**2))
    # Aplica a função cosseno
    cos_factor = np.cos(2 * np.pi * x_theta / lambd + psi)

    gabor_filter = exp_factor * cos_factor

    return gabor_filter
