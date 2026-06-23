import json
import numpy as np

def ler_parametros(caminho):

    with open(caminho, 'r', encoding="utf-8") as arquivo:
        parametros = json.load(arquivo)
    
    return parametros

def gerador_parametrico(parametros):

    tamanho_mascara = parametros['tamanho_mascara']
    sigma = parametros['sigma']
    lambd = parametros['lambda']
    gama = parametros['gamma']
    psi = parametros['psi']

    resultado = {}

    for angulo in parametros['orientacoes_graus']:
        radiano = np.radians(angulo)
        resultado[angulo] = filtros_gabor(radiano, sigma, lambd, gama, psi, tamanho_mascara)
    
    return resultado


def filtros_gabor(theta, sigma, lambd, gama, phi, mascara_tam):

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
    cos_factor = np.cos(2 * np.pi * x_theta / lambd + phi)

    gabor_filter = exp_factor * cos_factor

    return gabor_filter
