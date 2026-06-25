import numpy as np

def non_maximum_suppression(orientacao_final, magnitude_final):
    '''
    orientacao_final: Matriz 2D com os ângulos de orientação dos pixels (em graus)
    magnitude_final: Matriz 2D com as magnitudes dos pixels
    '''
    
    # 1. Inicializar a matriz de saída com zeros (mesmo tamanho da Magnitude_Final)
    comprimento, largura = magnitude_final.shape
    imagem_nms = np.zeros((comprimento, largura))

    # 2. Iterar pelos pixels internos
    for i in range(1, comprimento - 1):
        for j in range(1, largura - 1):
            
            angulo = orientacao_final[i, j]
            mag_atual = magnitude_final[i, j]
            
            # Garantir que o ângulo esteja entre 0 e 180
            if (angulo < 0):
                angulo += 180
                
            # Definir as magnitudes dos vizinhos padrão como 0
            v1, v2 = 0, 0
            
            # Classe 0° -> Vizinhos de Cima e Baixo
            if (0 <= angulo < 22.5) or (157.5 <= angulo <= 180):
                v1 = magnitude_final[i - 1, j]
                v2 = magnitude_final[i + 1, j]
                
            # Classe 45° -> Vizinhos da Diagonal Secundária (Superior Esq / Inferior Dir)
            elif (22.5 <= angulo < 67.5):
                v1 = magnitude_final[i - 1, j - 1]
                v2 = magnitude_final[i + 1, j + 1]
                
            # Classe 90° -> Vizinhos da Esquerda e Direita
            elif (67.5 <= angulo < 112.5):
                v1 = magnitude_final[i, j - 1]
                v2 = magnitude_final[i, j + 1]
                
            # Classe 135° -> Vizinhos da Diagonal Principal (Superior Dir / Inferior Esq)
            elif (112.5 <= angulo < 157.5):
                v1 = magnitude_final[i - 1, j + 1]
                v2 = magnitude_final[i + 1, j - 1]

            # 3. Teste do Máximo Local
            # Se o pixel atual for maior ou igual a ambos os vizinhos, ele é o pico!
            if (mag_atual >= v1) and (mag_atual >= v2):
                imagem_nms[i, j] = mag_atual
            else:
                imagem_nms[i, j] = 0  # Suprime o pixel (zera)