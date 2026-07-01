import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from src.correlacao_2d import *
from src.gerador_filtros_gabor import *
from src.filtragem_orientada import *
from src.magnitude import *
from src.nms import *
from src.histerese import *
from src.sobel import *
from src.expansao_histograma import *

def carregar_bancos_gabor(pasta_parametros):
    bancos_gabor = []

    for filtro in sorted(pasta_parametros.iterdir()):
        if filtro.is_file() and filtro.suffix == '.json':
            parametros_gabor = ler_parametros(filtro)
            filtros_gabor = gerar_filtros_gabor_orientados(parametros_gabor)
            bancos_gabor.append((filtro.stem, filtros_gabor))

    return bancos_gabor


def main ():
    pasta = Path('./imagens_fonte')
    pasta_parametros = Path('./filtros_gabor')
    bancos_gabor = carregar_bancos_gabor(pasta_parametros)

    for nome_filtro, filtros_gabor in bancos_gabor:
        for i in filtros_gabor.keys():
            filtro = expandir_histograma(filtros_gabor[i])
            plt.imshow(filtro, cmap='gray')
            plt.axis('off')
            plt.savefig(f'./imagens_filtros_gabor/{nome_filtro}/{nome_filtro}_{i}.png', bbox_inches='tight', pad_inches=0)
            plt.close()

    for arquivo in pasta.iterdir():
        if arquivo.is_file():
            img = cv2.imread(str(arquivo))

            # Canny tradicional
            kernel_gaussiano = carregar_matriz("filtros/gaussiano_5x5.txt")
            imagem_suavizada_gray = correlacao_gray(img, kernel_gaussiano)
            magnitudes, orientacoes = magnitude_orientada_sobel(imagem_suavizada_gray)

            # Imagem das magnitudes
            imagem_magnitudes = expandir_histograma(magnitudes)
            plt.imshow(imagem_magnitudes, cmap='gray')
            plt.axis('off')
            plt.savefig(f'./canny_tradicional/magnitude/{arquivo.stem}_MagTradicional.png', bbox_inches='tight', pad_inches=0)
            plt.close()

            nms = non_maximum_suppression(orientacoes, magnitudes)

            # Imagem do NMS
            imagem_nms = expandir_histograma(nms)
            plt.imshow(imagem_nms, cmap='gray')
            plt.axis('off')
            plt.savefig(f'./canny_tradicional/nms/{arquivo.stem}_NMSTradicional.png', bbox_inches='tight', pad_inches=0)
            plt.close()

            max_val = nms.max()
            Thigh = max_val * 0.2
            Tlow = max_val * 0.1

            resultado = histerese(nms, Tlow=Tlow, Thigh=Thigh)
            plt.imshow(resultado, cmap='gray')
            plt.axis('off')
            plt.savefig(f'./canny_tradicional/resultado/{arquivo.stem}_ResultadoTradicional.png', bbox_inches='tight', pad_inches=0)
            plt.close()
                                      
            # Canny modificado
            for nome_filtro, filtros_gabor in bancos_gabor:
                filtragem = correlacao_orientada(img, filtros_gabor)
                magnitudes = magnitude_combinada(filtragem)

                # Imagem das magnitudes orientadas
                for i in magnitudes.keys():
                    imagem_magnitudes = expandir_histograma(magnitudes[i])
                    plt.imshow(imagem_magnitudes, cmap='gray')
                    plt.axis('off')
                    plt.savefig(f'./canny_modificado/magnitude/{nome_filtro}/{arquivo.stem}_{nome_filtro}_Mag{i}Modificado.png', bbox_inches='tight', pad_inches=0)
                    plt.close()

                orientacao_final, magnitude_final = magnitude_maxima_gabor(magnitudes)

                # Imagem das magnitudes finais
                imagem_magnitudes_final = expandir_histograma(magnitude_final)
                plt.imshow(imagem_magnitudes_final, cmap='gray')
                plt.axis('off')
                plt.savefig(f'./canny_modificado/magnitude/{nome_filtro}/{arquivo.stem}_{nome_filtro}_MagFinalModificado.png', bbox_inches='tight', pad_inches=0)
                plt.close()

                nms = non_maximum_suppression(orientacao_final, magnitude_final)

                # Imagem do NMS
                imagem_nms = expandir_histograma(nms)
                plt.imshow(imagem_nms, cmap='gray')
                plt.axis('off')
                plt.savefig(f'./canny_modificado/nms/{arquivo.stem}_{nome_filtro}_NMSModificado.png', bbox_inches='tight', pad_inches=0)
                plt.close()

                max_val = nms.max()
                Thigh = max_val * 0.2
                Tlow = max_val * 0.1

                resultado = histerese(nms, Tlow=Tlow, Thigh=Thigh)

                plt.imshow(resultado, cmap='gray')
                plt.axis('off')
                plt.savefig(f'./canny_modificado/resultado/{arquivo.stem}_{nome_filtro}_ResultadoModificado.png', bbox_inches='tight', pad_inches=0)
                plt.close()


if(__name__ == "__main__"):
    main()