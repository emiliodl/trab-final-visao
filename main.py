import cv2
import numpy as np

# Função para calcular a densidade de pixels brancos em uma ROI
def calcular_densidade(roi):
    _, thresh = cv2.threshold(roi, 128, 255, cv2.THRESH_BINARY)
    return np.sum(thresh == 255) / thresh.size

# Carregando a imagem diretamente
image_path = 'Vagas-de-Estacionamento-Entenda-o-que-Pode-e-o-Que-nao-Pode-no-Condominio.webp'
frame = cv2.imread(image_path)

# Verificar se a imagem foi carregada corretamente
if frame is None:
    print("Erro ao carregar a imagem. Verifique o caminho do arquivo.")
else:
    # Pré-processamento: Transformar a imagem para cinza e aplicar filtro para reduzir ruído
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detecção de bordas
    edges = cv2.Canny(blurred, 50, 150)

    # Definindo as ROIs (você precisará definir estas coordenadas)
    vagas = [(100, 200, 50, 30), (200, 200, 50, 30)]  # Lista de tuplas (x, y, largura, altura)

    for (x, y, w, h) in vagas:
        vaga_roi = edges[y:y+h, x:x+w]
        densidade = calcular_densidade(vaga_roi)
        
        # Condição simples para verificar se a vaga parece ocupada
        if densidade > 0.1:  # Limiar de densidade, ajuste conforme necessário
            cor = (0, 0, 255)  # Vermelho, vaga ocupada
        else:
            cor = (0, 255, 0)  # Verde, vaga livre

        # Desenhando um retângulo ao redor da vaga
        cv2.rectangle(frame, (x, y), (x+w, y+h), cor, 2)
        cv2.putText(frame, f"Dens: {densidade:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

    # Visualizar bordas detectadas e o resultado final
    cv2.imshow('Edges', edges)
    cv2.imshow('Frame', frame)
    cv2.waitKey(0)  # Aguarda uma tecla ser pressionada
    cv2.destroyAllWindows()
