import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# Função para calcular a densidade de pixels brancos em uma ROI
def calcular_densidade(roi):
    _, thresh = cv2.threshold(roi, 128, 255, cv2.THRESH_BINARY)
    return np.sum(thresh == 255) / thresh.size

# Função para detectar vagas utilizando YOLO
def detectar_vagas_yolo(image):
    # Carregar o modelo YOLO pré-treinado
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    results = model(image)
    return results.pandas().xyxy[0]

# Carregar a imagem
image_path = 'Vagas-de-Estacionamento-Entenda-o-que-Pode-e-o-Que-nao-Pode-no-Condominio.webp'
frame = cv2.imread(image_path)

# Verificar se a imagem foi carregada corretamente
if frame is None:
    print("Erro ao carregar a imagem. Verifique o caminho do arquivo.")
else:
    # Detectar vagas usando YOLO
    results_yolo = detectar_vagas_yolo(frame)
    vagas_yolo = []
    for index, row in results_yolo.iterrows():
        if row['name'] == 'car':  # Ajustar conforme necessário
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            vagas_yolo.append((x1, y1, x2 - x1, y2 - y1))

    for (x, y, w, h) in vagas_yolo:
        vaga_roi = frame[y:y+h, x:x+w]
        densidade = calcular_densidade(vaga_roi)
        
        # Condição simples para verificar se a vaga parece ocupada
        if densidade > 0.1:  # Limiar de densidade, ajuste conforme necessário
            cor = (0, 0, 255)  # Vermelho, vaga ocupada
        else:
            cor = (0, 255, 0)  # Verde, vaga livre

        # Desenhando um retângulo ao redor da vaga
        cv2.rectangle(frame, (x, y), (x+w, y+h), cor, 2)
        cv2.putText(frame, f"Dens: {densidade:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

    # Mostrar a imagem com as detecções
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title('Detecção de Vagas de Estacionamento com YOLO')
    plt.axis('off')
    plt.show()
