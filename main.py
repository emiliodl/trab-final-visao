import cv2
import numpy as np
import torch

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

# Função para detectar vagas utilizando contornos
def detectar_vagas_contornos(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    vagas = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            if w > 20 and h > 20:
                vagas.append((x, y, w, h))
    
    return vagas

# Carregando a imagem
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

    # Detectar vagas usando contornos
    vagas_contornos = detectar_vagas_contornos(frame)

    # Combinar resultados
    todas_vagas = vagas_yolo + vagas_contornos

    for (x, y, w, h) in todas_vagas:
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

    # Visualizar bordas detectadas e o resultado final
    cv2.imshow('Frame', frame)
    cv2.waitKey(0)  # Aguarda uma tecla ser pressionada
    cv2.destroyAllWindows()
