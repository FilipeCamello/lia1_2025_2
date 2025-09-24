import cv2
import time
import psutil
import pandas as pd
from ultralytics import YOLO

model = YOLO('model/best_br.pt')
video = cv2.VideoCapture('videos/epi-2.mp4')

if not video.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

output_video = cv2.VideoWriter(
    'video/saved_predictions.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps, (frame_width, frame_height)
)

metricas = []

while True:
    check, img = video.read()
    if not check:
        print("Não foi possível ler o frame. Finalizando...")
        break

    inicio = time.time()

    # Predição YOLO
    results = model(img, verbose=False)[0]
    nomes = results.names

    for box in results.boxes:
        # Coordenadas da bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Classe
        cls = int(box.cls.item())
        nomeClasse = nomes[cls]

        # Confiança
        conf = float(box.conf.item())
        if conf < 0.4:
            continue

        # Texto na imagem
        texto = f'{nomeClasse} - {conf:.2f}'
        cv2.putText(img, texto, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

        # Bounding box por classe
        if nomeClasse in ['pessoa', 'com_capacete', 'com_colete']:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        elif nomeClasse in ['sem_capacete', 'sem_colete']:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 3)

    # Tempo de inferência e métricas
    tempo_inferencia = time.time() - inicio
    fps_atual = 1 / tempo_inferencia if tempo_inferencia > 0 else 0
    uso_cpu = psutil.cpu_percent()
    uso_memoria = psutil.virtual_memory().percent

    metricas.append({
        "Tempo_inferencia (s)": round(tempo_inferencia, 4),
        "FPS": round(fps_atual, 2),
        "Uso_CPU (%)": uso_cpu,
        "Uso_Memória (%)": uso_memoria
    })

    # Exibe e grava frame
    cv2.imshow('IMG', img)
    output_video.write(img)

    if cv2.waitKey(int(1000/fps)) & 0xFF == 27:
        break

video.release()
output_video.release()
cv2.destroyAllWindows()

df = pd.DataFrame(metricas)
df.to_excel("video/metricas_yolo.xlsx", index=False)
print("Métricas salvas em 'video/metricas_yolo.xlsx'")