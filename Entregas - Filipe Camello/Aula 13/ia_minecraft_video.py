import cv2
from ultralytics import YOLO
import yt_dlp

# Carregar o modelo YOLO
model = YOLO('model/minecraft_best.pt')

#classes: cat, chicken, cow, dog, dolphin, horse, iron golem, pig, rabbit, sheep, villager

# Importando vídeo do youtube
def get_youtube_stream(url):
    """Obtém a URL do stream diretamente usando yt-dlp"""
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'quiet': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info['url']
    except Exception as e:
        print(f"Erro ao obter stream: {e}")
        return None

# Importando vídeo do youtube - use a URL completa
video_url = "https://www.youtube.com/shorts/ZgihpwdUFEI"

# Obter URL do stream
stream_url = get_youtube_stream(video_url)

print("Conectando ao stream do YouTube...")
video = cv2.VideoCapture(stream_url)


# Ler a webcam
#video = cv2.VideoCapture(0)

# Configurações para melhor performance
frame_skip = 1  # Processar 1 frame a cada X frames
frame_count = 0
resize_factor = 1  # Reduzir resolução para 50%

# Dicionário de nomes das classes
class_names = model.names

while True:
    check, img = video.read()
    
    # Realizar a predição na imagem
    results = model.predict(img, verbose=False, save=True)

    # Extrair detecções e desenhar na imagem
    for result in results:
        for box in result.boxes:
            # Obter coordenadas, confiança e classe
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convertendo coordenadas para inteiros
            confidence = box.conf[0]
            cls = int(box.cls[0])
            
            # Obter o nome da classe
            class_name = class_names[cls] if cls in class_names else 'Desconhecido'
            
            # Desenhar o retângulo na imagem
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Adicionar o nome da classe e a confiança
            cv2.putText(img, f'{class_name} ({confidence:.2f})', 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar o vídeo com as detecções
    cv2.imshow('Detectando em LIA 2025', img)

    # Pressionar 'Esc' para sair
    if cv2.waitKey(1) == 27:
        break

# Liberar a captura e destruir todas as janelas
video.release()
cv2.destroyAllWindows()