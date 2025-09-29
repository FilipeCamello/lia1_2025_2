import cv2
import numpy as np
from ultralytics import YOLO
import pyautogui
import tkinter as tk
import threading
import time

# Carregar o modelo YOLO
model = YOLO('model/minecraft_best.pt')
class_names = model.names

# Variáveis globais
running = True
detections = []

def capture_screen():
    """Captura a tela inteira do computador"""
    try:
        screenshot = pyautogui.screenshot()
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame
    except Exception as e:
        print(f"Erro ao capturar tela: {e}")
        return None

def detection_worker():
    """Thread para processamento das detecções"""
    global detections, running
    
    print("🔍 Iniciando thread de detecção...")
    
    while running:
        try:
            # Capturar tela
            frame = capture_screen()
            if frame is None:
                time.sleep(0.1)
                continue
            
            original_height, original_width = frame.shape[:2]
            
            # Fazer detecção com tamanho otimizado
            resized_width, resized_height = 640, 640
            small_frame = cv2.resize(frame, (resized_width, resized_height))
            
            # Configurações para melhor detecção
            results = model.predict(
                small_frame, 
                verbose=False, 
                save=False,
                imgsz=640,
                conf=0.3,
                iou=0.5
            )
            
            # Processar resultados e escalar coordenadas
            current_detections = []
            scale_x = original_width / resized_width
            scale_y = original_height / resized_height
            
            detection_count = 0
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    cls = int(box.cls[0])
                    class_name = class_names[cls] if cls in class_names else 'Desconhecido'
                    
                    # Escalar coordenadas
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    
                    # Garantir que as coordenadas estão dentro da tela
                    x1 = max(0, min(x1, original_width))
                    x2 = max(0, min(x2, original_width))
                    y1 = max(0, min(y1, original_height))
                    y2 = max(0, min(y2, original_height))
                    
                    current_detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(confidence),
                        'class_name': class_name
                    })
                    detection_count += 1
            
            detections = current_detections
            if detection_count > 0:
                print(f"🎯 {detection_count} detecção(ões) encontrada(s)")
            
            time.sleep(0.2)
            
        except Exception as e:
            print(f"❌ Erro na thread de detecção: {e}")
            time.sleep(0.5)

def create_transparent_overlay():
    """Cria overlay verdadeiramente transparente"""
    global running, detections
    
    # Criar janela principal
    root = tk.Tk()
    root.title("Detector em Tempo Real - LIA 2025")
    
    # Obter resolução da tela
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    print(f"🖥️ Configurando overlay transparente para {screen_width}x{screen_height}")
    
    # CONFIGURAÇÃO PARA JANELA TRANSPARENTE NO WINDOWS
    root.attributes('-fullscreen', True)
    root.attributes('-topmost', True)  # Sempre na frente
    root.attributes('-transparentcolor', 'white')  # Tornar branco transparente
    root.configure(bg='white')  # Fundo branco que será transparente
    root.overrideredirect(True)  # Remover bordas da janela
    
    # Criar canvas com fundo transparente
    canvas = tk.Canvas(
        root,
        bg='white',  # Mesma cor que será transparente
        highlightthickness=0,
        borderwidth=0
    )
    canvas.pack(fill=tk.BOTH, expand=True)
    
    # Variáveis de controle
    paused = False
    show_info = True
    
    def draw_detection(canvas, detection):
        """Desenha uma detecção no canvas"""
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        class_name = detection['class_name']
        
        # Desenhar retângulo (usar cores que não sejam brancas)
        canvas.create_rectangle(
            x1, y1, x2, y2,
            outline='#00FF00',  # Verde
            width=3,
            fill='',  # Transparente
            tags="detection"
        )
        
        # Desenhar texto
        text = f'{class_name} {confidence:.2f}'
        canvas.create_text(
            x1, y1 - 25,
            text=text,
            fill='#00FF00',  # Verde
            font=('Arial', 10, 'bold'),
            anchor='nw',
            tags="detection_text"
        )
    
    def update_overlay():
        """Atualiza o overlay com as detecções atuais"""
        if not running:
            root.quit()
            return
        
        # Limpar canvas (importante: não limpar completamente para manter transparência)
        canvas.delete("detection")
        canvas.delete("detection_text")
        canvas.delete("info")
        canvas.delete("controls")
        canvas.delete("status_bg")
        
        if not paused and detections:
            # Desenhar detecções
            for detection in detections:
                draw_detection(canvas, detection)
        
        # Desenhar informações de status (em cores não-brancas)
        if show_info:
            # Fundo semi-transparente para informações
            canvas.create_rectangle(
                5, 5, 300, 80,
                fill='black',
                stipple='gray50',  # Padrão de pontilhado para transparência
                tags="status_bg"
            )
            
            status_color = '#FF0000' if paused else '#00FF00'
            status_text = f"Detecções: {len(detections)} | {'PAUSADO' if paused else 'ATIVO'}"
            
            canvas.create_text(
                10, 10,
                text=status_text,
                fill=status_color,
                font=('Arial', 12, 'bold'),
                anchor='nw',
                tags="info"
            )
            
            controls_text = "ESC: Sair | ESPAÇO: Pausar | I: Info"
            canvas.create_text(
                10, 35,
                text=controls_text,
                fill='#FFFFFF', 
                font=('Arial', 10),
                anchor='nw',
                tags="controls"
            )
        
        # Agendar próxima atualização
        root.after(50, update_overlay)  # 20 FPS
    
    def on_key_press(event):
        nonlocal paused, show_info
        
        if event.keysym == 'Escape':
            global running
            running = False
            print("👋 Saindo...")
            root.quit()
        elif event.keysym == 'space':
            paused = not paused
            status = "pausada" if paused else "retomada"
            print(f"⏸️ Detecção {status}")
        elif event.keysym == 'i' or event.keysym == 'I':
            show_info = not show_info
            print(f"ℹ️ Informações {'ocultas' if not show_info else 'visíveis'}")
    
    def pass_through_click(event):
        """Faz com que os cliques passem pela janela transparente"""
        # Esta função permite que cliques passem para aplicações abaixo
        root.attributes('-alpha', 0.01)  # Quase invisível momentaneamente
        root.update()
        root.after(10, lambda: root.attributes('-alpha', 1.0))  # Restaurar
    
    # Configurar bindings
    root.bind('<KeyPress>', on_key_press)
    root.bind('<Button-1>', pass_through_click)  # Clique esquerdo
    root.bind('<Button-2>', pass_through_click)  # Clique do meio  
    root.bind('<Button-3>', pass_through_click)  # Clique direito
    
    # Focar na janela para capturar teclas
    root.focus_force()
    
    print("✅ Overlay transparente ativo!")
    print("🎮 Controles:")
    print("   - ESC: Sair")
    print("   - ESPAÇO: Pausar/Continuar") 
    print("   - I: Mostrar/ocultar informações")
    print("   - Cliques passam para aplicações abaixo")
    
    # Iniciar atualização
    update_overlay()
    
    # Iniciar loop principal
    try:
        root.mainloop()
    except Exception as e:
        print(f"❌ Erro na interface: {e}")
    finally:
        running = False

# VERSÃO ALTERNATIVA SE A PRIMEIRA NÃO FUNCIONAR
def create_click_through_overlay():
    """Versão alternativa com janela clicável"""
    global running, detections
    
    root = tk.Tk()
    root.title("Detector LIA 2025")
    
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    # Configuração para janela clicável
    root.attributes('-fullscreen', True)
    root.attributes('-topmost', True)
    root.attributes('-alpha', 0.01)  # Quase invisível
    root.configure(bg='black')
    root.overrideredirect(True)
    
    # Tornar a janela clicável (pass-through)
    try:
        # No Windows, podemos usar esta abordagem
        root.wm_attributes("-transparent", "black")
    except:
        pass
    
    canvas = tk.Canvas(root, bg='black', highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)
    
    def update():
        if not running:
            root.quit()
            return
        
        canvas.delete("all")
        
        # Apenas desenhar as detecções (o resto é transparente)
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            name = detection['class_name']
            
            # Desenhar apenas as caixas de detecção
            canvas.create_rectangle(x1, y1, x2, y2, outline='green', width=3)
            canvas.create_text(x1, y1-20, text=f"{name} {conf:.2f}", 
                            fill='green', font=('Arial', 10, 'bold'))
        
        root.after(50, update)
    
    def on_key(event):
        if event.keysym == 'Escape':
            global running
            running = False
            root.quit()
    
    root.bind('<Key>', on_key)
    root.focus_force()
    
    print("✅ Overlay clicável ativo!")
    update()
    root.mainloop()

def main():
    """Função principal"""
    global running
    
    print("🚀 Iniciando Detector de Tela Transparente")
    
    # Iniciar thread de detecção
    detection_thread = threading.Thread(target=detection_worker, daemon=True)
    detection_thread.start()
    
    print("⏳ Aguardando inicialização...")
    time.sleep(2)
    
    # Tentar primeiro o método transparente
    try:
        create_transparent_overlay()
    except Exception as e:
        print(f"❌ Método transparente falhou: {e}")
        print("🔄 Tentando método clicável...")
        try:
            create_click_through_overlay()
        except Exception as e2:
            print(f"❌ Todos os métodos falharam: {e2}")
    finally:
        running = False
        print("👋 Programa finalizado")

if __name__ == "__main__":
    main()