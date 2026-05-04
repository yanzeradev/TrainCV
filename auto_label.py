import cv2
import os
from ultralytics import YOLO

# --- CONFIGURAÇÕES ---
MODEL_PATH = r"runs\detect\SenseVision_Fundacao\modelo_base_rtx3060-6\weights\best.pt"
VIDEO_PATH = r"inputs\fausto2.mp4" 

# Criar a estrutura exigida pelo formato MOT 1.1
OUTPUT_DIR = "dataset_mot"
GT_DIR = os.path.join(OUTPUT_DIR, "gt")
os.makedirs(GT_DIR, exist_ok=True)

def gerar_arquivos_cvat():
    """Gera o arquivo de labels que o CVAT exige para o formato MOT"""
    print("\n🔧 Gerando arquivo de classes para o CVAT...")
    labels_path = os.path.join(OUTPUT_DIR, "labels.txt")
    with open(labels_path, "w") as f:
        # O CVAT lerá esta primeira linha como índice 0
        f.write("Homem\n")
    print("✅ 'labels.txt' gerado com sucesso.")

def auto_track_video():
    print(f"🧠 Carregando a Fundação SenseClean com Rastreamento...")
    model = YOLO(MODEL_PATH)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Erro ao abrir o vídeo.")
        return

    # Chama a função para criar o labels.txt
    gerar_arquivos_cvat()

    frame_id = 1 # O formato MOT começa a contar do frame 1
    
    # O arquivo MOT é um único .txt com todos os frames e IDs
    gt_file_path = os.path.join(GT_DIR, "gt.txt")
    
    print(f"🎬 Iniciando Rastreamento Contínuo (Isso pode demorar um pouco)...")
    
    with open(gt_file_path, "w") as f:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # CORREÇÃO 1: Removido o stream=True, pois estamos passando apenas 1 frame por vez!
            results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)
            
            # Se encontrou pessoas e conseguiu dar IDs a elas
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                
                for box, track_id, conf in zip(boxes_xyxy, track_ids, confs):
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    
                    # CORREÇÃO 2: A Sintaxe oficial do MOT 1.1 alterada (classe de 1 para 0)
                    # frame_id, track_id, bbox_left, bbox_top, bbox_width, bbox_height, conf, class_id, visibility, -1
                    linha_mot = f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{width:.2f},{height:.2f},{conf:.2f},0,1,-1\n"
                    f.write(linha_mot)
                    
            if frame_id % 100 == 0:
                print(f"⏳ Processado até o frame {frame_id}...")
                
            frame_id += 1

    cap.release()
    print(f"\n✅ Rastreamento concluído! Arquivo salvo em: {gt_file_path}")
    print("👉 Para o CVAT: Entre na pasta 'dataset_mot_cvat', selecione o arquivo 'labels.txt' e a pasta 'gt' JUNTOS, crie um .zip e importe!")

if __name__ == "__main__":
    auto_track_video()