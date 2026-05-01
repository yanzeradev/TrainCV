import os
import torch
from ultralytics import YOLO

# --- CONFIGURAÇÕES DE CAMINHO ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# Aponta para o arquivo que acabamos de criar
DATA_YAML = os.path.join(BASE_DIR, "dataset_yolo_mega", "dataset.yaml") 

# --- CONFIGURAÇÕES DO MODELO ---
# yolo11s (Small) é excelente para a RTX 3060: treina rápido e absorve bem os detalhes.
MODEL_VARIANT = "yolo11s.pt"

def train_fundacao_gpu():
    # 1. VERIFICAÇÃO DE HARDWARE E FORÇANDO CUDA
    if not torch.cuda.is_available():
        print("❌ ERRO CRÍTICO: CUDA não detectado! O PyTorch não está vendo a sua RTX 3060.")
        print("O treino cairá para a CPU e levará semanas. Abortando...")
        return
    
    device = 0 # Força o uso da primeira GPU NVIDIA (Sua RTX 3060)
    print(f"🚀 Iniciando a Forja da Fundação na GPU: {torch.cuda.get_device_name(0)}")

    # 2. INICIALIZAÇÃO
    model = YOLO(MODEL_VARIANT)

    # 3. TREINAMENTO OTMIZADO PARA RTX 3060
    model.train(
        data=DATA_YAML,
        epochs=150,
        imgsz=640,
        
        # --- AJUSTES DE MEMÓRIA (VRAM) ---
        # Removi o AutoBatch (batch=-1) pois ele às vezes avalia mal a VRAM da RTX 3060 no Windows.
        # Batch 16 ou 32 é o ideal (se der erro de memória, mude para 16).
        batch=16, 
        device=device,
        workers=4, # A RTX 3060 precisa de dados rápidos. 4 ou 8 (se sua CPU aguentar).

        project="SenseVision_Fundacao",
        name="modelo_base_rtx3060",
        
        # --- HIPERPARÂMETROS DE ALTA PERFORMANCE ---
        patience=25,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        close_mosaic=10,
        
        # --- AUGMENTATION ---
        mosaic=1.0,
        mixup=0.1,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4
    )

    print(f"✅ Treinamento concluído. O modelo fundacional está em: SenseVision_Fundacao/modelo_base_rtx3060/weights/best.pt")

if __name__ == "__main__":
    train_fundacao_gpu()