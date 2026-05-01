import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.huggingface as fouh

def criar_mega_dataset():
    print("🚀 A iniciar a Engenharia do Mega Dataset Fundacional...")

    # 1. Criar um dataset base (vazio) que vai receber tudo
    mega_dataset = fo.Dataset("meu_mega_dataset_pessoas")
    mega_dataset.persistent = True # Guarda na base de dados do FiftyOne para não perder

    # ==========================================
    # DATASET 1: COCO (Foco na Generalização)
    # ==========================================
    print("📦 A transferir COCO...")
    coco = foz.load_zoo_dataset(
        "coco-2017",
        splits=["train"],
        label_types=["detections"],
        classes=["person"],
        max_samples=5000, # Ajuste conforme o seu hardware
        dataset_name="tmp_coco"
    )
    mega_dataset.merge_samples(coco)

    # ==========================================
    # DATASET 2: CrowdHuman via HuggingFace (Foco em Oclusão)
    # ==========================================
    print("👥 A transferir CrowdHuman...")
    crowd = fouh.load_from_hub("jamarks/CrowdHuman-train")
    
    # O CrowdHuman pode trazer labels como 'person'. Juntamos ao mega_dataset.
    mega_dataset.merge_samples(crowd)

    # ==========================================
    # DATASET 3: Open Images V7 (Foco em Casos Raros e Diversidade)
    # ==========================================
    print("🌍 A transferir Open Images V7...")
    # ATENÇÃO: O Open Images usa "Person" (Maiúsculo). 
    oi = foz.load_zoo_dataset(
        "open-images-v7",
        splits=["train"],
        label_types=["detections"],
        classes=["Person"], 
        max_samples=2000,
        dataset_name="tmp_oi"
    )
    
    # 💡 O TRUQUE DE MESTRE: Renomear "Person" para "person" antes de fundir
    # Isso garante que o YOLO veja tudo como a classe 0
    mapa_de_classes = {"Person": "person"}
    oi.map_labels("ground_truth", mapa_de_classes)
    
    mega_dataset.merge_samples(oi)

    # ==========================================
    # RESUMO E EXPORTAÇÃO
    # ==========================================
    print(f"✅ Fusão concluída! O Mega Dataset tem agora {len(mega_dataset)} imagens.")

    export_dir = "dataset_yolo_mega"
    print(f"⚙️ A exportar para o formato YOLOv11 no diretório: {export_dir} ...")
    
    # Ao exportar, forçamos a classe "person" a ser a única exportada, ignorando 
    # acidentalmente carros ou outros objetos que vieram de "arrasto" nas imagens
    mega_dataset.export(
        export_dir=export_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        classes=["person"]
    )
    
    print("🎯 Concluído! O seu modelo está pronto para um treino de elite.")

if __name__ == "__main__":
    criar_mega_dataset()