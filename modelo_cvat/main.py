import json
import base64
from PIL import Image
import io
import numpy as np
from ultralytics import YOLO

def init_context(context):
    context.logger.info("Iniciando IA SenseClean com Suporte GPU...")
    # Caminho absoluto dentro do container
    model = YOLO("/opt/nuclio/best.pt")
    context.user_data.model = model
    context.logger.info("Modelo YOLO carregado com sucesso.")

def handler(context, event):
    context.logger.info("Recebendo requisição de inferência...")
    try:
        data = event.body
        if isinstance(data, dict):
            buf = io.BytesIO(base64.b64decode(data["image"]))
        else:
            buf = io.BytesIO(data)

        image = Image.open(buf).convert("RGB")
        
        # Inferência
        results = context.user_data.model(image, conf=0.5)[0]
        
        encoded_results = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label_name = results.names[class_id]
            
            encoded_results.append({
                "confidence": str(round(conf, 2)),
                "label": label_name,
                "points": [x1, y1, x2, y2],
                "type": "rectangle",
            })

        return context.Response(body=json.dumps(encoded_results),
                                headers={},
                                content_type='application/json',
                                status_code=200)
    except Exception as e:
        context.logger.error(f"Erro na inferência: {str(e)}")
        return context.Response(body=str(e), status_code=500)