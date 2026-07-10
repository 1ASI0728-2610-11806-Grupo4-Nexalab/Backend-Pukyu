import cv2
import numpy as np
import time
import os
from ultralytics import YOLO
from app.core.config import get_settings
from app.models.schemas import Detection, BoundingBox, VisionResponse

# Lazy-load del modelo YOLO (se descarga automáticamente la primera vez)
_yolo_model: YOLO | None = None


def get_yolo_model() -> YOLO:
    """Devuelve el modelo YOLO singleton (lazy-loaded)."""
    global _yolo_model
    if _yolo_model is None:
        settings = get_settings()
        model_name = settings.YOLO_MODEL + ".pt"
        _yolo_model = YOLO(model_name)
    return _yolo_model


# Lazy-load de EasyOCR (tarda ~2s en iniciar)
_ocr_reader = None


def get_ocr_reader():
    """Devuelve el reader de EasyOCR singleton (lazy-loaded)."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(['en', 'es'], gpu=False)
    return _ocr_reader


def extract_dominant_color(image: np.ndarray) -> str:
    """Extrae el color predominante de una imagen de vehículo usando HSV."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Calcular el histograma de Hue
    h_hist = cv2.calcHist([hsv], [0], None, [18], [0, 180])
    s_mean = np.mean(hsv[:, :, 1])
    v_mean = np.mean(hsv[:, :, 2])
    
    # Si la saturación es baja, es blanco/gris/negro
    if s_mean < 40:
        if v_mean > 180:
            return "Blanco"
        elif v_mean < 60:
            return "Negro"
        else:
            return "Gris"
    
    # Determinar color por el pico del histograma de Hue
    dominant_hue = np.argmax(h_hist) * 10  # Cada bin = 10 grados
    
    color_map = {
        (0, 15): "Rojo",
        (15, 35): "Naranja",
        (35, 75): "Amarillo",
        (75, 150): "Verde",
        (150, 195): "Azul Celeste",
        (195, 260): "Azul",
        (260, 290): "Violeta",
        (290, 340): "Rosado",
        (340, 360): "Rojo",
    }
    
    for (low, high), color_name in color_map.items():
        if low <= dominant_hue < high:
            return color_name
    
    return "Indeterminado"


def read_license_plate(plate_image: np.ndarray) -> tuple[str, float]:
    """Lee los caracteres de una placa usando EasyOCR.
    Retorna (texto, confianza).
    """
    reader = get_ocr_reader()
    
    # Preprocesar: escala de grises + aumento de contraste
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    results = reader.readtext(gray)
    
    if not results:
        return ("", 0.0)
    
    # Concatenar todos los textos detectados
    full_text = " ".join([r[1] for r in results])
    avg_confidence = sum(r[2] for r in results) / len(results)
    
    # Limpiar caracteres no alfanuméricos
    clean_text = "".join(c for c in full_text if c.isalnum() or c == '-').upper()
    
    return (clean_text, round(avg_confidence, 4))


async def analyze_image(image_bytes: bytes) -> VisionResponse:
    """Analiza una imagen completa: detecta vehículos, color y placa.
    
    Args:
        image_bytes: Bytes de la imagen subida por el frontend.
    
    Returns:
        VisionResponse con estado de ocupación y detecciones.
    """
    start_time = time.time()
    
    # Decodificar la imagen
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return VisionResponse(
            status="ERROR",
            processing_time_ms=0,
            model_version="yolov8n",
            detections=[]
        )
    
    model = get_yolo_model()
    
    # Ejecutar YOLO
    results = model(img, verbose=False)
    
    detections: list[Detection] = []
    has_vehicle = False
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Clases COCO para vehículos: 2=car, 3=motorcycle, 5=bus, 7=truck
            if cls_id in [2, 3, 5, 7] and conf > 0.4:
                has_vehicle = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Extraer color del vehículo
                vehicle_crop = img[y1:y2, x1:x2]
                color = extract_dominant_color(vehicle_crop) if vehicle_crop.size > 0 else "Indeterminado"
                
                detections.append(Detection(
                    type="vehicle",
                    confidence=round(conf, 4),
                    bounding_box=BoundingBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1),
                    color=color
                ))
                
                # Intentar leer placa (zona inferior del vehículo)
                plate_region_y1 = y1 + int((y2 - y1) * 0.6)
                plate_crop = img[plate_region_y1:y2, x1:x2]
                
                if plate_crop.size > 0:
                    plate_text, plate_conf = read_license_plate(plate_crop)
                    if plate_text and plate_conf > 0.3:
                        detections.append(Detection(
                            type="license_plate",
                            confidence=plate_conf,
                            bounding_box=BoundingBox(
                                x=x1,
                                y=plate_region_y1,
                                w=x2 - x1,
                                h=y2 - plate_region_y1
                            ),
                            text=plate_text
                        ))
    
    processing_time = int((time.time() - start_time) * 1000)
    
    return VisionResponse(
        status="OCCUPIED" if has_vehicle else "EMPTY",
        processing_time_ms=processing_time,
        model_version=get_settings().YOLO_MODEL,
        detections=detections
    )


async def analyze_video(video_path: str, frame_interval: int = 30) -> list[VisionResponse]:
    """Analiza un video frame-by-frame extrayendo detecciones cada N frames.
    
    Args:
        video_path: Ruta al archivo de video.
        frame_interval: Cada cuántos frames analizar (default: 1 por segundo a 30fps).
    
    Returns:
        Lista de VisionResponse, uno por cada frame analizado.
    """
    if not os.path.exists(video_path):
        return []
    
    cap = cv2.VideoCapture(video_path)
    frame_results: list[VisionResponse] = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            # Codificar frame a bytes y reutilizar analyze_image
            _, buffer = cv2.imencode('.jpg', frame)
            result = await analyze_image(buffer.tobytes())
            frame_results.append(result)
        
        frame_count += 1
    
    cap.release()
    return frame_results
