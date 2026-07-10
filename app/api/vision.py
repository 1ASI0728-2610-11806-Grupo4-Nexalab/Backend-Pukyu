import os
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.services.vision_service import analyze_image, analyze_video
from app.models.schemas import VisionResponse

router = APIRouter(prefix="/api/v1/cv", tags=["Visión Artificial"])


@router.post("/analyze-image", response_model=VisionResponse)
async def analyze_image_endpoint(file: UploadFile = File(...)):
    """Recibe una imagen y retorna detecciones de vehículos, color y placa.
    
    El frontend sube una imagen (simulando la cámara IoT) y el backend
    ejecuta YOLOv8 + OpenCV + EasyOCR para determinar:
    - Si la cochera está OCUPADA o VACÍA.
    - El color predominante del vehículo.
    - Los caracteres de la placa (si son legibles).
    """
    # Validar que sea una imagen
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen (jpg, png, etc.)")
    
    # Leer los bytes de la imagen
    image_bytes = await file.read()
    
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="La imagen está vacía.")
    
    # Procesar con el servicio de visión
    result = await analyze_image(image_bytes)
    
    return result


@router.post("/analyze-video")
async def analyze_video_endpoint(
    video_name: str = Query(..., description="Nombre del video en simulation/sample_videos/"),
    frame_interval: int = Query(30, description="Cada cuántos frames analizar")
):
    """Analiza un video pregrabado de la carpeta de simulación.
    
    El frontend envía el nombre del video y el backend lo procesa
    frame-by-frame, reportando cambios de estado de la cochera.
    """
    video_path = os.path.join("simulation", "sample_videos", video_name)
    
    if not os.path.exists(video_path):
        raise HTTPException(
            status_code=404,
            detail=f"Video '{video_name}' no encontrado en simulation/sample_videos/"
        )
    
    results = await analyze_video(video_path, frame_interval)
    
    # Resumen del análisis
    occupied_frames = sum(1 for r in results if r.status == "OCCUPIED")
    total_frames = len(results)
    
    return {
        "video": video_name,
        "total_frames_analyzed": total_frames,
        "occupied_frames": occupied_frames,
        "empty_frames": total_frames - occupied_frames,
        "final_status": results[-1].status if results else "UNKNOWN",
        "frame_details": results
    }


@router.get("/sample-videos")
async def list_sample_videos():
    """Lista los videos disponibles en la carpeta de simulación."""
    video_dir = os.path.join("simulation", "sample_videos")
    
    if not os.path.exists(video_dir):
        return {"videos": []}
    
    videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    return {"videos": videos}
