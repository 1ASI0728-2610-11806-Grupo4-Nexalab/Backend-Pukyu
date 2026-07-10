from pydantic import BaseModel, Field
from typing import Optional


# ─── Vision Schemas ───

class BoundingBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class Detection(BaseModel):
    type: str = Field(description="Tipo de detección: 'vehicle' o 'license_plate'")
    confidence: float = Field(ge=0, le=1, description="Nivel de confianza (0.0 a 1.0)")
    bounding_box: BoundingBox
    color: Optional[str] = Field(None, description="Color predominante del vehículo")
    text: Optional[str] = Field(None, description="Texto leído de la placa")


class VisionResponse(BaseModel):
    status: str = Field(description="OCCUPIED o EMPTY")
    processing_time_ms: int
    model_version: str
    detections: list[Detection] = []


# ─── Chatbot Schemas ───

class ChatRequest(BaseModel):
    thread_id: Optional[str] = Field(None, description="ID del hilo de chat (si existe)")
    user_message: str = Field(min_length=1, description="Mensaje del usuario")
    garage_id: Optional[str] = Field(None, description="ID de la cochera para contexto")


class ChatResponse(BaseModel):
    reply: str = Field(description="Respuesta generada por Parky AI")
    thread_id: str = Field(description="ID del hilo de chat")
    is_ai_generated: bool = True


# ─── Health Check ───

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    services: dict
