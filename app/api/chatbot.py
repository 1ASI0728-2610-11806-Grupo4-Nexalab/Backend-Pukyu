from fastapi import APIRouter, HTTPException
from app.services.chatbot_service import chat
from app.models.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api/v1/chat", tags=["Chatbot IA"])


@router.post("/message", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    """Envía un mensaje al chatbot Parky AI y recibe una respuesta inteligente.
    
    El chatbot usa Llama 3 (vía Groq) con contexto real de las cocheras
    obtenido directamente de la base de datos Supabase (RAG ligero).
    
    - Si se proporciona un `garage_id`, el chatbot consultará los datos
      reales de esa cochera (precio, altura, disponibilidad) para dar
      respuestas precisas.
    - Si se proporciona un `thread_id`, se cargará el historial de la
      conversación para mantener coherencia.
    """
    try:
        result = await chat(
            user_message=request.user_message,
            thread_id=request.thread_id,
            garage_id=request.garage_id
        )
        
        return ChatResponse(
            reply=result["reply"],
            thread_id=result["thread_id"],
            is_ai_generated=result["is_ai_generated"]
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar el mensaje: {str(e)}"
        )
