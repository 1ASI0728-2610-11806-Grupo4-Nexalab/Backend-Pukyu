from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.core.config import get_settings
from app.core.supabase_client import get_supabase

# System prompt que define la personalidad de Parky AI
SYSTEM_PROMPT = """Eres "Parky AI", el asistente virtual inteligente de la aplicación de cocheras Parky.

Tu personalidad:
- Eres amable, profesional y conciso.
- Respondes siempre en español.
- Tienes un tono ligeramente divertido pero respetuoso.

Tu función:
- Ayudar a conductores con preguntas sobre cocheras (precios, disponibilidad, características).
- Ayudar a propietarios con dudas sobre su gestión.
- Dar recomendaciones basadas en los datos reales de las cocheras.

Reglas estrictas:
- Si no sabes algo, dilo honestamente. NUNCA inventes datos de cocheras.
- Si te dan contexto de una cochera específica, usa esos datos para responder con precisión.
- No respondas preguntas que no estén relacionadas con estacionamiento o cocheras.
- Mantén tus respuestas breves (máximo 3-4 oraciones) a menos que te pidan más detalle.
"""


def _get_llm() -> ChatGroq:
    """Crea una instancia del LLM de Groq con Llama 3."""
    settings = get_settings()
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=500,
    )


async def get_garage_context(garage_id: str) -> str:
    """Consulta Supabase para obtener contexto real de una cochera.
    
    Retorna un texto con los datos relevantes de la cochera.
    """
    supabase = get_supabase()
    
    # Obtener datos de la cochera con sus spots
    result = supabase.table('garages').select(
        '*, parking_spots(spot_label, default_price_per_hour, is_active)'
    ).eq('id', garage_id).single().execute()
    
    if not result.data:
        return "No se encontró información sobre esta cochera."
    
    g = result.data
    spots = g.get('parking_spots', [])
    active_spots = [s for s in spots if s.get('is_active')]
    
    context = f"""
DATOS REALES DE LA COCHERA:
- Dirección: {g.get('address', 'N/A')}
- Distrito: {g.get('district', 'N/A')}
- Techada: {'Sí' if g.get('is_covered') else 'No'}
- Cargador eléctrico: {'Sí' if g.get('has_electric_charger') else 'No'}
- Cámara IA activa: {'Sí' if g.get('camera_enabled') else 'No'}
- Límite de altura: {g.get('height_limit_mts', 'Sin restricción')} metros
- Espacios disponibles: {len(active_spots)} de {len(spots)}
- Precio por hora: S/ {active_spots[0]['default_price_per_hour'] if active_spots else 'N/A'}
"""
    return context


async def get_chat_history(thread_id: str, limit: int = 10) -> list:
    """Recupera los últimos N mensajes de un hilo de chat desde Supabase."""
    supabase = get_supabase()
    
    result = supabase.table('chat_messages').select('*').eq(
        'thread_id', thread_id
    ).order('created_at', desc=False).limit(limit).execute()
    
    messages = []
    for msg in (result.data or []):
        if msg.get('is_ai_generated'):
            messages.append(AIMessage(content=msg['content']))
        else:
            messages.append(HumanMessage(content=msg['content']))
    
    return messages


async def chat(user_message: str, thread_id: str | None = None, garage_id: str | None = None) -> dict:
    """Procesa un mensaje del usuario y retorna la respuesta de Parky AI.
    
    Args:
        user_message: Mensaje escrito por el usuario.
        thread_id: ID del hilo de chat (opcional, para mantener contexto).
        garage_id: ID de la cochera (opcional, para contexto RAG).
    
    Returns:
        Dict con reply, thread_id e is_ai_generated.
    """
    llm = _get_llm()
    
    # Construir los mensajes para el LLM
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    
    # Agregar contexto de la cochera si se proporciona
    if garage_id:
        context = await get_garage_context(garage_id)
        messages.append(SystemMessage(content=context))
    
    # Agregar historial del chat si existe un thread
    if thread_id:
        history = await get_chat_history(thread_id)
        messages.extend(history)
    
    # Agregar el mensaje actual del usuario
    messages.append(HumanMessage(content=user_message))
    
    # Llamar al LLM
    response = llm.invoke(messages)
    ai_reply = response.content
    
    # Guardar los mensajes en Supabase
    supabase = get_supabase()
    
    # Si no hay thread_id, crear uno nuevo (simplificado para MVP)
    if not thread_id:
        import uuid
        thread_id = str(uuid.uuid4())
    
    # Guardar mensaje del usuario
    try:
        supabase.table('chat_messages').insert({
            'thread_id': thread_id,
            'content': user_message,
            'is_ai_generated': False
        }).execute()
        
        # Guardar respuesta de la IA
        supabase.table('chat_messages').insert({
            'thread_id': thread_id,
            'content': ai_reply,
            'is_ai_generated': True
        }).execute()
    except Exception as e:
        # No fallar si no se puede guardar (el thread podría no existir aún en la BD)
        print(f"Warning: Could not save chat messages: {e}")
    
    return {
        "reply": ai_reply,
        "thread_id": thread_id,
        "is_ai_generated": True
    }
