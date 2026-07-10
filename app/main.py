from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import get_settings
from app.api.vision import router as vision_router
from app.api.chatbot import router as chatbot_router

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Backend de Inteligencia Artificial para Parky: Visión Computacional (YOLOv8) y Chatbot (Llama 3 vía Groq).",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — Permitir que el frontend Angular se conecte
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",        # Angular dev server
        "http://localhost:4000",        # Angular SSR dev server
        "https://frontend-parky.vercel.app", # Vercel Frontend
        "https://*.vercel.app",         # Si el frontend se despliega en Vercel (subdominios)
        "*",                            # Temporalmente abierto para desarrollo
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar los routers
app.include_router(vision_router)
app.include_router(chatbot_router)


@app.get("/", tags=["Health"])
async def root():
    """Health check del backend."""
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "services": {
            "vision": "YOLOv8 + OpenCV + EasyOCR",
            "chatbot": "Llama 3 via Groq API",
            "database": "Supabase (PostgreSQL)"
        }
    }


@app.get("/health", tags=["Health"])
async def health():
    """Health check detallado para Koyeb."""
    return {"status": "healthy"}
