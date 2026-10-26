"""
main.py - FastAPI Application Entry Point for Candidate Line Breach Intelligence (CLBI)
"""

import os
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.routes import router
from src.api.ndws_routes import ndws_router

app = FastAPI(
    title="Candidate Line Breach Intelligence (CLBI) API",
    description=(
        "CLBI REST API provides an experimental model score for generated candidate-line examples. "
        "The model is trained on synthetic features and labels, not verified incident observations; "
        "it does not consume live telemetry and its rankings are not validated for operational use."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configure Development CORS
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Exception Handlers for Structured Clean Error Responses
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    if errors:
        first_err = errors[0]
        msg = first_err.get("msg", "Validation error")
        loc = " -> ".join([str(x) for x in first_err.get("loc", [])])
        detailed_msg = f"{loc}: {msg}" if loc else msg
    else:
        detailed_msg = str(exc)
        
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={
            "error": "validation_error",
            "message": detailed_msg
        }
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if isinstance(exc.detail, dict):
        error_code = exc.detail.get("error", "http_error")
        message = exc.detail.get("message", str(exc.detail))
    else:
        error_code = "http_error"
        message = str(exc.detail)
        
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": error_code,
            "message": message
        }
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "validation_error",
            "message": str(exc)
        }
    )

# Include API routers
app.include_router(router)
app.include_router(ndws_router)

# Mount Static Assets and Dashboard Frontend
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        return FileResponse(os.path.join(static_dir, "index.html"))

    @app.get("/demo", include_in_schema=False)
    async def serve_demo():
        return FileResponse(os.path.join(static_dir, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
