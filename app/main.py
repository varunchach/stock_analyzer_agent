"""FastAPI entrypoint — request/response flow, error handling, logging, and
the streaming endpoint that powers the UI's "agent thinking" panel.
"""
import json
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.errors import AppError
from app.logging_config import configure_logging, get_logger, log_event
from app.pipeline import run_pipeline, stream_pipeline
from app.schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse

configure_logging(settings.log_level)
logger = get_logger("main")

app = FastAPI(title="TradeSetu Stock Analyzer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    trace_id = str(uuid.uuid4())
    request.state.trace_id = trace_id
    start = time.perf_counter()
    log_event(logger, "info", "Request started", trace_id, path=request.url.path, method=request.method)
    try:
        response = await call_next(request)
    except Exception as exc:
        log_event(logger, "error", "Unhandled exception", trace_id, error=str(exc))
        return JSONResponse(status_code=500, content={"error": "Internal server error", "trace_id": trace_id})
    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    response.headers["X-Trace-Id"] = trace_id
    log_event(logger, "info", "Request finished", trace_id, status_code=response.status_code, duration_ms=duration_ms)
    return response


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    trace_id = getattr(request.state, "trace_id", "unknown")
    log_event(logger, "warning", "Handled AppError", trace_id, error=exc.safe_message, **exc.context)
    return JSONResponse(status_code=exc.status_code, content={"error": exc.safe_message, "trace_id": trace_id})


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", llm_model=settings.llm_model, guard_model=settings.guard_model)


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest, request: Request):
    trace_id = request.state.trace_id
    thread_id = payload.thread_id or trace_id
    return run_pipeline(payload.query, thread_id, trace_id)


@app.post("/api/v1/analyze/stream")
async def analyze_stream(payload: AnalyzeRequest, request: Request):
    trace_id = request.state.trace_id
    thread_id = payload.thread_id or trace_id

    def event_source():
        for event in stream_pipeline(payload.query, thread_id, trace_id):
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")


STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
