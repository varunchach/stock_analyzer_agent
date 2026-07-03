"""Custom exceptions mapped to safe HTTP responses in main.py. Internal
details (stack traces, raw tool errors) are logged but never sent to the client.
"""


class AppError(Exception):
    """Base class — carries a safe, user-facing message and an HTTP status."""

    status_code = 500
    safe_message = "Something went wrong. Please try again."

    def __init__(self, safe_message: str | None = None, **context):
        self.safe_message = safe_message or self.safe_message
        self.context = context
        super().__init__(self.safe_message)


class GuardrailViolation(AppError):
    status_code = 422

    def __init__(self, stage: str, reason: str, category: str | None = None):
        self.stage = stage
        self.reason = reason
        self.category = category
        super().__init__(safe_message=reason, stage=stage, category=category)


class UpstreamServiceError(AppError):
    """LLM, Tavily, or Yahoo Finance failed / timed out."""

    status_code = 503
    safe_message = "A data provider is temporarily unavailable. Please try again shortly."


class InvalidRequestError(AppError):
    status_code = 400
