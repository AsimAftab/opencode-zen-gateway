# -*- coding: utf-8 -*-

# OpenCode Zen Gateway
# https://github.com/AsimAftab/opencode-zen-gateway
# Copyright (C) 2026 AsimAftab
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Exception handlers for OpenCode Zen Gateway.

Contains functions for handling validation errors and other exceptions
in a JSON-serialization compatible format.
"""

from typing import Any, List, Dict, Optional

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.exceptions import HTTPException as StarletteHTTPException


# Map HTTP status codes to Anthropic error `type` strings so gateway-originated
# errors carry the same taxonomy the real Messages API uses.
_ANTHROPIC_ERROR_TYPES: Dict[int, str] = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    422: "invalid_request_error",
    429: "rate_limit_error",
    500: "api_error",
    502: "api_error",
    503: "api_error",
    504: "api_error",
    529: "overloaded_error",
}


def _is_anthropic_path(path: Any) -> bool:
    """Returns True for Anthropic-dialect endpoints (/v1/messages...)."""
    return isinstance(path, str) and path.startswith("/v1/messages")


def _is_openai_path(path: Any) -> bool:
    """Returns True for OpenAI-dialect endpoints."""
    return isinstance(path, str) and (
        path.startswith("/v1/chat/completions") or path.startswith("/v1/models")
    )


def _anthropic_error_type(status_code: int) -> str:
    """Maps a status code to an Anthropic error type, defaulting to api_error."""
    return _ANTHROPIC_ERROR_TYPES.get(status_code, "api_error")


def _anthropic_error_body(status_code: int, message: str) -> Dict[str, Any]:
    """Builds an Anthropic-shaped error envelope."""
    return {
        "type": "error",
        "error": {"type": _anthropic_error_type(status_code), "message": message},
    }


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """
    Renders HTTPExceptions in the dialect of the endpoint that raised them.

    FastAPI's default handler wraps every HTTPException as {"detail": ...}.
    Claude Code and OpenAI clients expect their own error envelopes, so
    gateway-originated errors (401 auth failures, 502/504 transport failures)
    are reshaped here based on the request path. Unknown paths keep the default
    {"detail": ...} shape.

    Args:
        request: FastAPI Request object.
        exc: The raised Starlette/FastAPI HTTPException.

    Returns:
        JSONResponse in the appropriate error shape, preserving any headers.
    """
    path = request.url.path
    detail = exc.detail
    headers: Optional[Dict[str, str]] = getattr(exc, "headers", None)

    # Capture debug logs for gateway-originated errors on the API endpoints
    # (e.g. 401 auth failures, re-raised 502/504 transport failures) so
    # DEBUG_MODE=errors records them, not only 422 validation errors.
    if _is_anthropic_path(path) or _is_openai_path(path):
        try:
            from opencode_zen.debug_logger import debug_logger
            if debug_logger:
                debug_logger.flush_on_error(exc.status_code, str(detail))
        except ImportError:
            pass

    if _is_anthropic_path(path):
        # An already-shaped Anthropic error (e.g. the 401 auth detail) passes through.
        if isinstance(detail, dict) and detail.get("type") == "error" and "error" in detail:
            content: Dict[str, Any] = detail
        elif isinstance(detail, dict):
            message = detail.get("message") or detail.get("detail") or str(detail)
            content = _anthropic_error_body(exc.status_code, str(message))
        else:
            content = _anthropic_error_body(exc.status_code, str(detail))
        return JSONResponse(status_code=exc.status_code, content=content, headers=headers)

    if _is_openai_path(path):
        if isinstance(detail, dict) and "error" in detail:
            content = detail
        elif isinstance(detail, dict):
            message = detail.get("message") or str(detail)
            content = {"error": {"type": "api_error", "message": str(message)}}
        else:
            content = {"error": {"type": "api_error", "message": str(detail)}}
        return JSONResponse(status_code=exc.status_code, content=content, headers=headers)

    # Unknown path — keep FastAPI's default envelope.
    return JSONResponse(status_code=exc.status_code, content={"detail": detail}, headers=headers)


def sanitize_validation_errors(errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts validation errors to JSON-serializable format.
    
    Pydantic may include bytes objects in the 'input' field, which
    are not JSON-serializable. This function converts them to strings.
    
    Args:
        errors: List of validation errors from Pydantic
    
    Returns:
        List of errors with bytes converted to strings
    """
    sanitized = []
    for error in errors:
        sanitized_error = {}
        for key, value in error.items():
            if isinstance(value, bytes):
                # Convert bytes to string
                sanitized_error[key] = value.decode("utf-8", errors="replace")
            elif isinstance(value, (list, tuple)):
                # Recursively process lists
                sanitized_error[key] = [
                    v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v
                    for v in value
                ]
            else:
                sanitized_error[key] = value
        sanitized.append(sanitized_error)
    return sanitized


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Pydantic validation error handler.
    
    Logs error details and returns an informative response.
    Correctly handles bytes objects in errors by converting them to strings.
    Also flushes debug logs for validation errors when DEBUG_MODE is enabled.
    
    Args:
        request: FastAPI Request object
        exc: Validation exception from Pydantic
    
    Returns:
        JSONResponse with error details and status 422
    """
    body = await request.body()
    body_str = body.decode("utf-8", errors="replace")
    
    # Sanitize errors for JSON serialization
    sanitized_errors = sanitize_validation_errors(exc.errors())
    
    logger.error(f"Validation error (422): {sanitized_errors}")
    # Log body at DEBUG level to avoid cluttering console with potentially large payloads
    # logger.debug(f"Request body: {body_str[:500]}...")
    
    # Flush debug logs for validation errors
    # This is called AFTER middleware has initialized debug logging,
    # so all app logs during request processing will be captured
    try:
        from opencode_zen.debug_logger import debug_logger
        if debug_logger:
            error_message = f"Validation error: {sanitized_errors}"
            debug_logger.flush_on_error(422, error_message)
    except ImportError:
        pass  # debug_logger not available

    # Claude Code expects the Anthropic error envelope; the real Messages API
    # returns 400 invalid_request_error (not 422) for malformed requests.
    if _is_anthropic_path(request.url.path):
        message = _summarize_validation_errors(sanitized_errors)
        return JSONResponse(
            status_code=400,
            content=_anthropic_error_body(400, message),
        )

    return JSONResponse(
        status_code=422,
        content={"detail": sanitized_errors, "body": body_str[:500]},
    )


def _summarize_validation_errors(errors: List[Dict[str, Any]]) -> str:
    """
    Renders Pydantic validation errors as a single human-readable message.

    Args:
        errors: Sanitized validation errors.

    Returns:
        A compact "loc: msg; loc: msg" summary for the error envelope.
    """
    parts: List[str] = []
    for err in errors:
        loc = err.get("loc", [])
        loc_str = ".".join(str(p) for p in loc) if isinstance(loc, (list, tuple)) else str(loc)
        msg = err.get("msg", "invalid value")
        parts.append(f"{loc_str}: {msg}" if loc_str else str(msg))
    return "; ".join(parts) or "Request validation failed"