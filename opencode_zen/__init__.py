# -*- coding: utf-8 -*-

from opencode_zen.config import APP_VERSION as __version__
__author__ = "AsimAftab"

from opencode_zen.cache import ModelInfoCache
from opencode_zen.http_client import KiroHttpClient
from opencode_zen.routes_openai import router

from opencode_zen.config import (
    PROXY_API_KEY,
    HIDDEN_MODELS,
    APP_VERSION,
)

from opencode_zen.models_openai import (
    ChatCompletionRequest,
    ChatMessage,
    OpenAIModel,
    ModelList,
)

from opencode_zen.converters_openai import build_opencode_payload
from opencode_zen.converters_core import (
    extract_text_content,
    merge_adjacent_messages,
)

from opencode_zen.exceptions import (
    validation_exception_handler,
    sanitize_validation_errors,
)

__all__ = [
    "__version__",
    "ModelInfoCache",
    "KiroHttpClient",
    "router",
    "PROXY_API_KEY",
    "HIDDEN_MODELS",
    "APP_VERSION",
    "ChatCompletionRequest",
    "ChatMessage",
    "OpenAIModel",
    "ModelList",
    "build_opencode_payload",
    "extract_text_content",
    "merge_adjacent_messages",
    "validation_exception_handler",
    "sanitize_validation_errors",
]