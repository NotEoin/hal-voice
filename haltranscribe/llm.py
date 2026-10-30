"""Model access for hal-transcribe: the chat client lives in `halvoice.llm`."""

from halvoice.llm import Unavailable, chat

__all__ = ["Unavailable", "chat"]
