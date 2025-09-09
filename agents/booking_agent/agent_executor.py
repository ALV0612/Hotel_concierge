import asyncio
import logging
from typing import Dict, List
from typing_extensions import override

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message

# Optional types
try:
    from a2a.types import TextPart  # noqa: F401
except Exception:
    TextPart = object  # fallback

from agents.booking_agent.agent import BookingAgent

logger = logging.getLogger(__name__)

# MESSAGE LIMITS to prevent SQLite blob overflow
MAX_MESSAGE_LENGTH = 800  # Truncate messages longer than this
MAX_CONTEXT_LENGTH = 600  # Truncate context portion


def _truncate_message(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> str:
    """Truncate message safely to prevent SQLite blob overflow"""
    if not text or len(text) <= max_length:
        return text
    
    # Try to truncate at natural break points
    truncated = text[:max_length]
    
    # Look for last sentence ending within reasonable range
    for punct in ['. ', '.\n', '? ', '! ']:
        last_punct = truncated.rfind(punct)
        if last_punct > max_length * 0.7:  # At least 70% of desired length
            return truncated[:last_punct + 1] + " [...]"
    
    # Look for last word boundary
    last_space = truncated.rfind(' ')
    if last_space > max_length * 0.8:  # At least 80% of desired length
        return truncated[:last_space] + " [...]"
    
    # Hard truncate as last resort
    return truncated + " [...]"


def _extract_text_from_context(ctx: RequestContext) -> str:
    """Extract text from A2A Message parts with smart truncation."""
    msg = getattr(ctx, "message", None)
    if not msg:
        logger.debug("No message in context")
        return ""

    parts: List[object] = getattr(msg, "parts", []) or []
    logger.debug(f"Message parts: {len(parts)} items")

    out: List[str] = []
    for p in parts:
        # 1) Nested Part(root=TextPart(...))
        if hasattr(p, "root"):
            root = getattr(p, "root")
            txt = getattr(root, "text", None)
            if isinstance(txt, str) and txt.strip():
                out.append(txt.strip())
                continue

        # 2) Direct TextPart
        txt = getattr(p, "text", None)
        if isinstance(txt, str) and txt.strip():
            out.append(txt.strip())
            continue

        # 3) DataPart-like
        data = getattr(p, "data", None)
        if isinstance(data, dict):
            for key in ("text", "query", "message", "prompt", "input"):
                val = data.get(key)
                if isinstance(val, str) and val.strip():
                    out.append(val.strip())
                    break

    result = " ".join(out).strip()
    
    # TRUNCATE to prevent SQLite overflow
    if len(result) > MAX_MESSAGE_LENGTH:
        logger.warning(f"Message truncated from {len(result)} to {MAX_MESSAGE_LENGTH} chars")
        result = _truncate_message(result)
    
    logger.debug(f"Final extracted text: '{result[:100]}...' (length: {len(result)})")
    return result


def _get_session_id(ctx: RequestContext) -> str:
    """Lấy session id ổn định (fallback về fixed)."""
    FIXED_SESSION = "ohana-permanent-session"

    # Try various ways to get session ID
    sid = getattr(ctx, "session_id", None) or getattr(ctx, "sessionId", None)
    
    # Try from request params
    req = getattr(ctx, "request", None)
    if req and hasattr(req, "params"):
        params = req.params
        sid = sid or getattr(params, "sessionId", None) or getattr(params, "conversationId", None)
    
    # Try from session object
    session = getattr(ctx, "session", None)
    if session:
        sid = sid or getattr(session, "id", None)

    if sid:
        logger.debug(f"Using session: {sid}")
        return str(sid)

    logger.debug(f"Using default session: {FIXED_SESSION}")
    return FIXED_SESSION


class BookingAgentExecutor(AgentExecutor):
    """
    A2A Standard BookingAgentExecutor with message truncation:
    
    - Follows A2A protocol exactly
    - Uses framework's event handling
    - Single BookingAgent instance (resource efficient)
    - Per-session locking for thread safety
    - MESSAGE TRUNCATION to prevent SQLite blob overflow
    """

    def __init__(self, max_sessions: int = 100, **kwargs):
        self._max_sessions = max_sessions
        if kwargs:
            logger.debug("BookingAgentExecutor: ignore extra kwargs: %s", list(kwargs.keys()))

        self._agent: BookingAgent | None = None
        self._initialized: bool = False
        self._init_lock = asyncio.Lock()
        self._session_locks: Dict[str, asyncio.Lock] = {}

    def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create session-specific lock"""
        lock = self._session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[session_id] = lock
        return lock

    async def _ensure_agent(self):
        """Ensure BookingAgent singleton is initialized"""
        if self._initialized and self._agent is not None:
            return
        async with self._init_lock:
            if self._initialized and self._agent is not None:
                return
            logger.info("Initializing BookingAgent singleton...")
            self._agent = BookingAgent()
            await self._agent.initialize()
            self._initialized = True
            logger.info("BookingAgent singleton ready")

    @override
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        A2A Standard execute method with message truncation.
        """
        try:
            # 1) Extract input & session with TRUNCATION
            query = _extract_text_from_context(context)
            session_id = _get_session_id(context)

            logger.info(f"Processing request: session={session_id}, text_len={len(query)}")

            if not query:
                await event_queue.enqueue_event(new_agent_text_message("Xin hãy nhập yêu cầu đặt phòng."))
                return

            # 2) Ensure agent is ready
            await self._ensure_agent()
            agent = self._agent
            if agent is None:
                await event_queue.enqueue_event(new_agent_text_message("❌ Lỗi: Agent chưa sẵn sàng."))
                return

            # 3) Process with session lock
            lock = self._get_session_lock(session_id)
            async with lock:
                logger.debug("Calling BookingAgent...")
                
                # ADDITIONAL TRUNCATION before sending to agent
                safe_query = _truncate_message(query, MAX_MESSAGE_LENGTH)
                if len(safe_query) != len(query):
                    logger.info(f"Query further truncated for agent: {len(query)} -> {len(safe_query)}")
                
                response_text = await agent.chat(safe_query, session_id=session_id)
                logger.debug(f"Agent response: '{response_text[:100]}...'")
                
                # 4) Send response through EventQueue
                if response_text and response_text.strip():
                    await event_queue.enqueue_event(new_agent_text_message(response_text))
                    logger.debug("Response event enqueued")
                else:
                    # Fallback
                    fallback_msg = "Xin lỗi, tôi không thể xử lý yêu cầu này."
                    await event_queue.enqueue_event(new_agent_text_message(fallback_msg))
                    logger.warning("Used fallback response")
                    
        except Exception as e:
            logger.exception(f"BookingAgent execution error: {e}")
            error_response = f"❌ Lỗi hệ thống: {str(e)}"
            await event_queue.enqueue_event(new_agent_text_message(error_response))

    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel not supported for BookingAgent"""
        raise Exception("Cancel not supported for BookingAgent")

    async def cleanup(self):
        """Cleanup on shutdown"""
        if self._agent and hasattr(self._agent, "aclose"):
            try:
                await self._agent.aclose()
                logger.info("BookingAgent cleanup completed")
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")
        self._agent = None
        self._initialized = False
        self._session_locks.clear()