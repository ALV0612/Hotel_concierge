import asyncio
import logging
from uuid import uuid4
from typing import Dict, Any, List
from typing_extensions import override

# A2A SDK
from a2a.server.agent_execution import AgentExecutor as A2AExecutorBase, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message

# CrewAI Agent (đồng bộ)
from agents.get_info_agent.agent import GetInfoAgentCrew as GetInfoAgent

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


def _extract_text(ctx: RequestContext) -> str:
    """Rút text từ RequestContext.message.parts (đa dạng cấu trúc)."""
    msg = getattr(ctx, "message", None)
    if not msg:
        return ""
    parts: List[Any] = getattr(msg, "parts", []) or []
    texts: List[str] = []

    for p in parts:
        # Case: Part(root=TextPart(...))
        if hasattr(p, "root"):
            actual = p.root
            if hasattr(actual, "text") and isinstance(actual.text, str):
                t = actual.text.strip()
                if t:
                    texts.append(t)
                    continue

        # Case: direct TextPart
        if hasattr(p, "text") and isinstance(p.text, str):
            t = p.text.strip()
            if t:
                texts.append(t)
                continue

        # Fallback attr
        t = getattr(p, "text", None)
        if isinstance(t, str) and t.strip():
            texts.append(t.strip())
            continue

        # Case: data dict
        data = getattr(p, "data", None)
        if isinstance(data, dict):
            for key in ("text", "query", "message", "prompt", "input"):
                v = data.get(key)
                if isinstance(v, str) and v.strip():
                    texts.append(v.strip())
                    break

    return " ".join(texts).strip()


def _get_session_id(ctx: RequestContext) -> str:
    """Lấy session_id từ nhiều chỗ khả dĩ trong RequestContext."""
    sid = getattr(ctx, "session_id", None) or getattr(ctx, "sessionId", None)
    req = getattr(ctx, "request", None)
    if not sid and req is not None:
        params = getattr(req, "params", None)
        sid = getattr(params, "sessionId", None)
    return sid or uuid4().hex


class GetInfoAgentExecutor(A2AExecutorBase):
    """
    Executor cho CrewAI GetInfoAgent:
    - Mỗi session có 1 agent (giữ history riêng).
    - Gọi agent.ask(...) trong thread bằng asyncio.to_thread để không block event loop.
    - Không khởi tạo/đóng MCP theo kiểu async; CrewAI agent tự quản lý (threaded client).
    """

    def __init__(self, max_sessions: int = 200) -> None:
        self._agents: Dict[str, GetInfoAgent] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._max_sessions = max_sessions

    def _get_agent(self, session_id: str) -> GetInfoAgent:
        """Lấy/tạo agent cho 1 session; giới hạn số session và cleanup agent cũ."""
        if session_id not in self._agents:
            agent = GetInfoAgent()  # CrewAI agent: đồng bộ
            self._agents[session_id] = agent

            # Nếu quá giới hạn, pop agent cũ nhất và cleanup
            if len(self._agents) > self._max_sessions:
                oldest = next(iter(self._agents.keys()))
                old_agent = self._agents.pop(oldest, None)
                if old_agent:
                    asyncio.create_task(self._cleanup_agent(old_agent))
        return self._agents[session_id]

    async def _cleanup_agent(self, agent: GetInfoAgent) -> None:
        """Đóng tài nguyên agent (đồng bộ) trong thread riêng."""
        try:
            await asyncio.to_thread(agent.close)
        except Exception as e:
            logger.debug(f"Cleanup error: {e}")

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        """1 lock cho mỗi session để serialize truy vấn của cùng một phiên."""
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    @override
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        session_id = _get_session_id(context)
        user_text = _extract_text(context)

        logger.info(f"[GetInfoAgent] session={session_id} text='{user_text[:80]}...'")

        if not user_text:
            await event_queue.enqueue_event(new_agent_text_message("Xin hãy nhập yêu cầu."))
            return

        agent = self._get_agent(session_id)
        lock = self._get_lock(session_id)

        try:
            async with lock:
                # CrewAI agent là đồng bộ → chạy trong thread
                reply: str = await asyncio.to_thread(agent.ask, user_text)

        except Exception as e:
            logger.exception(f"GetInfoAgent error: {e}")
            reply = (
                "⚠️ Đã xảy ra lỗi khi xử lý yêu cầu của bạn.\n\n"
                "💡 Vui lòng kiểm tra cấu hình MCP server và biến môi trường (ví dụ: Google Sheets, GEMINI_API_KEY)."
            )

        await event_queue.enqueue_event(new_agent_text_message(reply))

    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # A2A side: chưa hỗ trợ cancel tác vụ đang chạy của CrewAI agent.
        await event_queue.enqueue_event(new_agent_text_message("⛔ Tính năng hủy tác vụ chưa được hỗ trợ."))
