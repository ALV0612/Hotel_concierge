# agents/booking_agent.py
# A2A-Compatible Booking Agent (async) — LangGraph + AsyncSqliteSaver + MCP STDIO (Per-Session DB + Auto-Purge)

import os, sys, json, asyncio, operator, logging, re, time, contextlib
from typing import Dict, List, Optional, TypedDict, Annotated, Type, Any
from datetime import datetime, timedelta
from enum import Enum
from contextlib import AsyncExitStack
from pathlib import Path

# LangGraph & LangChain
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, ConfigDict, validator

# Async SQLite
import aiosqlite

# MCP
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# ===================== Config =====================
TODAY = datetime.now().date()
TOMORROW = TODAY + timedelta(days=1)
MODEL = os.getenv("OHANA_MODEL", "gemini-2.5-flash")
SERVER_PATH = os.getenv("OHANA_MCP_SERVER", "./mcp_server/server_booking_mcp.py")
TOOL_CREATE = os.getenv("OHANA_CREATE_TOOL", "create_booking")  # tên tool MCP
CHECKPOINT_NS = os.getenv("OHANA_CHECKPOINT_NS", "ohana.booking.v1")

BOOKING_ALLOW_FAKE = os.getenv("BOOKING_ALLOW_FAKE") == "1"

# Per-session DB mode + purge policies
SESSION_DB_MODE       = os.getenv("OHANA_SESSION_DB", "1") == "1"          # bật per-session DB
SESSION_DB_EPHEMERAL  = os.getenv("OHANA_SESSION_EPHEMERAL", "0") == "1"  # :memory: mỗi session
PURGE_ON_FINISH       = os.getenv("OHANA_PURGE_ON_FINISH", "1") == "1"    # đặt xong -> xoá
PURGE_ON_CANCEL       = os.getenv("OHANA_PURGE_ON_CANCEL", "1") == "1"    # hủy -> xoá

# TTL optional (off by default if env not set)
OHANA_TTL_MIN         = int(os.getenv("OHANA_TTL_MINUTES", "0"))           # im lặng N phút -> xoá (0 = tắt)
OHANA_JANITOR_MIN     = int(os.getenv("OHANA_JANITOR_MINUTES", "10"))      # chu kỳ janitor quét (phút)

# DB path mặc định (chỉ dùng nếu SESSION_DB_MODE=0 hoặc làm base dir cho session files)
def _resolve_db_path() -> Path:
    if os.environ.get("OHANA_DB_PATH"):
        p = Path(os.environ["OHANA_DB_PATH"]).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    base = (
        os.environ.get("OHANA_DB_DIR")
        or os.environ.get("LOCALAPPDATA")
        or str(Path.home() / ".ohana" / "langgraph")
    )
    base = Path(base); base.mkdir(parents=True, exist_ok=True)
    return (base / "lg_memory.db").resolve()

def _sqlite_uri(p: Path) -> str:
    return "sqlite+aiosqlite:///" + p.as_posix()

DB_PATH = _resolve_db_path()

if not os.getenv("GOOGLE_API_KEY"):
    logger.warning("GOOGLE_API_KEY not found in environment")
    print("⚠️ GOOGLE_API_KEY not set - Gemini will not work!")
else:
    print(f"✅ GOOGLE_API_KEY found: {os.environ['GOOGLE_API_KEY'][:10]}...")

# ===================== Enhanced Context Parser =====================
# (Giữ nguyên toàn bộ parser như bản của bạn)
class ContextParser:
    """Parse rich context from Host Agent messages (text + embedded JSON)."""
    @staticmethod
    def _is_valid_vietnamese_name(name: str) -> bool:
        if not name or len(name.strip()) < 3:
            return False
        name = name.strip(); words = name.split()
        if len(words) < 2: return False
        if len(words) > 6: return False
        invalid_patterns = [r'\d', r'@', r'^\d{9,11}$', r'[^\w\sÀ-ỹ]{2,}',
                            r'\b(khách\s+sạn|hotel|booking|phòng|check|in|out|sđt|phone|email|standard|deluxe|suite|twin|double|single|giá|vnđ|vnd|đồng)\b',
                            r'\.(com|vn|org)', r'\b(oh\d+|oh\s+\d+)\b', r'\d{1,2}[/-]\d{1,2}[/-]\d{4}', r'\d{4}-\d{2}-\d{2}']
        for pattern in invalid_patterns:
            if re.search(pattern, name, re.IGNORECASE):
                return False
        vietnamese_name_chars = r'^[A-Za-zÀ-ỹ]+$'
        for word in words:
            if not re.match(vietnamese_name_chars, word):
                return False
        return all(word[0].isupper() for word in words)

    @staticmethod
    def _extract_json_blocks(message: str) -> List[dict]:
        blocks = []
        for m in re.finditer(r'[:：]\s*(\{[\s\S]*?\})', message):
            try: blocks.append(json.loads(m.group(1)))
            except Exception: pass
        for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", message, re.IGNORECASE):
            try: blocks.append(json.loads(m.group(1)))
            except Exception: pass
        if not blocks:
            m = re.search(r'(\{[\s\S]*\})', message)
            if m:
                try: blocks.append(json.loads(m.group(1)))
                except Exception: pass
        return blocks

    @staticmethod
    def _norm_date(s: str) -> str:
        if "/" in s:
            d, m, y = s.split("/")
            return f"{y}-{m.zfill(2)}-{d.zfill(2)}"
        return s

    @classmethod
    def extract_booking_info_from_context(cls, message: str) -> Dict:
        info: Dict = {}
        lines = message.split('\n'); current_message_part = ""
        for i, line in enumerate(lines):
            if re.search(r'\bOH\d{3}\b', line, re.IGNORECASE):
                if not any(keyword in line.lower() for keyword in ['lịch sử','context','thông tin đặt phòng']):
                    current_message_part = line
        if not current_message_part:
            current_message_part = message
        room_patterns = [r'\bOH(\d{3})\b', r'\b(\d{3})\s*(?:đi|nha|nhé|ạ)?\b', r'\b(\d{3})\b', r'phòng\s+(\d{3})\b', r'đặt\s+(\d{3})\b']
        for pattern in room_patterns:
            m = re.search(pattern, current_message_part, re.IGNORECASE)
            if m:
                room_num = m.group(1)
                info['room_id'] = f'OH{room_num}'
                break
        if 'room_id' not in info:
            for pattern in [r'phòng\s+(OH\d{3})', r'đặt\s+(OH\d{3})', r'(OH\d{3})']:
                m = re.search(pattern, message, re.IGNORECASE)
                if m:
                    info['room_id'] = m.group(1).upper()
                    break
        for pattern in [r'cho\s+(\d+)\s+người\b', r'(\d+)\s+người\b', r'cho\s+(\d+)\s+khách\b', r'(\d+)\s+khách\b']:
            m = re.search(pattern, message, re.IGNORECASE)
            if m:
                count = int(m.group(1))
                if 1 <= count <= 10: info['num_guests'] = count
                break
        for pat in [
            r'từ\s+(?:ngày\s+)?(\d{4}-\d{2}-\d{2})\s+(?:đến|->|→)\s+(?:ngày\s+)?(\d{4}-\d{2}-\d{2})',
            r'từ\s+(?:ngày\s+)?(\d{1,2}/\d{1,2}/\d{4})\s+(?:đến|->|→)\s+(?:ngày\s+)?(\d{1,2}/\d{1,2}/\d{4})',
            r'(\d{4}-\d{2}-\d{2})\s+(?:đến|->|→)\s+(\d{4}-\d{2}-\d{2})',
            r'(\d{1,2}/\d{1,2}/\d{4})\s+(?:đến|->|→)\s+(\d{1,2}/\d{1,2}/\d{4})',
        ]:
            m = re.search(pat, message, re.IGNORECASE)
            if m:
                info['check_in']  = cls._norm_date(m.group(1))
                info['check_out'] = cls._norm_date(m.group(2))
                break
        for label, key in [('check-in','check_in'),('check_out','check_out'),('check-out','check_out')]:
            m = re.search(rf'{label}[:\s]+(?:ngày\s+)?(\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}/\d{{1,2}}/\d{{4}})', message, re.IGNORECASE)
            if m: info[key] = cls._norm_date(m.group(1))
        name_patterns = [
            r'^([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)*)\,\s*0\d{9,10}$',
            r'^([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)*)\,\s+0\d{9,10}$',
            r'tên\s+(?:khách\s+)?(?:là\s+)?([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+){1,4})\b',
            r'tôi\s+là\s+([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+){1,4})\b',
            r'họ\s+tên[:\s]+([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+){1,4})\b',
            r'\b([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+){1,3})\s+0\d{9,10}\b',
        ]
        for pattern in name_patterns:
            m = re.search(pattern, message, re.IGNORECASE)
            if m:
                candidate_name = m.group(1).strip()
                candidate_name = re.sub(r'\b(sđt|SĐT|phone|số|điện|thoại|khách|sạn|hotel|phòng|room|check|in|out|booking|giá|vnđ|vnd)\b', '', candidate_name, flags=re.IGNORECASE).strip()
                candidate_name = re.sub(r'[,.:;]+', ' ', candidate_name).strip()
                candidate_name = re.sub(r'\s+', ' ', candidate_name)
                if cls._is_valid_vietnamese_name(candidate_name):
                    info['guest_name'] = candidate_name
                    break
        phone_patterns = [r'[A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)*,\s*(0\d{9,10})\b', r'\b(?:sđt|SĐT|phone|số\s+điện\s+thoại)[:\s]+(0\d{9,10})\b', r'\b(0\d{9,10})\b(?![.\d])']
        for pattern in phone_patterns:
            m = re.search(pattern, message, re.IGNORECASE)
            if m:
                phone = m.group(1)
                if re.match(r'^0\d{9,10}$', phone):
                    info['phone_number'] = phone
                    break
        for block in cls._extract_json_blocks(message):
            lrq = (block.get("last_room_query") or block.get("lastQuery") or {})
            if isinstance(lrq, dict):
                info.setdefault('num_guests', lrq.get('guests'))
                if lrq.get('check_in'):  info.setdefault('check_in',  lrq['check_in'])
                if lrq.get('check_out'): info.setdefault('check_out', lrq['check_out'])
            for key in ("booking","request","data","current"):
                b = block.get(key)
                if isinstance(b, dict):
                    if 'room_id' not in info:
                        info.setdefault('room_id', b.get('room_id') or b.get('room'))
                    info.setdefault('guest_name',   b.get('guest_name') or b.get('name'))
                    info.setdefault('phone_number', b.get('phone_number') or b.get('phone'))
                    if b.get('check_in'):  info.setdefault('check_in',  b['check_in'])
                    if b.get('check_out'): info.setdefault('check_out', b['check_out'])
                    if b.get('guests'):    info.setdefault('num_guests', b['guests'])
        return {k: v for k, v in info.items() if v is not None}

# ===================== State & Models =====================
class BookingIntent(str, Enum):
    GREETING = "greeting"
    BOOKING_REQUEST = "booking_request"
    PROVIDE_INFO = "provide_info"
    CONFIRM = "confirm"
    CANCEL = "cancel"
    UNCLEAR = "unclear"
    CONTEXT_RICH = "context_rich"

class BookingInfo(BaseModel):
    room_id: Optional[str] = None
    check_in: Optional[str] = None
    check_out: Optional[str] = None
    guest_name: Optional[str] = None
    phone_number: Optional[str] = None
    num_guests: Optional[int] = None
    @validator("check_in", "check_out")
    def _valid_date(cls, v):
        if v: datetime.strptime(v, "%Y-%m-%d")
        return v

class BookingState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], operator.add]
    user_input: str
    intent: Optional[BookingIntent]
    booking_data: BookingInfo
    conversation_stage: str
    last_response: str
    next_action: str
    confirmation_pending: bool
    error_count: int
    context_info: Optional[Dict]
    should_purge: bool

# ===================== MCP Tools =====================
class MCPTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    session: ClientSession
    def _fmt(self, result: types.CallToolResult) -> str:
        try:
            if getattr(result, "structured_content", None):
                return json.dumps(result.structured_content, ensure_ascii=False)
            text = "\n".join(getattr(c, "text", "") for c in (result.content or []))
            try:
                return json.dumps(json.loads(text), ensure_ascii=False)
            except Exception:
                return json.dumps({"result": text}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

class CreateBookingInput(BaseModel):
    room_id: str = Field(description="OH###")
    check_in: str
    check_out: str
    guest_name: str
    phone_number: Optional[str] = None

class CreateBookingTool(MCPTool):
    name: str = TOOL_CREATE
    description: str = "Create new booking when info is complete and confirmed"
    args_schema: Type[BaseModel] = CreateBookingInput
    def _run(self, *args, **kwargs) -> str:
        raise NotImplementedError("Use async _arun")
    async def _arun(self, **kwargs) -> str:
        return self._fmt(await self.session.call_tool(self.name, kwargs))

# ===================== Workflow =====================
class BookingWorkflow:
    def __init__(self, mcp_session: ClientSession, tools_cache_ref: Dict[str, set] | None = None):
        self.llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0.1, convert_system_message_to_human=True)
        self.extract_llm = self.llm.with_structured_output(BookingInfo)
        self.tools = [CreateBookingTool(session=mcp_session)]
        self.context_parser = ContextParser()
        self._tools_cache_ref = tools_cache_ref

    async def detect_intent(self, state: BookingState) -> BookingState:
        state.setdefault("messages", [])
        state.setdefault("booking_data", BookingInfo())
        state.setdefault("conversation_stage", "greeting")
        state.setdefault("confirmation_pending", False)
        state.setdefault("error_count", 0)
        msg = state.get("user_input", "") or ""
        if state["conversation_stage"] == "ready":
            state["conversation_stage"] = "greeting"; state["booking_data"] = BookingInfo()
        context_info = self.context_parser.extract_booking_info_from_context(msg)
        state["context_info"] = context_info
        if state.get("confirmation_pending"):
            t = msg.strip().lower()
            confirm_words = ["có","co","ok","okay","đồng ý","yes","y","đúng","dung","xác nhận","được","ừ","uhm"]
            cancel_words = ["không","khong","ko","no","cancel","hủy","huy","từ chối","thôi"]
            intent = BookingIntent.CANCEL if any(w in t for w in cancel_words) else (BookingIntent.CONFIRM if any(w in t for w in confirm_words) else BookingIntent.UNCLEAR)
        elif "context" in msg.lower() or len(context_info) >= 3:
            intent = BookingIntent.CONTEXT_RICH
        else:
            intent = self._rule_based_intent(msg)
            if intent == BookingIntent.UNCLEAR:
                try:
                    prompt = ("Phân loại intent (greeting, booking_request, provide_info, confirm, cancel, unclear).\n"
                              f"Tin nhắn: \"{msg}\"\nChỉ trả về đúng một nhãn.")
                    intent_str = (await self.llm.ainvoke([HumanMessage(content=prompt)])).content.strip().lower()
                    intent = BookingIntent(intent_str) if intent_str in BookingIntent._value2member_map_ else self._fallback_intent(msg)
                except Exception as e:
                    intent = self._fallback_intent(msg)
        state["intent"] = intent
        state["messages"].append(HumanMessage(content=msg))
        state["next_action"] = self._route_intent(intent, state)
        return state

    def _rule_based_intent(self, t: str) -> BookingIntent:
        s = t.lower().strip()
        if any(w in s for w in ["hi","hello","xin chào","chào","helo","alo"]): return BookingIntent.GREETING
        if any(w in s for w in ["đặt","book","booking","muốn đặt","tôi muốn"]) or any(w in s for w in ["phòng","room","oh"]): return BookingIntent.BOOKING_REQUEST
        if any(w in s for w in ["có","yes","ok","xác nhận","đồng ý","được","ừ"]) and len(s) < 20: return BookingIntent.CONFIRM
        if any(w in s for w in ["không","no","hủy","cancel","thôi","từ chối"]): return BookingIntent.CANCEL
        if any(ch.isdigit() for ch in s): return BookingIntent.PROVIDE_INFO
        if any(w in s for w in ["tên","tôi là","tên tôi","họ tên","ngày","tháng","check-in","check-out","hôm nay","ngày mai"]): return BookingIntent.PROVIDE_INFO
        return BookingIntent.UNCLEAR

    def _fallback_intent(self, t: str) -> BookingIntent:
        s = t.lower()
        if any(k in s for k in ["hi","hello","xin chào","chào"]): return BookingIntent.GREETING
        if any(k in s for k in ["đặt","book","booking","muốn đặt"]): return BookingIntent.BOOKING_REQUEST
        if any(k in s for k in ["có","yes","ok","xác nhận","đồng ý"]): return BookingIntent.CONFIRM
        if any(k in s for k in ["không","no","hủy","cancel"]): return BookingIntent.CANCEL
        return BookingIntent.PROVIDE_INFO

    def _route_intent(self, intent: BookingIntent, state: BookingState) -> str:
        stage = state.get("conversation_stage", "greeting")
        if intent == BookingIntent.GREETING: return "handle_greeting"
        if intent == BookingIntent.CONTEXT_RICH: return "handle_context_rich"
        if intent == BookingIntent.BOOKING_REQUEST: return "handle_booking_request"
        if intent == BookingIntent.CONFIRM and state.get("confirmation_pending"): return "execute_booking"
        if intent == BookingIntent.CANCEL: return "handle_cancel"
        if intent == BookingIntent.PROVIDE_INFO and stage == "collecting": return "collect_info"
        return "collect_info" if stage == "collecting" else "handle_unclear"

    async def handle_context_rich(self, state: BookingState) -> BookingState:
        state.setdefault("messages", [])
        current = state.get("booking_data", BookingInfo())
        merged = self._merge_context_info(current, state.get("context_info", {}) or {})
        state["booking_data"] = merged
        missing = self._get_missing_fields(merged)
        if not missing:
            resp = self._format_confirmation_summary(merged)
            state.update(conversation_stage="confirming", confirmation_pending=True, next_action="wait_input")
        else:
            resp = self._ask_for_missing_fields(missing, merged)
            state.update(conversation_stage="collecting", next_action="wait_input")
        state["messages"].append(AIMessage(content=resp))
        state["last_response"] = resp
        return state

    async def handle_booking_request(self, state: BookingState) -> BookingState:
        state.setdefault("messages", [])
        msg = state.get("user_input", "")
        ctx = state.get("context_info") or {}
        if ctx:
            merged = self._merge_context_info(state.get("booking_data", BookingInfo()), ctx)
            state["booking_data"] = merged
        else:
            prompt = (f"Extract booking info from Vietnamese message.\nToday: {TODAY}\nMessage: \"{msg}\"\n"
                      f"Rules:\n- \"hôm nay\"={TODAY}, \"ngày mai\"={TOMORROW}\n"
                      f"- Room format: OH + 3 digits (e.g., OH203)\n- Return None for missing fields.")
            try:
                info = await self.extract_llm.ainvoke([HumanMessage(content=prompt)])
                state["booking_data"] = self._merge_booking_info(state.get("booking_data"), info)
            except Exception as e:
                logger.error(f"extract error: {e}")
        missing = self._get_missing_fields(state["booking_data"])
        if not missing:
            resp = self._format_confirmation_summary(state["booking_data"])
            state.update(conversation_stage="confirming", confirmation_pending=True, next_action="wait_input")
        else:
            resp = self._ask_for_missing_fields(missing, state["booking_data"])
            state.update(conversation_stage="collecting", next_action="wait_input")
        state["messages"].append(AIMessage(content=resp)); state["last_response"] = resp
        return state

    async def collect_info(self, state: BookingState) -> BookingState:
        state.setdefault("messages", [])
        msg = state.get("user_input", "")
        ctx = state.get("context_info") or {}
        if ctx:
            merged = self._merge_context_info(state.get("booking_data", BookingInfo()), ctx)
            state["booking_data"] = merged
        else:
            prompt = (f"Update booking info with new information.\nCurrent: {state.get('booking_data').dict() if state.get('booking_data') else {}}\n"
                      f"New message: \"{msg}\"\nToday: {TODAY}\nExtract any booking information.")
            try:
                info = await self.extract_llm.ainvoke([HumanMessage(content=prompt)])
                state["booking_data"] = self._merge_booking_info(state.get("booking_data"), info)
            except Exception as e:
                logger.error(f"collect error: {e}")
                resp = "Xin lỗi, tôi chưa hiểu rõ. Bạn mô tả lại giúp nhé?"
                state["error_count"] = state.get("error_count", 0) + 1
                state["messages"].append(AIMessage(content=resp))
                state["last_response"] = resp
                state["next_action"] = "wait_input"
                return state
        missing = self._get_missing_fields(state["booking_data"])
        if not missing:
            resp = self._format_confirmation_summary(state["booking_data"])
            state.update(conversation_stage="confirming", confirmation_pending=True)
        else:
            resp = self._ask_for_missing_fields(missing, state["booking_data"])
        state["messages"].append(AIMessage(content=resp)); state["last_response"] = resp; state["next_action"] = "wait_input"
        return state

    async def execute_booking(self, state: BookingState) -> BookingState:
        state.setdefault("messages", [])
        d: BookingInfo = state["booking_data"]
        try:
            if not self._validate_booking_data(d):
                resp = "❌ Thông tin chưa hợp lệ. Vui lòng kiểm tra phòng/ngày/tên."
                state["next_action"] = "collect_info"
                state["messages"].append(AIMessage(content=resp)); state["last_response"] = resp
                return state
            payload = {k: getattr(d, k) for k in ["room_id","check_in","check_out","guest_name","phone_number"] if getattr(d, k) is not None}
            tool_names = set()
            try:
                tools_list = await self.tools[0].session.list_tools()
                tool_names = {t.name for t in (tools_list.tools or [])}
            except Exception as e:
                logger.warning(f"list_tools failed before call: {e}")
            if TOOL_CREATE not in tool_names:
                if BOOKING_ALLOW_FAKE:
                    bid = f"DEMO-{d.room_id}-{d.check_in.replace('-','')}"
                    resp = ( "✅ (DEMO) ĐẶT PHÒNG THÀNH CÔNG!\n\n"
                             f"📋 Mã booking: {bid}\n🏨 Phòng: {d.room_id}\n👤 Khách: {d.guest_name}\n"
                             f"📅 Check-in: {d.check_in}\n📅 Check-out: {d.check_out}\n📱 SĐT: {d.phone_number or 'Không có'}\n")
                    state.update(conversation_stage="ready", confirmation_pending=False, booking_data=BookingInfo(), next_action="wait_input")
                    state["messages"].append(AIMessage(content=resp)); state["last_response"]=resp
                    return state
                else:
                    resp = ("⚠️ Hệ thống chưa sẵn sàng để tạo booking (thiếu tool `create_booking`).\n"
                            f"Đã ghi nhận: Phòng {d.room_id}, {d.check_in}→{d.check_out}, Khách {d.guest_name or 'Chưa có'}, ĐT {d.phone_number or 'Chưa có'}.")
                    state["next_action"] = "wait_input"
                    state["messages"].append(AIMessage(content=resp)); state["last_response"]=resp
                    return state
            async def _call_create():
                return json.loads(await self.tools[0]._arun(**payload))
            try:
                result = await _call_create()
            except (BrokenPipeError, ConnectionResetError, EOFError, RuntimeError) as e:
                logger.warning(f"tool call error '{e}', retrying once...")
                await asyncio.sleep(0.2)
                result = await _call_create()
            if result.get("success"):
                bid = result.get("booking_id", "N/A")
                resp = (
                    "✅ ĐẶT PHÒNG THÀNH CÔNG!\n\n"
                    f"📋 Mã booking: {bid}\n"
                    f"🏨 Phòng: {d.room_id}\n"
                    f"👤 Khách: {d.guest_name}\n"
                    f"📅 Check-in: {d.check_in}\n"
                    f"📅 Check-out: {d.check_out}\n"
                    f"📱 SĐT: {d.phone_number or 'Không có'}\n"
                )
                state.update(conversation_stage="ready", confirmation_pending=False, booking_data=BookingInfo(), next_action="wait_input")
            else:
                msg = result.get("message") or result.get("error") or "Lỗi không xác định"
                resp = f"❌ Không thể đặt phòng: {msg}"
                state["next_action"] = "wait_input"
        except Exception as e:
            logger.error(f"execute error: {e}")
            resp = f"❌ Lỗi hệ thống: {e}"
            state["next_action"] = "wait_input"
        state["messages"].append(AIMessage(content=resp)); state["last_response"] = resp
        return state

    def handle_greeting(self, state: BookingState) -> BookingState:
        state.setdefault("messages", [])
        resp = ("🏨 Xin chào! Tôi là trợ lý đặt phòng Ohana Hotel.\n\n"
                "Bạn chỉ cần cho biết: • Phòng muốn đặt (VD: OH203) • Ngày check-in/check-out • Tên và số điện thoại\n"
                "Bạn muốn đặt phòng nào ạ?")
        state.update(conversation_stage="greeting", next_action="wait_input")
        state["messages"].append(AIMessage(content=resp)); state["last_response"] = resp
        return state

    def handle_cancel(self, state: BookingState) -> BookingState:
        state.setdefault("messages", [])
        # đặt cờ purge và kết thúc ngay vòng này
        state.update(booking_data=BookingInfo(), confirmation_pending=False, conversation_stage="greeting", next_action="reset_state")
        state["should_purge"] = True
        resp = "Đã hủy quá trình đặt phòng. Bạn có thể bắt đầu lại bất cứ lúc nào."
        state["messages"].append(AIMessage(content=resp)); state["last_response"] = resp
        return state

    def handle_unclear(self, state: BookingState) -> BookingState:
        state.setdefault("messages", [])
        stage = state.get("conversation_stage", "greeting")
        if stage == "confirming" and state.get("confirmation_pending"):
            resp = "Bạn muốn xác nhận đặt phòng không? (Có/Không)"
        elif stage == "collecting":
            missing = self._get_missing_fields(state.get("booking_data"))
            resp = self._ask_for_missing_fields(missing, state.get("booking_data")) if missing else self._format_confirmation_summary(state["booking_data"])
        else:
            resp = "Tôi có thể giúp bạn đặt phòng. Bạn muốn đặt phòng nào, ngày nào ạ?"
        state["messages"].append(AIMessage(content=resp)); state["last_response"] = resp; state["next_action"] = "wait_input"
        return state

    # ---------- Helpers ----------
    def _merge_context_info(self, current: BookingInfo, ctx: Dict) -> BookingInfo:
        data = current.dict()
        for k in ["room_id","check_in","check_out","guest_name","phone_number","num_guests"]:
            if ctx.get(k): data[k] = ctx[k]
        return BookingInfo(**data)

    def _merge_booking_info(self, old: Optional[BookingInfo], new: BookingInfo) -> BookingInfo:
        if not old: return new
        data = old.dict()
        for k, v in new.dict().items():
            if v is not None: data[k] = v
        return BookingInfo(**data)

    def _get_missing_fields(self, d: Optional[BookingInfo]) -> List[str]:
        if not d: return ["room_id","check_in","check_out","guest_name"]
        need = ["room_id","check_in","check_out","guest_name","phone_number"]
        return [k for k in need if not getattr(d, k, None)]

    def _ask_for_missing_fields(self, missing: List[str], d: Optional[BookingInfo]) -> str:
        labels = {"room_id":"số phòng","check_in":"ngày check-in","check_out":"ngày check-out","guest_name":"tên khách","phone_number":"số điện thoại"}
        have = []
        if d:
            if d.room_id: have.append(f"Phòng {d.room_id}")
            if d.guest_name: have.append(f"Khách {d.guest_name}")
            if d.check_in: have.append(f"Check-in {d.check_in}")
            if d.check_out: have.append(f"Check-out {d.check_out}")
        prefix = f"Tôi đã có: {', '.join(have)}\n\n" if have else ""
        return prefix + "Vui lòng cho biết: " + ", ".join(labels.get(f, f) for f in missing)

    def _format_confirmation_summary(self, d: BookingInfo) -> str:
        return ("📋 XÁC NHẬN THÔNG TIN ĐẶT PHÒNG:\n\n"
                f"🏨 Phòng: {d.room_id}\n"
                f"📅 Check-in: {d.check_in}\n"
                f"📅 Check-out: {d.check_out}\n"
                f"👤 Tên khách: {d.guest_name}\n"
                f"📱 Điện thoại: {d.phone_number or 'Chưa có'}\n\n"
                "✅ Xác nhận đặt phòng? (Gõ 'Có' hoặc 'Không')")

    def _validate_booking_data(self, d: BookingInfo) -> bool:
        try:
            if not all([d.room_id, d.check_in, d.check_out, d.guest_name, d.phone_number]): return False
            ci = datetime.strptime(d.check_in, "%Y-%m-%d").date()
            co = datetime.strptime(d.check_out, "%Y-%m-%d").date()
            if not (TODAY <= ci < co): return False
            phone_digits = "".join(ch for ch in (d.phone_number or "") if ch.isdigit())
            return 9 <= len(phone_digits) <= 11
        except Exception:
            return False

# ===================== A2A-Compatible Agent =====================
class BookingAgent:
    """
    - Per-session DB isolation: mỗi session có checkpointer/graph riêng (file .db riêng hoặc :memory:)
    - Auto-purge: khi đặt xong / hủy (và TTL tùy chọn) sẽ đóng + xoá file DB của session
    - MCP stdio server giữ 1 process sống dài (không respawn mỗi request)
    """
    def __init__(self):
        # Graph/checkpointer chung (chỉ dùng nếu SESSION_DB_MODE=0)
        self._graph = None
        self._checkpointer: Optional[AsyncSqliteSaver] = None
        self._db = None
        self._db_path = DB_PATH

        # Per-session containers
        self._sessions: Dict[str, Dict[str, Any]] = {}

        # Lifecycles
        self._stack: AsyncExitStack | None = None
        self._initialized = False

        # MCP health/cache
        self._mcp_session: ClientSession | None = None
        self._last_tools_check = 0.0
        self._tools_cache: set[str] = set()

        # TTL/Janitor
        self._last_seen: dict[str, float] = {}
        self._ttl_seconds = max(0, OHANA_TTL_MIN) * 60
        self._janitor_interval = max(1, OHANA_JANITOR_MIN) * 60
        self._janitor_task: asyncio.Task | None = None

    # --------- Per-session helpers ---------
    def _sanitize_session_id(self, session_id: str) -> str:
        s = re.sub(r"[^A-Za-z0-9._-]+", "-", session_id.strip())
        return s or "default"

    def _session_db_path(self, session_id: str) -> Path:
        base = Path(os.environ.get("OHANA_DB_DIR") or self._db_path.parent).resolve()
        (base / "sessions").mkdir(parents=True, exist_ok=True)
        return (base / "sessions" / f"{self._sanitize_session_id(session_id)}.db").resolve()

    async def _new_session_checkpointer(self, session_id: str) -> tuple[AsyncExitStack, AsyncSqliteSaver, Optional[aiosqlite.Connection], Optional[Path]]:
        # mỗi session có stack riêng, lồng trong global stack để cleanup đồng bộ
        stack = AsyncExitStack()
        await self._stack.enter_async_context(stack)
        if SESSION_DB_EPHEMERAL:
            cp = await stack.enter_async_context(AsyncSqliteSaver.from_conn_string(":memory:"))
            return stack, cp, None, None
        db_path = self._session_db_path(session_id)
        uri = "sqlite+aiosqlite:///" + db_path.as_posix()
        try:
            cp = await stack.enter_async_context(AsyncSqliteSaver.from_conn_string(uri))
            return stack, cp, None, db_path
        except Exception as e:
            logger.warning(f"[{session_id}] from_conn_string failed ({e}); fallback to direct aiosqlite")
            db = await aiosqlite.connect(db_path.as_posix())
            await db.execute("PRAGMA journal_mode=WAL;")
            await db.execute("PRAGMA busy_timeout=5000;")
            await db.commit()
            cp = AsyncSqliteSaver(db)
            return stack, cp, db, db_path

    def _build_graph(self, checkpointer: AsyncSqliteSaver) -> Any:
        wf = BookingWorkflow(self._mcp_session, tools_cache_ref={"tools": self._tools_cache} if self._tools_cache else None)
        g = StateGraph(BookingState)
        g.add_node("detect_intent", wf.detect_intent)
        g.add_node("handle_greeting", wf.handle_greeting)
        g.add_node("handle_context_rich", wf.handle_context_rich)
        g.add_node("handle_booking_request", wf.handle_booking_request)
        g.add_node("collect_info", wf.collect_info)
        g.add_node("execute_booking", wf.execute_booking)
        g.add_node("handle_cancel", wf.handle_cancel)
        g.add_node("handle_unclear", wf.handle_unclear)
        g.set_entry_point("detect_intent")
        def after_intent(st: BookingState):
            return st["next_action"]
        def after_action(st: BookingState):
            nxt = st.get("next_action", "wait_input")
            return END if nxt in {"wait_input","complete","reset_state"} else nxt
        g.add_conditional_edges(
            "detect_intent",
            after_intent,
            {
                "handle_greeting": "handle_greeting",
                "handle_context_rich": "handle_context_rich",
                "handle_booking_request": "handle_booking_request",
                "collect_info": "collect_info",
                "execute_booking": "execute_booking",
                "handle_cancel": "handle_cancel",
                "handle_unclear": "handle_unclear",
            },
        )
        for n in ["handle_greeting","handle_context_rich","handle_booking_request","collect_info","handle_cancel","handle_unclear"]:
            g.add_conditional_edges(n, after_action)
        return g.compile(checkpointer=checkpointer)

    async def _ensure_session_graph(self, session_id: str):
        if not SESSION_DB_MODE:
            return
        if session_id in self._sessions:
            return
        stack, cp, db, db_path = await self._new_session_checkpointer(session_id)
        graph = self._build_graph(cp)
        self._sessions[session_id] = {"stack": stack, "checkpointer": cp, "db": db, "db_path": db_path, "graph": graph}
        logger.info(f"[{session_id}] session graph initialized at {db_path or ':memory:'}")

    async def _close_session_resources(self, session_id: str):
        sess = self._sessions.pop(session_id, None)
        if not sess:
            return
        with contextlib.suppress(Exception):
            await sess["stack"].aclose()
        db_path = sess.get("db_path")
        if db_path and db_path.exists():
            with contextlib.suppress(Exception):
                os.remove(db_path)

    async def _purge_session(self, session_id: str):
        await self._close_session_resources(session_id)
        self._last_seen.pop(session_id, None)
        logger.info(f"[{session_id}] purged session DB")

    async def _janitor_loop(self):
        try:
            while self._initialized:
                await asyncio.sleep(self._janitor_interval)
                if self._ttl_seconds <= 0:
                    continue
                now = time.time()
                for sid, last in list(self._last_seen.items()):
                    if (now - last) > self._ttl_seconds:
                        logger.info(f"[JANITOR] TTL exceeded for '{sid}', purging session")
                        await self._purge_session(sid)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"Janitor loop error: {e}")

    # --------- MCP health ---------
    async def _ensure_mcp_alive(self, force: bool = False):
        if (not self._mcp_session) or (not self._initialized):
            await self.initialize(); return
        if not force and time.time() - self._last_tools_check < 5:
            return
        try:
            tools_list = await self._mcp_session.list_tools()
            self._tools_cache = {t.name for t in (tools_list.tools or [])}
            self._last_tools_check = time.time()
        except Exception as e:
            logger.warning(f"MCP unhealthy ({e}); reinitializing...")
            await self.aclose()
            await self.initialize()

    async def initialize(self, server_path: str | None = None):
        if self._initialized:
            return
        try:
            mcp_path = Path(server_path or SERVER_PATH).resolve()
            if not mcp_path.exists():
                raise FileNotFoundError(f"MCP server not found: {mcp_path}")
            project_root = mcp_path.parents[1] if len(mcp_path.parents) >= 2 else mcp_path.parent
            env = os.environ.copy(); env["PYTHONUTF8"] = "1"
            env["PYTHONPATH"] = os.pathsep.join([str(project_root), env.get("PYTHONPATH", "")])
            self._stack = AsyncExitStack()
            read, write = await self._stack.enter_async_context(
                stdio_client(
                    StdioServerParameters(
                        command=sys.executable,
                        args=["-u", "-X", "utf8", str(mcp_path)],
                        cwd=str(project_root),
                        env=env,
                    )
                )
            )
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._mcp_session = session
            try:
                tools_list = await session.list_tools()
                self._tools_cache = {t.name for t in (tools_list.tools or [])}
                logger.info(f"MCP tools available: {sorted(self._tools_cache)}")
            except Exception as e:
                logger.warning(f"list_tools failed during init: {e}")

            # If per-session DB mode: we skip building global graph/checkpointer here
            if not SESSION_DB_MODE:
                # Fallback single DB (legacy mode)
                Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
                uri = _sqlite_uri(self._db_path)
                try:
                    self._checkpointer = await self._stack.enter_async_context(AsyncSqliteSaver.from_conn_string(uri))
                    logger.info(f"AsyncSqliteSaver via URI: {uri}")
                except Exception as e:
                    logger.warning(f"from_conn_string failed ({e}); fallback to aiosqlite.connect()")
                    self._db = await aiosqlite.connect(self._db_path.as_posix())
                    await self._db.execute("PRAGMA journal_mode=WAL;")
                    await self._db.execute("PRAGMA busy_timeout=5000;")
                    await self._db.commit()
                    self._checkpointer = AsyncSqliteSaver(self._db)
                    logger.info(f"AsyncSqliteSaver (direct aiosqlite) at {self._db_path}")
                # Build global graph
                wf = BookingWorkflow(session, tools_cache_ref={"tools": self._tools_cache} if self._tools_cache else None)
                g = StateGraph(BookingState)
                g.add_node("detect_intent", wf.detect_intent)
                g.add_node("handle_greeting", wf.handle_greeting)
                g.add_node("handle_context_rich", wf.handle_context_rich)
                g.add_node("handle_booking_request", wf.handle_booking_request)
                g.add_node("collect_info", wf.collect_info)
                g.add_node("execute_booking", wf.execute_booking)
                g.add_node("handle_cancel", wf.handle_cancel)
                g.add_node("handle_unclear", wf.handle_unclear)
                g.set_entry_point("detect_intent")
                def after_intent(st: BookingState):
                    return st["next_action"]
                def after_action(st: BookingState):
                    nxt = st.get("next_action", "wait_input")
                    return END if nxt in {"wait_input","complete","reset_state"} else nxt
                g.add_conditional_edges(
                    "detect_intent",
                    after_intent,
                    {
                        "handle_greeting": "handle_greeting",
                        "handle_context_rich": "handle_context_rich",
                        "handle_booking_request": "handle_booking_request",
                        "collect_info": "collect_info",
                        "execute_booking": "execute_booking",
                        "handle_cancel": "handle_cancel",
                        "handle_unclear": "handle_unclear",
                    },
                )
                for n in ["handle_greeting","handle_context_rich","handle_booking_request","collect_info","handle_cancel","handle_unclear"]:
                    g.add_conditional_edges(n, after_action)
                self._graph = g.compile(checkpointer=self._checkpointer)
                logger.info(f"BookingAgent initialized (legacy single DB). DB at: {self._db_path}")
            else:
                self._graph = None
                logger.info("BookingAgent initialized (per-session DB mode). Graphs will be created per session.")

            self._initialized = True

            # start janitor if TTL > 0
            if self._ttl_seconds > 0 and self._janitor_task is None:
                self._janitor_task = asyncio.create_task(self._janitor_loop())

        except Exception as e:
            logger.error(f"Failed to initialize BookingAgent: {e}")
            await self.aclose()
            raise

    async def aclose(self):
        self._initialized = False
        # Close all session resources
        for sid in list(self._sessions.keys()):
            with contextlib.suppress(Exception):
                await self._close_session_resources(sid)
        self._sessions.clear()
        # Stop janitor
        if self._janitor_task:
            try:
                self._janitor_task.cancel()
                with contextlib.suppress(Exception):
                    await self._janitor_task
            finally:
                self._janitor_task = None
        # Close global stack & db
        if self._stack:
            with contextlib.suppress(Exception):
                await self._stack.aclose()
            self._stack = None
        if self._db:
            with contextlib.suppress(Exception):
                await self._db.close()
            self._db = None
        self._graph = None
        self._mcp_session = None
        self._tools_cache = set()
        self._last_tools_check = 0.0

    # -------------------- Public APIs --------------------
    async def chat(self, message: str, session_id: str = "default") -> str:
        """Gọi từ client/host: trả về 1 response và giữ memory theo session_id."""
        if not self._initialized:
            await self.initialize()
        if not self._mcp_session:
            return "Lỗi: MCP chưa sẵn sàng."
        await self._ensure_mcp_alive()

        # TTL purge trước nếu có
        if session_id in self._last_seen and self._ttl_seconds > 0 and time.time() - self._last_seen[session_id] > self._ttl_seconds:
            await self._purge_session(session_id)

        # Graph theo session
        if SESSION_DB_MODE:
            await self._ensure_session_graph(session_id)
            graph = self._sessions[session_id]["graph"]
        else:
            graph = self._graph
            if not graph:
                return "Lỗi: Agent chưa được khởi tạo đúng cách."

        try:
            cfg = {"configurable": {"thread_id": session_id, "checkpoint_ns": CHECKPOINT_NS}}
            result_state: BookingState = await graph.ainvoke({"user_input": message}, config=cfg)

            # cập nhật last_seen
            self._last_seen[session_id] = time.time()

            # quyết định purge
            should_purge = False
            if result_state.get("should_purge"):
                should_purge = True
            if PURGE_ON_FINISH and result_state.get("conversation_stage") == "ready" and not result_state.get("confirmation_pending", False):
                should_purge = True
            if PURGE_ON_CANCEL and str(result_state.get("intent") or "") in {"cancel"}:
                should_purge = True
            if should_purge and SESSION_DB_MODE:
                await self._purge_session(session_id)

            return result_state.get("last_response", "Xin lỗi, có lỗi xảy ra.")
        except Exception as e:
            logger.exception(f"Chat error (session {session_id}): {e}")
            return f"Đã có lỗi khi xử lý: {str(e)}"

    # ---------- A2A wrapper ----------
    async def process_a2a(self, payload: Dict) -> Dict:
        """
        Request shape:
        {
          "skill": "ohana.booking",
          "input": { "text": "..." },
          "options": { "run_to_completion": false },
          "session": { "id": "ohana-permanent-session" }
        }
        """
        if not self._initialized:
            await self.initialize()
        await self._ensure_mcp_alive()

        session_id = ((payload.get("session") or {}).get("id")) or "default"
        text = ((payload.get("input") or {}).get("text")) or ""

        # TTL purge trước
        if session_id in self._last_seen and self._ttl_seconds > 0 and time.time() - self._last_seen[session_id] > self._ttl_seconds:
            await self._purge_session(session_id)

        if SESSION_DB_MODE:
            await self._ensure_session_graph(session_id)
            graph = self._sessions[session_id]["graph"]
        else:
            graph = self._graph
            if not graph:
                return {"messages": [{"role": "assistant", "content": "Lỗi: Agent chưa được khởi tạo đúng cách."}]}

        cfg = {"configurable": {"thread_id": session_id, "checkpoint_ns": CHECKPOINT_NS}}
        result_state: BookingState = await graph.ainvoke({"user_input": text}, config=cfg)

        self._last_seen[session_id] = time.time()

        should_purge = False
        if result_state.get("should_purge"):
            should_purge = True
        if PURGE_ON_FINISH and result_state.get("conversation_stage") == "ready" and not result_state.get("confirmation_pending", False):
            should_purge = True
        if PURGE_ON_CANCEL and str(result_state.get("intent") or "") in {"cancel"}:
            should_purge = True
        if should_purge and SESSION_DB_MODE:
            await self._purge_session(session_id)

        reply = result_state.get("last_response", "")
        state_out = {
            "conversation_stage": result_state.get("conversation_stage"),
            "booking_data": (result_state.get("booking_data") or BookingInfo()).dict(),
            "confirmation_pending": result_state.get("confirmation_pending", False),
            "intent": (result_state.get("intent") or "") and str(result_state.get("intent").value),
            "last_response": reply,
        }

        return {"messages": [{"role": "assistant", "content": reply}], "state": state_out, "session": {"id": session_id}}

# ---------- Local test ----------
async def test_agent():
    agent = BookingAgent()
    queries = [
        "Xin chào",
        "Context: Khách đã hỏi về phòng cho 4 người từ 2025-09-13 đến 2025-09-14. Bây giờ khách muốn đặt phòng OH404.",
        "tên khách Hoàng An, số điện thoại 0901234567",
        "Có, xác nhận đặt phòng",
    ]
    sid = "test_session"
    for q in queries:
        print(f"👤 User: {q}")
        ans = await agent.chat(q, session_id=sid)
        print(f"🤖 Bot: {ans}\n")
    await agent.aclose()

if __name__ == "__main__":
    asyncio.run(test_agent())
