# agents/booking_agent_optimized.py
# Optimized Booking Agent - Reduced LLM calls, faster processing

import os, sys, json, asyncio, operator, logging, re, time, contextlib
from typing import Dict, List, Optional, TypedDict, Annotated, Type, Any
from datetime import datetime, timedelta
from enum import Enum
from contextlib import AsyncExitStack
from pathlib import Path
from functools import lru_cache

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
load_dotenv(".env")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# ===================== Optimized Config =====================
TODAY = datetime.now().date()
TOMORROW = TODAY + timedelta(days=1)
MODEL = os.getenv("OHANA_MODEL", "gemini-2.5-flash")  # Fast model
SERVER_PATH = os.getenv("OHANA_BOOKING_MCP_SERVER", "./mcp_server/server_booking_mcp.py")
TOOL_CREATE = os.getenv("OHANA_CREATE_TOOL", "create_booking")
CHECKPOINT_NS = os.getenv("OHANA_CHECKPOINT_NS", "ohana.booking.v1")

BOOKING_ALLOW_FAKE = os.getenv("BOOKING_ALLOW_FAKE", "0") == "1"  # Default to fake for speed

# Simplified DB mode - use single DB by default for better performance
SESSION_DB_MODE = os.getenv("OHANA_SESSION_DB", "0") == "1"  # Default OFF
SESSION_DB_EPHEMERAL = os.getenv("OHANA_SESSION_EPHEMERAL", "1") == "1"  # Default memory
PURGE_ON_FINISH = os.getenv("OHANA_PURGE_ON_FINISH", "0") == "1"  # Default OFF
PURGE_ON_CANCEL = os.getenv("OHANA_PURGE_ON_CANCEL", "0") == "1"  # Default OFF

# Longer intervals for better performance
OHANA_TTL_MIN = int(os.getenv("OHANA_TTL_MINUTES", "0"))  # TTL OFF
OHANA_JANITOR_MIN = int(os.getenv("OHANA_JANITOR_MINUTES", "30"))  # Longer interval

def _resolve_db_path() -> Path:
    # FORCE D drive path - override everything
    base_dir = "D:/test/ohana_data"
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    return (base / "booking.db").resolve()

DB_PATH = _resolve_db_path()

# ===================== Optimized Context Parser =====================
class OptimizedContextParser:
    """Fast context parser with compiled regex and minimal processing."""
    
    # Pre-compiled regex patterns for better performance
    ROOM_PATTERNS = [
        re.compile(r'\bOH(\d{3})\b', re.IGNORECASE),
        re.compile(r'\b(\d{3})\s*(?:đi|nha|nhé|ạ)?\b'),
        re.compile(r'phòng\s+(\d{3})\b', re.IGNORECASE),
    ]
    
    GUEST_PATTERNS = [
        re.compile(r'([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)*)\,\s*0\d{9,10}$'),
        re.compile(r'tên\s+(?:khách\s+)?(?:là\s+)?([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+){1,4})\b', re.IGNORECASE),
        re.compile(r'tôi\s+là\s+([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+){1,4})\b', re.IGNORECASE),
    ]
    
    PHONE_PATTERNS = [
        re.compile(r'[A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)*,\s*(0\d{9,10})\b'),
        re.compile(r'\b(?:sđt|SĐT|phone|số\s+điện\s+thoại)[:\s]+(0\d{9,10})\b', re.IGNORECASE),
        re.compile(r'\b(0\d{9,10})\b(?![.\d])'),
    ]
    
    DATE_PATTERNS = [
        re.compile(r'từ\s+(?:ngày\s+)?(\d{4}-\d{2}-\d{2})\s+(?:đến|->|→)\s+(?:ngày\s+)?(\d{4}-\d{2}-\d{2})', re.IGNORECASE),
        re.compile(r'từ\s+(?:ngày\s+)?(\d{1,2}/\d{1,2}/\d{4})\s+(?:đến|->|→)\s+(?:ngày\s+)?(\d{1,2}/\d{1,2}/\d{4})', re.IGNORECASE),
    ]
    
    GUEST_COUNT_PATTERNS = [
        re.compile(r'cho\s+(\d+)\s+người\b', re.IGNORECASE),
        re.compile(r'(\d+)\s+người\b', re.IGNORECASE),
        re.compile(r'(\d+)\s+khách\b', re.IGNORECASE),
    ]

    @staticmethod
    @lru_cache(maxsize=200)
    def _is_valid_vietnamese_name_cached(name: str) -> bool:
        """Cached validation for Vietnamese names."""
        if not name or len(name.strip()) < 3:
            return False
        name = name.strip()
        words = name.split()
        if len(words) < 2 or len(words) > 6:
            return False
        
        # Fast checks for invalid patterns
        if any(char.isdigit() for char in name):
            return False
        if '@' in name or '.com' in name.lower():
            return False
        
        # Check for common invalid keywords
        invalid_words = {'khách', 'sạn', 'hotel', 'booking', 'phòng', 'check', 'sđt', 'phone', 'email'}
        if any(word.lower() in invalid_words for word in words):
            return False
            
        return all(word[0].isupper() for word in words)

    @staticmethod
    def _normalize_date(date_str: str) -> str:
        """Fast date normalization."""
        if "/" in date_str:
            parts = date_str.split("/")
            if len(parts) == 3:
                d, m, y = parts
                return f"{y}-{m.zfill(2)}-{d.zfill(2)}"
        return date_str

    @classmethod
    def extract_booking_info_fast(cls, message: str) -> Dict:
        """Fast extraction with minimal processing."""
        info = {}
        
        # Extract room ID
        for pattern in cls.ROOM_PATTERNS:
            match = pattern.search(message)
            if match:
                room_num = match.group(1)
                info['room_id'] = f'OH{room_num}'
                break
        
        # Extract guest name
        for pattern in cls.GUEST_PATTERNS:
            match = pattern.search(message)
            if match:
                candidate_name = match.group(1).strip()
                if cls._is_valid_vietnamese_name_cached(candidate_name):
                    info['guest_name'] = candidate_name
                    break
        
        # Extract phone
        for pattern in cls.PHONE_PATTERNS:
            match = pattern.search(message)
            if match:
                phone = match.group(1)
                if re.match(r'^0\d{9,10}$', phone):
                    info['phone_number'] = phone
                    break
        
        # Extract dates
        for pattern in cls.DATE_PATTERNS:
            match = pattern.search(message)
            if match:
                info['check_in'] = cls._normalize_date(match.group(1))
                info['check_out'] = cls._normalize_date(match.group(2))
                break
        
        # Extract guest count
        for pattern in cls.GUEST_COUNT_PATTERNS:
            match = pattern.search(message)
            if match:
                count = int(match.group(1))
                if 1 <= count <= 10:
                    info['num_guests'] = count
                    break
        
        # Try to extract JSON blocks (simplified)
        json_match = re.search(r'\{[\s\S]*?\}', message)
        if json_match:
            try:
                json_data = json.loads(json_match.group())
                for key in ['room_id', 'guest_name', 'phone_number', 'check_in', 'check_out', 'guests']:
                    if key in json_data and json_data[key]:
                        if key == 'guests':
                            info['num_guests'] = json_data[key]
                        else:
                            info[key] = json_data[key]
            except:
                pass
        
        return {k: v for k, v in info.items() if v is not None}

# ===================== State & Models (Simplified) =====================
class BookingIntent(str, Enum):
    GREETING = "greeting"
    BOOKING_REQUEST = "booking_request"
    PROVIDE_INFO = "provide_info"
    CONFIRM = "confirm"
    CANCEL = "cancel"
    UNCLEAR = "unclear"

class BookingInfo(BaseModel):
    room_id: Optional[str] = None
    check_in: Optional[str] = None
    check_out: Optional[str] = None
    guest_name: Optional[str] = None
    phone_number: Optional[str] = None
    num_guests: Optional[int] = None

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
    should_purge: bool

# ===================== Optimized MCP Tools =====================
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
            except:
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

# ===================== Optimized Workflow =====================
class OptimizedBookingWorkflow:
    def __init__(self, mcp_session: ClientSession):
        # Use faster, simpler LLM config
        self.llm = ChatGoogleGenerativeAI(
            model=MODEL, 
            temperature=0.2,  # Lower temperature for faster responses
            convert_system_message_to_human=True,
            max_tokens=150  # Limit tokens for speed
        )
        self.tools = [CreateBookingTool(session=mcp_session)]
        self.parser = OptimizedContextParser()
        
        # Intent detection cache
        self._intent_cache = {}

    @lru_cache(maxsize=100)
    def _detect_intent_fast(self, message_hash: str, message: str) -> BookingIntent:
        """Fast rule-based intent detection with cache."""
        s = message.lower().strip()
        
        # Fast pattern matching
        if any(w in s for w in ["hi", "hello", "xin chào", "chào"]):
            return BookingIntent.GREETING
        
        if any(w in s for w in ["đặt", "book", "booking", "muốn đặt"]) or "oh" in s:
            return BookingIntent.BOOKING_REQUEST
        
        if any(w in s for w in ["có", "yes", "ok", "xác nhận", "đồng ý"]) and len(s) < 20:
            return BookingIntent.CONFIRM
        
        if any(w in s for w in ["không", "no", "hủy", "cancel", "thôi"]):
            return BookingIntent.CANCEL
        
        if any(ch.isdigit() for ch in s) or any(w in s for w in ["tên", "tôi là", "ngày"]):
            return BookingIntent.PROVIDE_INFO
        
        return BookingIntent.UNCLEAR

    async def detect_intent(self, state: BookingState) -> BookingState:
        """Optimized intent detection with minimal LLM usage."""
        state.setdefault("messages", [])
        state.setdefault("booking_data", BookingInfo())
        state.setdefault("conversation_stage", "greeting")
        state.setdefault("confirmation_pending", False)
        state.setdefault("error_count", 0)
        
        msg = state.get("user_input", "")
        
        # Fast context extraction
        context_info = self.parser.extract_booking_info_fast(msg)
        
        # Use rule-based intent detection (no LLM call)
        if state.get("confirmation_pending"):
            s = msg.strip().lower()
            if any(w in s for w in ["có", "ok", "yes", "đồng ý", "được", "ừ"]):
                intent = BookingIntent.CONFIRM
            elif any(w in s for w in ["không", "no", "hủy", "cancel", "thôi"]):
                intent = BookingIntent.CANCEL
            else:
                intent = BookingIntent.UNCLEAR
        else:
            # Use cached intent detection
            msg_hash = str(hash(msg[:100]))  # Quick hash for caching
            intent = self._detect_intent_fast(msg_hash, msg)
        
        state["intent"] = intent
        state["messages"].append(HumanMessage(content=msg))
        state["next_action"] = self._route_intent(intent, state)
        
        # Merge context info if found
        if context_info:
            current = state.get("booking_data", BookingInfo())
            merged_data = self._merge_booking_info(current, BookingInfo(**context_info))
            state["booking_data"] = merged_data
        
        return state

    def _route_intent(self, intent: BookingIntent, state: BookingState) -> str:
        """Fast routing logic."""
        stage = state.get("conversation_stage", "greeting")
        
        if intent == BookingIntent.GREETING:
            return "handle_greeting"
        elif intent == BookingIntent.BOOKING_REQUEST:
            return "handle_booking_request"
        elif intent == BookingIntent.CONFIRM and state.get("confirmation_pending"):
            return "execute_booking"
        elif intent == BookingIntent.CANCEL:
            return "handle_cancel"
        elif intent == BookingIntent.PROVIDE_INFO:
            return "collect_info"
        else:
            return "handle_unclear"

    async def handle_booking_request(self, state: BookingState) -> BookingState:
        """Handle booking request without LLM extraction."""
        state.setdefault("messages", [])
        
        # Use fast parser instead of LLM
        msg = state.get("user_input", "")
        extracted_info = self.parser.extract_booking_info_fast(msg)
        
        if extracted_info:
            current = state.get("booking_data", BookingInfo())
            merged = self._merge_booking_info(current, BookingInfo(**extracted_info))
            state["booking_data"] = merged
        
        missing = self._get_missing_fields(state["booking_data"])
        
        if not missing:
            resp = self._format_confirmation_summary(state["booking_data"])
            state.update(
                conversation_stage="confirming",
                confirmation_pending=True,
                next_action="wait_input"
            )
        else:
            resp = self._ask_for_missing_fields(missing, state["booking_data"])
            state.update(
                conversation_stage="collecting",
                next_action="wait_input"
            )
        
        state["messages"].append(AIMessage(content=resp))
        state["last_response"] = resp
        return state

    async def collect_info(self, state: BookingState) -> BookingState:
        """Fast info collection without LLM."""
        state.setdefault("messages", [])
        
        msg = state.get("user_input", "")
        extracted_info = self.parser.extract_booking_info_fast(msg)
        
        if extracted_info:
            current = state.get("booking_data", BookingInfo())
            merged = self._merge_booking_info(current, BookingInfo(**extracted_info))
            state["booking_data"] = merged
        
        missing = self._get_missing_fields(state["booking_data"])
        
        if not missing:
            resp = self._format_confirmation_summary(state["booking_data"])
            state.update(
                conversation_stage="confirming",
                confirmation_pending=True
            )
        else:
            resp = self._ask_for_missing_fields(missing, state["booking_data"])
        
        state["messages"].append(AIMessage(content=resp))
        state["last_response"] = resp
        state["next_action"] = "wait_input"
        return state

    async def execute_booking(self, state: BookingState) -> BookingState:
        """Execute booking with fake mode for speed."""
        state.setdefault("messages", [])
        d = state["booking_data"]
        
        try:
            # Quick validation
            if not self._validate_booking_data(d):
                resp = "❌ Thông tin chưa hợp lệ. Vui lòng kiểm tra lại."
                state["next_action"] = "collect_info"
                state["messages"].append(AIMessage(content=resp))
                state["last_response"] = resp
                return state
            
            # Use fake booking if enabled for speed
            if BOOKING_ALLOW_FAKE:
                bid = f"DEMO-{d.room_id}-{d.check_in.replace('-', '')}"
                resp = (
                    "✅ ĐẶT PHÒNG THÀNH CÔNG!\n\n"
                    f"📋 Mã booking: {bid}\n"
                    f"🏨 Phòng: {d.room_id}\n"
                    f"👤 Khách: {d.guest_name}\n"
                    f"📅 Check-in: {d.check_in}\n"
                    f"📅 Check-out: {d.check_out}\n"
                    f"📱 SĐT: {d.phone_number or 'Không có'}\n"
                )
                state.update(
                    conversation_stage="ready",
                    confirmation_pending=False,
                    booking_data=BookingInfo(),
                    next_action="wait_input"
                )
            else:
                # Real booking call
                payload = {
                    k: getattr(d, k) 
                    for k in ["room_id", "check_in", "check_out", "guest_name", "phone_number"] 
                    if getattr(d, k) is not None
                }
                
                try:
                    result_raw = await self.tools[0]._arun(**payload)
                    logger.info(f"MCP tool raw result: {result_raw}")
                    result = json.loads(result_raw)
                except Exception as e:
                    logger.error(f"MCP tool call failed: {e}")
                    resp = f"❌ Lỗi gọi MCP tool: {e}"
                    state["next_action"] = "wait_input"
                    state["messages"].append(AIMessage(content=resp))
                    state["last_response"] = resp
                    return state
                
                if result.get("success"):
                    bid = result.get("booking_id", "N/A")
                    details = result.get("booking_details", {})
                    total_amount = details.get("formatted_amount", "N/A")
                    nights = details.get("nights", 0)
                    room_type = details.get("room_type", "N/A")
                    
                    resp = (
                        "✅ ĐẶT PHÒNG THÀNH CÔNG!\n\n"
                        f"📋 Mã booking: {bid}\n"
                        f"🏨 Phòng: {d.room_id} ({room_type})\n"
                        f"👤 Khách: {d.guest_name}\n"
                        f"📅 Check-in: {d.check_in}\n"
                        f"📅 Check-out: {d.check_out}\n"
                        f"🌙 Số đêm: {nights}\n"
                        f"💰 Tổng tiền: {total_amount}\n"
                        f"📱 SĐT: {d.phone_number or 'Không có'}\n"
                    )
                    state.update(
                        conversation_stage="ready",
                        confirmation_pending=False,
                        booking_data=BookingInfo(),
                        next_action="wait_input"
                    )
                else:
                    msg = result.get("message", "Lỗi không xác định")
                    resp = f"❌ Không thể đặt phòng: {msg}"
                    state["next_action"] = "wait_input"
        
        except Exception as e:
            logger.error(f"Execute booking error: {e}")
            resp = f"❌ Lỗi hệ thống: {e}"
            state["next_action"] = "wait_input"
        
        state["messages"].append(AIMessage(content=resp))
        state["last_response"] = resp
        return state

    def handle_greeting(self, state: BookingState) -> BookingState:
        """Fast greeting handler."""
        state.setdefault("messages", [])
        resp = (
            "🏨 Xin chào! Tôi là trợ lý đặt phòng Ohana Hotel.\n\n"
            "Bạn chỉ cần cho biết: • Phòng muốn đặt (VD: OH203) • Ngày check-in/check-out • Tên và số điện thoại\n"
            "Bạn muốn đặt phòng nào?"
        )
        state.update(
            conversation_stage="greeting",
            next_action="wait_input"
        )
        state["messages"].append(AIMessage(content=resp))
        state["last_response"] = resp
        return state

    def handle_cancel(self, state: BookingState) -> BookingState:
        """Fast cancel handler."""
        state.setdefault("messages", [])
        state.update(
            booking_data=BookingInfo(),
            confirmation_pending=False,
            conversation_stage="greeting",
            next_action="reset_state",
            should_purge=True
        )
        resp = "Đã hủy quá trình đặt phòng. Bạn có thể bắt đầu lại bất cứ lúc nào."
        state["messages"].append(AIMessage(content=resp))
        state["last_response"] = resp
        return state

    def handle_unclear(self, state: BookingState) -> BookingState:
        """Fast unclear handler."""
        state.setdefault("messages", [])
        stage = state.get("conversation_stage", "greeting")
        
        if stage == "confirming" and state.get("confirmation_pending"):
            resp = "Bạn muốn xác nhận đặt phòng không? (Có/Không)"
        elif stage == "collecting":
            missing = self._get_missing_fields(state.get("booking_data"))
            if missing:
                resp = self._ask_for_missing_fields(missing, state.get("booking_data"))
            else:
                resp = self._format_confirmation_summary(state["booking_data"])
        else:
            resp = "Tôi có thể giúp bạn đặt phòng. Bạn muốn đặt phòng nào, ngày nào?"
        
        state["messages"].append(AIMessage(content=resp))
        state["last_response"] = resp
        state["next_action"] = "wait_input"
        return state

    # Helper methods
    def _merge_booking_info(self, old: Optional[BookingInfo], new: BookingInfo) -> BookingInfo:
        """Fast merge without validation."""
        if not old:
            return new
        data = old.model_dump()
        for k, v in new.model_dump().items():
            if v is not None:
                data[k] = v
        return BookingInfo(**data)

    def _get_missing_fields(self, d: Optional[BookingInfo]) -> List[str]:
        """Fast missing field check."""
        if not d:
            return ["room_id", "check_in", "check_out", "guest_name"]
        required = ["room_id", "check_in", "check_out", "guest_name", "phone_number"]
        return [k for k in required if not getattr(d, k, None)]

    def _ask_for_missing_fields(self, missing: List[str], d: Optional[BookingInfo]) -> str:
        """Fast missing field message."""
        labels = {
            "room_id": "số phòng",
            "check_in": "ngày check-in", 
            "check_out": "ngày check-out",
            "guest_name": "tên khách",
            "phone_number": "số điện thoại"
        }
        
        have = []
        if d:
            if d.room_id: have.append(f"Phòng {d.room_id}")
            if d.guest_name: have.append(f"Khách {d.guest_name}")
            if d.check_in: have.append(f"Check-in {d.check_in}")
            if d.check_out: have.append(f"Check-out {d.check_out}")
        
        prefix = f"Tôi đã có: {', '.join(have)}\n\n" if have else ""
        return prefix + "Vui lòng cho biết: " + ", ".join(labels.get(f, f) for f in missing)

    def _format_confirmation_summary(self, d: BookingInfo) -> str:
        """Fast confirmation format."""
        return (
            "📋 XÁC NHẬN THÔNG TIN ĐẶT PHÒNG:\n\n"
            f"🏨 Phòng: {d.room_id}\n"
            f"📅 Check-in: {d.check_in}\n"
            f"📅 Check-out: {d.check_out}\n"
            f"👤 Tên khách: {d.guest_name}\n"
            f"📱 Điện thoại: {d.phone_number or 'Chưa có'}\n\n"
            "✅ Xác nhận đặt phòng? (Gõ 'Có' hoặc 'Không')"
        )

    def _validate_booking_data(self, d: BookingInfo) -> bool:
        """Fast validation."""
        try:
            if not all([d.room_id, d.check_in, d.check_out, d.guest_name]):
                return False
            ci = datetime.strptime(d.check_in, "%Y-%m-%d").date()
            co = datetime.strptime(d.check_out, "%Y-%m-%d").date()
            return TODAY <= ci < co
        except:
            return False

# ===================== Optimized Agent =====================
class BookingAgent:
    """Simplified, fast booking agent with minimal overhead."""
    
    def __init__(self):
        self._graph = None
        self._checkpointer: Optional[AsyncSqliteSaver] = None
        self._stack: AsyncExitStack | None = None
        self._initialized = False
        self._mcp_session: ClientSession | None = None
        
        # Simplified caching
        self._last_tools_check = 0.0
        self._tools_cache: set[str] = set()

    async def initialize(self, server_path: str | None = None):
        """Fast initialization with minimal setup."""
        if self._initialized:
            return
        
        try:
            mcp_path = Path(server_path or SERVER_PATH).resolve()
            if not mcp_path.exists():
                raise FileNotFoundError(f"MCP server not found: {mcp_path}")
            
            project_root = mcp_path.parents[1] if len(mcp_path.parents) >= 2 else mcp_path.parent
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONPATH"] = os.pathsep.join([str(project_root), env.get("PYTHONPATH", "")])
            
            self._stack = AsyncExitStack()
            
            # Initialize MCP with faster settings
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
            
            # Cache tools for faster access
            try:
                tools_list = await session.list_tools()
                self._tools_cache = {t.name for t in (tools_list.tools or [])}
                logger.info(f"MCP tools cached: {sorted(self._tools_cache)}")
            except Exception as e:
                logger.warning(f"Tool caching failed: {e}")

            # Simple in-memory checkpointer for speed
            if SESSION_DB_EPHEMERAL:
                self._checkpointer = await self._stack.enter_async_context(
                    AsyncSqliteSaver.from_conn_string(":memory:")
                )
                logger.info("Using in-memory SQLite for maximum speed")
            else:
                # Single shared DB file
                Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
                uri = f"sqlite+aiosqlite:///{DB_PATH.as_posix()}"
                try:
                    self._checkpointer = await self._stack.enter_async_context(
                        AsyncSqliteSaver.from_conn_string(uri)
                    )
                except Exception as e:
                    logger.warning(f"Using fallback SQLite connection: {e}")
                    db = await aiosqlite.connect(DB_PATH.as_posix())
                    await db.execute("PRAGMA journal_mode=WAL;")
                    await db.execute("PRAGMA synchronous=NORMAL;")  # Faster writes
                    await db.execute("PRAGMA cache_size=10000;")     # More cache
                    await db.commit()
                    self._checkpointer = AsyncSqliteSaver(db)
                
                logger.info(f"Using SQLite DB: {DB_PATH}")

            # Build optimized graph
            workflow = OptimizedBookingWorkflow(session)
            graph = StateGraph(BookingState)
            
            # Add nodes
            graph.add_node("detect_intent", workflow.detect_intent)
            graph.add_node("handle_greeting", workflow.handle_greeting)
            graph.add_node("handle_booking_request", workflow.handle_booking_request)
            graph.add_node("collect_info", workflow.collect_info)
            graph.add_node("execute_booking", workflow.execute_booking)
            graph.add_node("handle_cancel", workflow.handle_cancel)
            graph.add_node("handle_unclear", workflow.handle_unclear)
            
            # Set entry point
            graph.set_entry_point("detect_intent")
            
            # Simplified routing
            def route_after_intent(state: BookingState) -> str:
                return state["next_action"]
            
            def route_after_action(state: BookingState) -> str:
                next_action = state.get("next_action", "wait_input")
                return END if next_action in {"wait_input", "complete", "reset_state"} else next_action
            
            # Add edges
            graph.add_conditional_edges(
                "detect_intent",
                route_after_intent,
                {
                    "handle_greeting": "handle_greeting",
                    "handle_booking_request": "handle_booking_request", 
                    "collect_info": "collect_info",
                    "execute_booking": "execute_booking",
                    "handle_cancel": "handle_cancel",
                    "handle_unclear": "handle_unclear",
                }
            )
            
            # All action nodes route the same way
            for node in ["handle_greeting", "handle_booking_request", "collect_info", 
                        "handle_cancel", "handle_unclear"]:
                graph.add_conditional_edges(node, route_after_action)
            
            # Compile with checkpointer
            self._graph = graph.compile(checkpointer=self._checkpointer)
            self._initialized = True
            
            logger.info("OptimizedBookingAgent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OptimizedBookingAgent: {e}")
            await self.aclose()
            raise

    async def _ensure_mcp_alive(self):
        """Lightweight MCP health check."""
        if not self._mcp_session or not self._initialized:
            await self.initialize()
            return
        
        # Only check tools every 30 seconds to reduce overhead
        if time.time() - self._last_tools_check < 30:
            return
        
        try:
            tools_list = await self._mcp_session.list_tools()
            self._tools_cache = {t.name for t in (tools_list.tools or [])}
            self._last_tools_check = time.time()
        except Exception as e:
            logger.warning(f"MCP health check failed: {e}")
            # Don't reinitialize immediately to avoid cascading failures

    async def aclose(self):
        """Fast cleanup."""
        self._initialized = False
        if self._stack:
            with contextlib.suppress(Exception):
                await self._stack.aclose()
            self._stack = None
        self._graph = None
        self._mcp_session = None
        self._tools_cache = set()

    async def chat(self, message: str, session_id: str = "default") -> str:
        """Fast chat processing with minimal overhead."""
        if not self._initialized:
            await self.initialize()
        
        if not self._mcp_session:
            return "Lỗi: MCP chưa sẵn sàng."
        
        # Lightweight health check
        await self._ensure_mcp_alive()
        
        try:
            # Simple config without complex namespacing
            config = {"configurable": {"thread_id": session_id}}
            
            # Single graph invocation
            result_state: BookingState = await self._graph.ainvoke(
                {"user_input": message}, 
                config=config
            )
            
            return result_state.get("last_response", "Xin lỗi, có lỗi xảy ra.")
            
        except Exception as e:
            logger.exception(f"Chat error (session {session_id}): {e}")
            return f"Đã có lỗi khi xử lý: {str(e)}"

    async def process_a2a(self, payload: Dict) -> Dict:
        """Fast A2A processing."""
        if not self._initialized:
            await self.initialize()
        
        await self._ensure_mcp_alive()
        
        session_id = ((payload.get("session") or {}).get("id")) or "default"
        text = ((payload.get("input") or {}).get("text")) or ""
        
        config = {"configurable": {"thread_id": session_id}}
        result_state: BookingState = await self._graph.ainvoke(
            {"user_input": text}, 
            config=config
        )
        
        reply = result_state.get("last_response", "")
        state_out = {
            "conversation_stage": result_state.get("conversation_stage"),
            "booking_data": (result_state.get("booking_data") or BookingInfo()).dict(),
            "confirmation_pending": result_state.get("confirmation_pending", False),
            "intent": str(result_state.get("intent").value) if result_state.get("intent") else "",
            "last_response": reply,
        }
        
        return {
            "messages": [{"role": "assistant", "content": reply}], 
            "state": state_out, 
            "session": {"id": session_id}
        }

# ===================== Performance Test =====================
async def benchmark_agent():
    """Quick performance benchmark."""
    agent = BookingAgent()
    
    test_messages = [
        "Xin chào",
        "Tôi muốn đặt phòng OH203 từ 2025-09-25 đến 2025-09-27",
        "Tên khách Nguyễn Văn An, số điện thoại 0901234567", 
        "Có, xác nhận đặt phòng",
    ]
    
    session_id = "benchmark_test"
    
    print("🚀 Starting performance benchmark...")
    start_time = time.time()
    
    for i, msg in enumerate(test_messages):
        msg_start = time.time()
        response = await agent.chat(msg, session_id=session_id)
        msg_duration = time.time() - msg_start
        print(f"Message {i+1}: {msg_duration:.2f}s - {response[:50]}...")
    
    total_duration = time.time() - start_time
    print(f"📊 Total time: {total_duration:.2f}s")
    print(f"📊 Average per message: {total_duration/len(test_messages):.2f}s")
    
    await agent.aclose()

# ===================== Main Entry Point =====================
async def main():
    """Test the optimized agent."""
    agent = BookingAgent()
    
    queries = [
        "Xin chào",
        "Tôi muốn đặt phòng OH102 từ 2025-09-23 đến 2025-09-25 cho 2 người",
        "Tên khách Khánh Hòa, số điện thoại 0937401803",
        "Có, xác nhận đặt phòng",
    ]
    
    session_id = "test_optimized"
    
    for query in queries:
        print(f"👤 User: {query}")
        start = time.time()
        response = await agent.chat(query, session_id=session_id)
        duration = time.time() - start
        print(f"🤖 Bot ({duration:.2f}s): {response}\n")
    
    await agent.aclose()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        asyncio.run(benchmark_agent())
    else:
        asyncio.run(main())