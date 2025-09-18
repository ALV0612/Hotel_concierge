import os, sys, json, asyncio, re, logging, warnings
from typing import Any, Dict, List, Optional, Type, Set
from pathlib import Path
from datetime import datetime, date, timedelta
from contextlib import AsyncExitStack

from pydantic import BaseModel, Field, ConfigDict
from dateutil import parser
from dotenv import load_dotenv

# CrewAI
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

# MCP (native client)
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

# nest_asyncio để wrap async -> sync tool.run
import nest_asyncio
import threading
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]  # C:\Desktop\test
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

# --- Safe runner for coroutines from sync contexts ---
def _run_coro_sync(coro):
    """Run an async coroutine from any context safely."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        nest_asyncio.apply(loop)
        return loop.run_until_complete(coro)

load_dotenv()
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['PYTHONUTF8'] = '1'
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# -------------------------
# Enhanced Config with Context Awareness - GPT-4 Mini
# -------------------------
TODAY = datetime.now().date()

def _resolve_model_name() -> str:
    # Support both OpenAI and Gemini models
    raw = os.getenv("OHANA_LLM_MODEL", "openai/gpt-4o-mini")
    
    # Handle OpenAI models
    if raw.startswith("gpt-"):
        return f"openai/{raw}"
    elif raw.startswith("openai/"):
        return raw
    
    # Handle Gemini models (fallback)
    if raw.startswith("gemini"):
        return f"gemini/{raw}" if "/" not in raw else raw
    
    # Default
    if "/" in raw:
        return raw
    
    return f"openai/{raw}"

MODEL = _resolve_model_name()

# API Key mapping for GPT-4 Mini
if not os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")

# Keep Gemini mapping for fallback
if not os.getenv("GEMINI_API_KEY") and os.getenv("GOOGLE_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# -------------------------
# Context Extraction Utils
# -------------------------
class ContextExtractor:
    """Extract useful context from Host Agent messages"""
    
    @staticmethod
    def extract_guest_count(text: str) -> Optional[int]:
        """Extract guest count from context"""
        patterns = [
            r'(\d+)\s+người',
            r'cho\s+(\d+)\s+khách',
            r'(\d+)\s+khách',
            r'phòng\s+cho\s+(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None
    
    @staticmethod
    def extract_dates(text: str) -> Dict[str, Optional[str]]:
        """Extract check-in/check-out dates"""
        result = {"check_in": None, "check_out": None}
        
        # Date range patterns
        range_patterns = [
            r'từ\s+ngày\s+(\d{4}-\d{2}-\d{2})\s+đến\s+(\d{4}-\d{2}-\d{2})',
            r'từ\s+(\d{4}-\d{2}-\d{2})\s+đến\s+(\d{4}-\d{2}-\d{2})',
            r'(\d{4}-\d{2}-\d{2})\s+đến\s+(\d{4}-\d{2}-\d{2})'
        ]
        
        for pattern in range_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["check_in"] = match.group(1)
                result["check_out"] = match.group(2)
                return result
        
        # Individual date patterns
        if 'check-in' in text.lower():
            match = re.search(r'check-in\s+(\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
            if match:
                result["check_in"] = match.group(1)
        
        if 'check-out' in text.lower():
            match = re.search(r'check-out\s+(\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
            if match:
                result["check_out"] = match.group(1)
        
        return result

# -------------------------
# Vietnamese Date Parser (Enhanced)
# -------------------------
class VietDateParser:
    DATE_PATTERNS = {
        r'hôm nay|today': 0,
        r'ngày mai|tomorrow': 1,
        r'ngày mốt': 2,
        r'cuối tuần|weekend': None,
        r'thứ hai': 0, r'thứ ba': 1, r'thứ tư': 2,
        r'thứ năm': 3, r'thứ sáu': 4, r'thứ bảy': 5, r'chủ nhật': 6
    }

    @classmethod
    def parse_with_context(cls, text: str, context_dates: Dict = None) -> Optional[str]:
        """Enhanced parsing with context awareness"""
        if not text:
            return None
            
        # First try context dates if available
        if context_dates:
            if 'check_in' in context_dates and context_dates['check_in']:
                return context_dates['check_in']
        
        # Then try regular parsing
        return cls.parse(text)
    
    @classmethod
    def parse(cls, text: str) -> Optional[str]:
        if not text:
            return None
        text = text.lower().strip()

        for pattern, offset in cls.DATE_PATTERNS.items():
            if re.search(pattern, text):
                if isinstance(offset, int):
                    if 'thứ' in pattern or 'chủ nhật' in pattern:
                        return cls._get_next_weekday(offset)
                    else:
                        target_date = TODAY + timedelta(days=offset)
                        return target_date.strftime('%Y-%m-%d')
                elif offset is None and 'cuối tuần' in pattern:
                    days_ahead = (5 - TODAY.weekday()) % 7
                    if days_ahead == 0:
                        days_ahead = 7
                    target_date = TODAY + timedelta(days=days_ahead)
                    return target_date.strftime('%Y-%m-%d')

        # dạng 12/9, 12-14/9
        if '/' in text:
            try:
                if '-' in text:
                    parts = text.split('-')
                    start_part = parts[0] + '/' + parts[1].split('/')[1]
                    return cls._parse_viet_format(start_part)
                else:
                    return cls._parse_viet_format(text)
            except:
                pass

        try:
            parsed = parser.parse(text)
            return parsed.date().strftime('%Y-%m-%d')
        except:
            return None

    @classmethod
    def _parse_viet_format(cls, text: str) -> Optional[str]:
        try:
            parts = text.strip().split('/')
            if len(parts) >= 2:
                day, month = int(parts[0]), int(parts[1])
                year = TODAY.year
                if month < TODAY.month or (month == TODAY.month and day < TODAY.day):
                    year += 1
                return date(year, month, day).strftime('%Y-%m-%d')
        except:
            return None

    @classmethod
    def _get_next_weekday(cls, weekday: int) -> str:
        days_ahead = weekday - TODAY.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        target_date = TODAY + timedelta(days=days_ahead)
        return target_date.strftime('%Y-%m-%d')

# -------------------------
# Enhanced MCP Threaded Client
# -------------------------
class MCPThreadedClient:
    def __init__(self, server_path: Optional[str] = None):
        server_py = Path(os.getenv("OHANA_MCP_SERVER", BASE_DIR / "mcp_server" / "server_info_mcp.py")).resolve()
        if not server_py.exists():
            raise FileNotFoundError(f"MCP server not found: {server_py}")
        self.server_py = server_py
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._stopping = threading.Event()
        self._stack: Optional[AsyncExitStack] = None
        self.session: Optional[ClientSession] = None

    def _thread_main(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._async_start())
            self._ready.set()
            self.loop.run_forever()
        finally:
            try:
                self.loop.run_until_complete(self._async_stop())
            except Exception as e:
                logger.warning("MCP shutdown warning: %s", e)
            self.loop.close()

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._thread_main, name="MCP-Loop", daemon=True)
        self.thread.start()
        self._ready.wait(timeout=10)
        if not self._ready.is_set():
            raise RuntimeError("Failed to start MCP thread/loop")

    def stop(self):
        if not (self.thread and self.thread.is_alive() and self.loop):
            return
        if not self._stopping.is_set():
            self._stopping.set()
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)
        self.thread = None
        self.loop = None

    async def _async_start(self):
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONPATH"] = os.pathsep.join([str(BASE_DIR), env.get("PYTHONPATH", "")])

        python_exe = sys.executable
        params = StdioServerParameters(
            command=python_exe,
            args=["-u", "-X", "utf8", str(self.server_py)],
            cwd=str(BASE_DIR),
            env=env,
        )
        self._stack = AsyncExitStack()
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()

    async def _async_stop(self):
        try:
            if self._stack is not None:
                await self._stack.aclose()
        finally:
            self._stack = None
            self.session = None

    def call(self, name: str, args: Dict[str, Any]):
        self.start()
        assert self.loop and self.session, "MCP not started"
        fut = asyncio.run_coroutine_threadsafe(self.session.call_tool(name, args), self.loop)
        return fut.result()

    def list_tools(self) -> List[str]:
        self.start()
        assert self.loop and self.session, "MCP not started"
        async def _list():
            info = await self.session.list_tools()
            return [t.name for t in info.tools]
        fut = asyncio.run_coroutine_threadsafe(_list(), self.loop)
        return fut.result()

# -------------------------
# Enhanced MCP Tools with Context Awareness
# -------------------------
class MCPTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "mcp_tool"
    description: str = "Wrapper tool"
    args_schema: Type[BaseModel] = BaseModel
    client: Any

    def format_result(self, result: types.CallToolResult) -> str:
        try:
            if getattr(result, "structured_content", None):
                return json.dumps(result.structured_content, ensure_ascii=False, indent=2)
            parts = []
            for c in result.content:
                if hasattr(c, "text") and c.text is not None:
                    parts.append(c.text)
                else:
                    parts.append(str(c))
            combined = "\n".join(parts).strip()
            if not combined:
                combined = "(no result)"
            try:
                parsed = json.loads(combined)
                return json.dumps(parsed, ensure_ascii=False, indent=2)
            except:
                return combined
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

# Input Schemas
class CheckAvailabilityInput(BaseModel):
    check_in: str = Field(description="Check-in (YYYY-MM-DD) hoặc tiếng Việt")
    check_out: str = Field(description="Check-out (YYYY-MM-DD) hoặc tiếng Việt")
    guests: int = Field(default=1, description="Số khách")
    room_type: Optional[str] = Field(default=None, description="Lọc theo loại phòng")

class RoomIdInput(BaseModel):
    room_id: str = Field(description="ID phòng, vd OH201")

class EmptyInput(BaseModel):
    pass

class DocumentSearchInput(BaseModel):
    query: str = Field(description="Query tiếng Việt/Anh")
    top_k: int = Field(default=3, description="Số kết quả")

# Enhanced Tools
class CheckAvailabilityTool(MCPTool):
    name: str = "check_availability"
    description: str = "Tìm phòng trống với context awareness"
    args_schema: Type[BaseModel] = CheckAvailabilityInput

    def _run(self, check_in: str, check_out: str, guests: int = 1, room_type: Optional[str] = None) -> str:
        context_extractor = ContextExtractor()
        
        # Try to extract additional context
        full_message = getattr(self, '_current_message', '')
        if full_message:
            context_guests = context_extractor.extract_guest_count(full_message)
            if context_guests and guests == 1:
                guests = context_guests
                
            context_dates = context_extractor.extract_dates(full_message)
            ci = VietDateParser.parse_with_context(check_in, context_dates) or (TODAY + timedelta(days=1)).strftime("%Y-%m-%d")
            co = VietDateParser.parse_with_context(check_out, context_dates)
        else:
            ci = VietDateParser.parse(check_in) or (TODAY + timedelta(days=1)).strftime("%Y-%m-%d")
            co = VietDateParser.parse(check_out)
            
        if not co:
            co = (datetime.strptime(ci, "%Y-%m-%d").date() + timedelta(days=1)).strftime("%Y-%m-%d")
            
        args = {"check_in": ci, "check_out": co, "guests": guests}
        if room_type:
            args["room_type"] = room_type
            
        result = self.client.call("check_availability", args)
        return self.format_result(result)

class GetRoomInfoTool(MCPTool):
    name: str = "get_room_info"
    description: str = "Lấy thông tin chi tiết phòng"
    args_schema: Type[BaseModel] = RoomIdInput
    
    def _run(self, room_id: str) -> str:
        result = self.client.call("get_room_info", {"room_id": room_id})
        return self.format_result(result)

class QueryHotelDocsTool(MCPTool):
    name: str = "query_hotel_docs"
    description: str = "Tìm kiếm nội quy/dịch vụ/quy định (RAG)"
    args_schema: Type[BaseModel] = DocumentSearchInput
    
    def _run(self, query: str, top_k: int = 3) -> str:
        try:
            full_message = getattr(self, '_current_message', '')
            if full_message and 'context' in full_message.lower():
                enhanced_query = f"Context: {full_message}\n\nQuery: {query}"
            else:
                enhanced_query = query
                
            result = self.client.call("query_hotel_docs", {"question": enhanced_query, "top_k": top_k})
            return self.format_result(result)
        except Exception as e:
            return json.dumps({"error": f"Lỗi RAG: {e}"}, ensure_ascii=False)

def build_tools(client: MCPThreadedClient) -> List[BaseTool]:
    return [
        CheckAvailabilityTool(client=client),
        GetRoomInfoTool(client=client),
        QueryHotelDocsTool(client=client),
    ]

# -------------------------
# Enhanced Agent wrapper (CrewAI) with GPT-4 Mini Support
# -------------------------
class GetInfoAgentCrew:
    """Enhanced CrewAI GetInfo Agent with GPT-4 Mini and shared memory awareness"""

    def __init__(self, session_name: str = "default") -> None:
        self.session_name = session_name
        self.client = MCPThreadedClient()
        self.llm = LLM(model=MODEL, temperature=0.1)
        self.history: List[str] = []
        self.context_extractor = ContextExtractor()

        self._agent = Agent(
            role="Ohana Concierge with Shared Memory (GPT-4 Mini)",
            goal=(
                "Hiểu nhu cầu đặt phòng & câu hỏi về khách sạn với CONTEXT AWARENESS. "
                f"Hôm nay: {TODAY.strftime('%Y-%m-%d')}. "
                "Có thể nhận context từ Host Agent và xử lý thông minh với GPT-4 Mini."
            ),
            backstory=(
                "Bạn là trợ lý đặt phòng thông minh của khách sạn Ohana sử dụng GPT-4 Mini với khả năng: "
                "1) Xử lý context từ cuộc trò chuyện trước đó "
                "2) Tự động trích xuất thông tin từ message phức tạp "
                "3) Ưu tiên phòng có sức chứa CHÍNH XÁC theo yêu cầu "
                "4) Kết hợp RAG để trả lời về nội quy/dịch vụ với context. "
                "5) Tận dụng khả năng reasoning mạnh mẽ của GPT-4 Mini."
            ),
            llm=self.llm,
            verbose=False,
            allow_delegation=False,
        )

        self._task = Task(
            description=(
                "Người dùng nhắn: '{user_input}'.\n"
                "Lịch sử (shared memory context):\n{history}\n\n"
                "NHIỆM VỤ NÂNG CAO (GPT-4 Mini Enhanced):\n"
                "1) CONTEXT AWARENESS: Phân tích message để tìm context từ Host Agent\n"
                "   - Tìm thông tin: số khách, ngày tháng, loại phòng\n"
                "   - Tự động điền context vào tools khi cần\n"
                "2) Phân loại câu hỏi: (A) booking/phòng hoặc (B) nội quy/dịch vụ\n"
                "3) Nếu (A): dùng tools booking với context awareness\n"
                "   - CHUẨN HOÁ ngày Việt sang YYYY-MM-DD\n"
                "   - QUY ĐỊNH HIỂN THỊ PHÒNG:\n"
                "     • Ưu tiên phòng có capacity CHÍNH XÁC\n"
                "     • CHỈ hiển thị capacity +1 khi HẾT phòng đúng size\n"
                "4) Nếu (B): ưu tiên 'query_hotel_docs' với enhanced context\n"
                "5) SMART RESPONSE: Tận dụng GPT-4 Mini reasoning để trả lời thông minh\n"
                "6) ADVANCED REASONING: Sử dụng khả năng suy luận của GPT-4 Mini cho các câu hỏi phức tạp\n\n"
                "ĐẦU RA: Trả lời ngắn gọn, tiếng Việt, context-aware với reasoning chính xác."
            ),
            expected_output="Câu trả lời cuối cùng (tiếng Việt), tận dụng GPT-4 Mini reasoning và shared memory context.",
            agent=self._agent,
            tools=build_tools(self.client),
        )

    def ask(self, message: str) -> str:
        # Store current message for tools to access
        for tool in self._task.tools:
            if hasattr(tool, '_current_message'):
                tool._current_message = message
        
        crew = Crew(
            agents=[self._agent],
            tasks=[self._task],
            process=Process.sequential,
            verbose=False,
        )
        inputs = {
            "user_input": message,
            "history": "\n".join(self.history[-8:])
        }
        out = crew.kickoff(inputs=inputs)
        text = getattr(out, "raw", None) or getattr(out, "final_output", None) or str(out)
        self.history.extend([f"User: {message}", f"Assistant: {text}"])
        return text

    def debug_mcp(self) -> str:
        names = self.client.list_tools()
        return json.dumps({"available": names}, ensure_ascii=False, indent=2)

    def close(self):
        try:
            self.client.stop()
        except Exception as e:
            logger.warning("Ignore MCP stop error: %s", e)

# -------------------------
# CLI
# -------------------------
def _ensure_api_key():
    """Check for OpenAI API key (primary) or Gemini API key (fallback)"""
    if MODEL.startswith("openai/"):
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ Thiếu OPENAI_API_KEY để sử dụng GPT-4 Mini. Vui lòng đặt OPENAI_API_KEY.")
            sys.exit(1)
    elif MODEL.startswith("gemini/"):
        if not os.getenv("GEMINI_API_KEY"):
            print("❌ Thiếu GEMINI_API_KEY (hoặc GOOGLE_API_KEY). Đặt 1 trong 2.")
            sys.exit(1)
    else:
        print(f"⚠️ Model không được nhận dạng: {MODEL}")

def run_single_query(q: str) -> str:
    _ensure_api_key()
    agent = GetInfoAgentCrew()
    try:
        return agent.ask(q)
    finally:
        agent.close()

def run_chat():
    _ensure_api_key()
    agent = GetInfoAgentCrew()
    print("🏨 OHANA HOTEL - Enhanced CrewAI Concierge (GPT-4 Mini)")
    print(f"🤖 Model: {MODEL}")
    print(f"📅 Hôm nay: {TODAY.strftime('%d/%m/%Y')}")
    print("Gõ 'quit' để thoát")
    print("-"*60)
    try:
        while True:
            msg = input("\n👤 Bạn: ").strip()
            if msg.lower() in {"quit", "q", "exit", "thoát"}:
                print("\n👋 Tạm biệt!")
                break
            if not msg:
                continue
            print("🤖 Đang xử lý...")
            ans = agent.ask(msg)
            print(f"\n🏨 Trợ lý: {ans}")
    except KeyboardInterrupt:
        print("\n👋 Bye!")
    finally:
        agent.close()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        _ensure_api_key()
        agent = GetInfoAgentCrew()
        try:
            info = agent.debug_mcp()
            print("🔧 MCP tools:", info)
        finally:
            agent.close()
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "--chat":
        run_chat()
        sys.exit(0)

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(run_single_query(query))
    else:
        demo = "Context: Khách đã hỏi về phòng cho 4 người từ 2025-09-13 đến 2025-09-14. Yêu cầu mới: Còn phòng Family không?"
        print(f"🤖 Demo: {demo}")
        print(run_single_query(demo))
        print(f"\n💡 Chat: python {sys.argv[0]} --chat")
        print(f"💡 Debug MCP: python {sys.argv[0]} --debug")