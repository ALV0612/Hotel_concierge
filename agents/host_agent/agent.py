# -*- coding: utf-8 -*-
import os, uuid, json, asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime
import httpx
from dotenv import load_dotenv

# Google ADK
from google.adk import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext
from google.genai import types as gt

load_dotenv()

# ====== Config ======
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

def _normalize_local_url(url: str) -> str:
    """Giữ nguyên localhost và đảm bảo có dấu '/' ở cuối để post vào root."""
    if not url:
        return url
    # Không cần đổi localhost -> 127.0.0.1 nữa, giữ nguyên localhost
    if not url.endswith("/"):
        url += "/"
    return url

# ====== Config ======
BOOKING_URL = "http://localhost:9999"
INFO_URL    = "http://localhost:10002"

AGENT_MAP = {
    "Ohana Booking Agent": BOOKING_URL,
    "Ohana GetInfo Agent": INFO_URL,
}

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ====== Shared Memory System với Dynamic Sessions ======
class SharedMemoryService:
    def __init__(self):
        self.conversation_history: List[Dict[str, Any]] = []
        self.user_preferences: Dict[str, Any] = {}
        self.booking_context: Dict[str, Any] = {}
        self.current_session: Optional[str] = None
        self.current_booking_id: Optional[str] = None
        self.last_activity: Optional[float] = None
        self.session_start_time: Optional[float] = None

    def should_reset_session(self, message: str) -> bool:
        """Phát hiện có cần reset session dựa trên message hoặc timeout."""
        if not message:
            return False

        msg_lower = message.lower().strip()

        # Tín hiệu reset – lệnh rõ ràng
        reset_signals = [
            "xin chào", "chào", "hello", "hi", "chao",
            "tôi muốn đặt phòng mới", "đặt phòng mới",
            "booking mới", "new booking",
            "bắt đầu lại", "start over", "restart",
            "session mới", "conversation mới", "chat mới",
            "reset", "new", "làm lại",
        ]

        # Khớp chính xác hoặc mở đầu câu
        for signal in reset_signals:
            if msg_lower == signal or msg_lower.startswith(signal + " "):
                return True

        # Mẫu câu mở đầu hội thoại phổ biến
        import re
        starter_patterns = [
            r'^(xin\s+)?chào\b',
            r'^hello\b',
            r'^hi\b',
            r'^tôi\s+muốn\s+đặt\s+phòng',
            r'^đặt\s+phòng',
            r'^book\s+room',
        ]
        for pattern in starter_patterns:
            if re.match(pattern, msg_lower):
                # Chỉ reset nếu đã có lịch sử trước đó
                if len(self.conversation_history) > 0:
                    return True

        # Kiểm tra timeout – reset sau 1 giờ không hoạt động
        if self.last_activity:
            import time
            if time.time() - self.last_activity > 3600:  # 1 giờ
                return True

        return False

    def get_or_create_session(self, message: str = "") -> str:
        """Tạo session mới hoặc dùng session hiện tại, có auto-reset theo ngữ cảnh."""
        import time
        current_time = time.time()

        # Tự động reset dựa trên message hoặc timeout
        if message and self.should_reset_session(message):
            if self.current_session:
                print(f"🗑️ Auto-detected session reset signal in: '{message[:30]}...'")
                return self.start_new_conversation()

        # Tạo session đầu tiên nếu chưa có
        if not self.current_session:
            timestamp = int(datetime.now().timestamp())
            self.current_session = f"ohana-session-{timestamp}"
            self.session_start_time = current_time
            print(f"✨ Created new session: {self.current_session}")

        # Cập nhật thời gian hoạt động
        self.last_activity = current_time
        return self.current_session

    def start_new_conversation(self) -> str:
        """Bắt đầu cuộc trò chuyện mới với session mới."""
        import time
        timestamp = int(datetime.now().timestamp())
        old_session = self.current_session
        old_history_count = len(self.conversation_history)

        # Tạo session mới
        self.current_session = f"ohana-session-{timestamp}"
        now = time.time()
        self.session_start_time = now
        self.last_activity = now

        # Xóa bối cảnh cũ cho cuộc trò chuyện mới
        self.conversation_history = []
        self.booking_context = {}
        self.current_booking_id = None

        print("🆕 New conversation started!")
        print(f"   Previous: {old_session} ({old_history_count} messages)")
        print(f"   Current:  {self.current_session}")

        return self.current_session

    def manual_reset(self) -> str:
        """Reset thủ công – dùng cho lệnh CLI."""
        return self.start_new_conversation()

    def add_message(self, agent: str, user_message: str, agent_response: str) -> None:
        """Thêm message vào lịch sử hội thoại."""
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "user_message": user_message,
            "agent_response": agent_response,
            "booking_id": self.current_booking_id,
            "session_id": self.current_session,
        })
        print(f"📝 Added to memory - {agent}: {user_message[:50]}...")

        # Cập nhật thời gian hoạt động
        import time
        self.last_activity = time.time()

    def get_recent_context(self, last_n: int = 3) -> List[Dict[str, Any]]:
        """Lấy n tin nhắn gần nhất trong session hiện tại."""
        return self.conversation_history[-last_n:] if self.conversation_history else []

    def update_booking_context(self, context: Dict[str, Any]) -> None:
        """Cập nhật context cho booking hiện tại."""
        self.booking_context.update(context)
        print(f"📋 Updated booking context: {list(context.keys())}")

    def get_full_context_summary(self) -> str:
        """Tóm tắt ngắn gọn - CHỈ DÙNG NỘI BỘ, KHÔNG GỬI ĐI."""
        recent = self.get_recent_context(2)  # CHỈ 2 tin nhắn gần nhất
        summary = ""

        if recent:
            summary = "Recent:\n"
            for msg in recent:
                # Cắt ngắn hơn nữa
                user_msg = msg['user_message'][:30] + "..." if len(msg['user_message']) > 30 else msg['user_message']
                summary += f"- {msg['agent']}: {user_msg}\n"

        # Chỉ lưu thông tin quan trọng nhất
        if self.booking_context and self.booking_context.get('last_room_query'):
            lrq = self.booking_context['last_room_query']
            summary += f"\nBooking: {lrq.get('guests')}p"
            if lrq.get('check_in'): 
                summary += f", {lrq.get('check_in')}"
            summary += "\n"

        # GIỚI HẠN TỔNG CHIỀU DÀI
        if len(summary) > 200:
            summary = summary[:200] + "..."
        
        return summary

    def start_new_booking(self) -> str:
        """Bắt đầu booking mới trong session hiện tại."""
        self.current_booking_id = f"booking-{uuid.uuid4().hex[:8]}"
        self.booking_context = {"booking_id": self.current_booking_id}
        print(f"🏨 Started new booking {self.current_booking_id} in session {self.current_session}")
        return self.current_booking_id

    def complete_booking(self) -> Optional[str]:
        """Hoàn thành booking hiện tại, giữ nguyên session."""
        if self.current_booking_id:
            completed_id = self.current_booking_id
            print(f"✅ Completed booking {completed_id}")
            self.current_booking_id = None
            self.booking_context = {}
            return completed_id
        return None

    def mark_booking_complete(self) -> None:
        """Đánh dấu booking đã hoàn thành."""
        completed = self.complete_booking()
        if completed:
            print(f"🎉 Booking {completed} marked as complete")

    def get_session_stats(self) -> Dict[str, Any]:
        """Lấy thống kê của session hiện tại."""
        import time
        return {
            "session_id": self.current_session,
            "message_count": len(self.conversation_history),
            "current_booking": self.current_booking_id,
            "context_items": len(self.booking_context),
            "uptime_minutes": round((time.time() - self.session_start_time) / 60, 1) if self.session_start_time else 0,
        }

    def reset_session(self) -> str:
        """Legacy – tương đương start_new_conversation."""
        return self.start_new_conversation()

# Global shared memory instance
shared_memory = SharedMemoryService()

# ====== A2A helpers ======
async def _post_jsonrpc(base_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=30) as cli:
        r = await cli.post(base_url, json=payload)
        r.raise_for_status()
        return r.json()

def _extract_text_from_task(result: Dict[str, Any]) -> str:
    # Thử status.message.parts
    status = result.get("status") or {}
    msg = status.get("message") or {}
    parts = msg.get("parts") or []
    if parts and isinstance(parts[0], dict):
        return parts[0].get("text") or parts[0].get("data") or ""
    # Fallback: result.message
    msg2 = (result.get("message") or {})
    parts2 = msg2.get("parts") or []
    if parts2 and isinstance(parts2[0], dict):
        return parts2[0].get("text") or parts2[0].get("data") or ""
    # Fallback: artifacts
    arts = result.get("artifacts") or []
    if arts and arts[0].get("parts"):
        p0 = arts[0]["parts"][0]
        return p0.get("text", "") or p0.get("data", "")
    return ""

async def _send_tasks(base_url: str, text: str, session_id: str) -> str:
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "tasks/send",
        "params": {
            "id": "task-" + uuid.uuid4().hex,
            "sessionId": session_id,
            "message": {"role": "user", "parts": [{"text": text}]},
        },
    }
    data = await _post_jsonrpc(base_url, payload)
    return _extract_text_from_task(data.get("result") or {})

async def _send_message(base_url: str, text: str, session_id: str) -> str:
    print(f"DEBUG _send_message: sessionId being sent: {session_id}")
    
    # CLEAN TEXT - remove context wrapper if present
    clean_text = text
    
    # Remove context wrapper nếu có
    if "Context cuộc trò chuyện:" in text:
        lines = text.split('\n')
        in_new_section = False
        clean_lines = []
        
        for line in lines:
            if line.startswith("Yêu cầu mới:") or line.startswith("Câu hỏi:"):
                in_new_section = True
                # Bỏ prefix "Yêu cầu mới:" hoặc "Câu hỏi:"
                clean_line = line.split(":", 1)[-1].strip()
                if clean_line:
                    clean_lines.append(clean_line)
            elif in_new_section:
                clean_lines.append(line)
        
        if clean_lines:
            clean_text = '\n'.join(clean_lines).strip()
    
    # Truncate nếu vẫn quá dài (backup safety)
    MAX_MESSAGE_LENGTH = 500  # Conservative limit
    if len(clean_text) > MAX_MESSAGE_LENGTH:
        clean_text = clean_text[:MAX_MESSAGE_LENGTH] + "..."
        print(f"WARNING: Message truncated to {MAX_MESSAGE_LENGTH} chars")
    
    print(f"DEBUG: Original text length: {len(text)}")
    print(f"DEBUG: Clean text length: {len(clean_text)}")
    print(f"DEBUG: Clean text preview: {clean_text[:100]}...")

    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": clean_text}],  # CHỈ GỬI CLEAN TEXT
                "messageId": uuid.uuid4().hex,
            },
            "sessionId": session_id,
        },
    }

    print(f"DEBUG _send_message: Clean payload size: {len(json.dumps(payload))} chars")

    data = await _post_jsonrpc(base_url, payload)
    parts = (data.get("result") or {}).get("parts") or []
    if parts and isinstance(parts[0], dict):
        return parts[0].get("text", "")
    return ""
async def _call_a2a(base_url: str, query: str, session_id: str, agent_name: str = "") -> str:
    print(f"DEBUG _call_a2a: Called with session_id: {session_id}")
    print(f"DEBUG _call_a2a: Query: {query}")
    print(f"DEBUG _call_a2a: URL: {base_url}")

    try:
        # Bảo đảm sessionId được truyền đúng trong payload
        out = await _send_message(base_url, query, session_id)
        print(f"DEBUG _call_a2a: Response: {out}")
        if out.strip():
            if agent_name:
                shared_memory.add_message(agent_name, query, out)
            return out
    except Exception as e:
        print(f"DEBUG _call_a2a: message/send failed: {e}")

    # Thử tasks/send làm phương án dự phòng
    try:
        out2 = await _send_tasks(base_url, query, session_id)
        if out2.strip():
            if agent_name:
                shared_memory.add_message(agent_name, query, out2)
            return out2
    except Exception as e:
        return f"⚠️ Lỗi gọi A2A: {e}"

    return "⚠️ Không nhận được phản hồi hợp lệ từ agent A2A."

# ====== Enhanced Tools với Shared Memory ======
async def query_rooms(
    guests: int,
    check_in: Optional[str] = None,
    check_out: Optional[str] = None,
    tool_context: ToolContext = None,
) -> str:
    """Hỏi GetInfo Agent về phòng trống - CHỈ GỬI QUERY THUẦN."""
    session_id = shared_memory.get_or_create_session()
    info_url = AGENT_MAP["Ohana GetInfo Agent"]

    # Tạo question thuần túy, KHÔNG có context wrapper
    question = f"Tôi cần tìm phòng cho {guests} người"
    if check_in and check_out:
        question += f" từ ngày {check_in} đến {check_out}"
    elif check_in:
        question += f" từ ngày {check_in}"
    question += ". Bạn có thể cho tôi biết những phòng nào còn trống không?"

    print(f"DEBUG: Sending clean question: {question}")

    # GỬI MESSAGE THUẦN - Backend agent sẽ tự load context từ session
    response = await _call_a2a(info_url, question, session_id, "GetInfo Agent")

    # Lưu lại để handoff sang Booking
    shared_memory.update_booking_context({
        "last_room_query": {
            "guests": guests,
            "check_in": check_in,
            "check_out": check_out,
            "response": response,
        }
    })

    return response

async def book_room(
    room_selection: str,
    tool_context: ToolContext = None,
) -> str:
    """Đặt phòng - CHỈ GỬI ROOM SELECTION THUẦN."""
    session_id = shared_memory.get_or_create_session()
    print(f"DEBUG: Using session for book_room: {session_id}")

    booking_url = AGENT_MAP["Ohana Booking Agent"]

    # CLEAN message - chỉ gửi room selection
    clean_message = room_selection.strip()
    
    # Chuẩn hóa ngày DD/MM/YYYY → YYYY-MM-DD
    import re
    def convert_date(text: str) -> str:
        pattern = r'(\d{1,2})/(\d{1,2})/(\d{4})'
        def repl(m):
            day, month, year = m.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        return re.sub(pattern, repl, text)

    clean_message = convert_date(clean_message)
    print(f"DEBUG: Sending clean booking message: {clean_message}")

    # GỬI MESSAGE THUẦN - Backend sẽ có context từ session
    response = await _call_a2a(booking_url, clean_message, session_id, "Booking Agent")
    return response

async def confirm_booking(
    confirmation: str,
    tool_context: ToolContext = None,
) -> str:
    """Xác nhận đặt phòng - CHỈ GỬI CONFIRMATION THUẦN."""
    session_id = shared_memory.get_or_create_session()
    booking_url = AGENT_MAP["Ohana Booking Agent"]

    # CLEAN confirmation
    clean_confirmation = confirmation.strip()
    
    # Chuẩn hóa ngày DD/MM/YYYY → YYYY-MM-DD
    import re
    def convert_date(text: str) -> str:
        pattern = r'(\d{1,2})/(\d{1,2})/(\d{4})'
        def repl(m):
            day, month, year = m.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        return re.sub(pattern, repl, text)

    clean_confirmation = convert_date(clean_confirmation)
    print(f"DEBUG: Sending clean confirmation: {clean_confirmation}")

    response = await _call_a2a(booking_url, clean_confirmation, session_id, "Booking Agent")
    return response

async def ask_info_agent(
    question: str,
    tool_context: ToolContext = None,
) -> str:
    """Hỏi GetInfo Agent - CHỈ GỬI QUESTION THUẦN."""
    session_id = shared_memory.get_or_create_session()
    info_url = AGENT_MAP["Ohana GetInfo Agent"]

    # CLEAN question - không thêm context wrapper
    clean_question = question.strip()
    print(f"DEBUG: Sending clean question to InfoAgent: {clean_question}")

    response = await _call_a2a(info_url, clean_question, session_id, "GetInfo Agent")
    return response

# ====== Host Agent (ADK) ======

def build_host_agent() -> Agent:
    from datetime import datetime, timedelta

    today = datetime.now()
    weekdays = ['Thứ Hai', 'Thứ Ba', 'Thứ Tư', 'Thứ Năm', 'Thứ Sáu', 'Thứ Bảy', 'Chủ Nhật']

    # Các mốc thời gian thường dùng
    tomorrow = today + timedelta(days=1)
    day_after_tomorrow = today + timedelta(days=2)
    sunday_this_week = today + timedelta(days=(6 - today.weekday()))

    instruction = f"""
Bạn là Ohana Host Agent – trung gian điều hướng thông minh với SHARED MEMORY SYSTEM.

THÔNG TIN NGÀY HIỆN TẠI:
- Hôm nay: {today.strftime('%Y-%m-%d')} ({weekdays[today.weekday()]}) – {today.strftime('%d/%m/%Y')}
- Ngày mai: {tomorrow.strftime('%Y-%m-%d')} ({weekdays[tomorrow.weekday()]}) – {tomorrow.strftime('%d/%m/%Y')}
- Ngày mốt: {day_after_tomorrow.strftime('%Y-%m-%d')} ({weekdays[day_after_tomorrow.weekday()]}) – {day_after_tomorrow.strftime('%d/%m/%Y')}
- Chủ nhật tuần này: {sunday_this_week.strftime('%Y-%m-%d')} – {sunday_this_week.strftime('%d/%m/%Y')}

SHARED MEMORY SYSTEM:
- TẤT CẢ agents (Host, GetInfo, Booking) dùng CHUNG session và memory
- Mỗi agent có thể truy cập toàn bộ lịch sử hội thoại
- Context được tự động truyền giữa các agents
- Không hỏi lại thông tin đã có

TOOLS AVAILABLE:
1) query_rooms(guests, check_in?, check_out?) – Tìm phòng trống (context-aware)
2) book_room(room_selection) – Đặt phòng với full context từ shared memory
3) confirm_booking(confirmation) – Xác nhận đặt phòng
4) ask_info_agent(question) – Hỏi thêm dịch vụ/thông tin (context-aware)

LUỒNG XỬ LÝ THÔNG MINH:
1) Tự nhớ thông tin đã thu thập (số khách, ngày, tên, SĐT)
2) Không hỏi lại thông tin đã có trong shared memory
3) Tự động truyền context giữa các bước và agents
4) Booking Agent nhận đủ context từ GetInfo Agent

QUY TẮC NHẬN DIỆN PHÒNG (ví dụ):
- Bất kỳ token là số **3 chữ số** (100–999) đứng độc lập hoặc đi kèm từ khoá đặt phòng
  ("phòng", "đặt", "lấy", "chốt", "ok", "nhé", "đi") thì xem là **số phòng**.
  → Chuẩn hóa thành "OH" + số đó (đủ 3 chữ số). Ví dụ:
  "101", "phòng 101", "lấy 201 nhé", "404 đi" → room_id = "OH101"/"OH201"/"OH404".
- Nếu đã có prefix "OH" thì giữ nguyên (ví dụ: "OH203").
- KHÔNG coi là số phòng nếu:
  + Có đơn vị/ký hiệu ngay sau số: "k", "K", "nghìn", "tr", "%", "cm", ... (vd: "101k", "101%").
  + Rõ ràng là **số người**: theo sau bởi "người/khách" (vd: "4 người").
  + Nằm trong định dạng ngày/giờ/khoảng: có dấu "/", ":", "-" liên quan tới ngày/giờ (vd: "10/1", "10:10", "101-103").
- Nếu có **nhiều** số 3 chữ số hợp lệ, ưu tiên số gần các từ "phòng/đặt/lấy/chốt/ok".
- Không suy diễn số người từ mã phòng ("404" **không** tự hiểu là 4 người).

Ví dụ:
User: "Tôi cần phòng cho 4 người cuối tuần" → query_rooms() → Lưu context: 4 người, cuối tuần.
User: "Giờ check-in là mấy giờ?" → ask_info_agent() – trả lời có xét context đặt phòng hiện tại.
"""

    return Agent(
        model=MODEL,
        name="Ohana_Host_Permanent_Session",
        instruction=instruction,
        tools=[query_rooms, book_room, confirm_booking, ask_info_agent],
    )

# ====== Runtime Wrapper ======
class HostRuntime:
    def __init__(self):
        self.agent = build_host_agent()
        self.runner = Runner(
            app_name=self.agent.name,
            agent=self.agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )
        self.user_id = "host"

    async def _ensure_session(self, session_id: str) -> None:
        """Bảo đảm session tồn tại trước khi sử dụng."""
        try:
            sess = await self.runner.session_service.get_session(
                app_name=self.agent.name,
                user_id=self.user_id,
                session_id=session_id,
            )
            if sess is None:
                await self.runner.session_service.create_session(
                    app_name=self.agent.name,
                    user_id=self.user_id,
                    state={"session_id": session_id},
                    session_id=session_id,
                )
        except Exception as e:
            print(f"Warning: Could not ensure session {session_id}: {e}")

    async def ask(self, text: str, session_id: str = "demo") -> str:
        await self._ensure_session(session_id)

        content = gt.Content(role="user", parts=[gt.Part.from_text(text=text)])

        try:
            async for ev in self.runner.run_async(
                user_id=self.user_id, session_id=session_id, new_message=content
            ):
                if ev.is_final_response():
                    if ev.content and ev.content.parts:
                        response = "\n".join([p.text for p in ev.content.parts if getattr(p, 'text', None)])
                        # Lưu lại tương tác của Host Agent vào shared memory
                        shared_memory.add_message("Host Agent", text, response)
                        return response
        except Exception as e:
            return f"⚠️ Lỗi xử lý: {e}"

        return "⚠️ Không có phản hồi."

# ====== Demo CLI ======
if __name__ == "__main__":
    async def demo():
        rt = HostRuntime()
        print("🏨 Ohana Host Agent - Shared Memory System")
        print("Tất cả agents giờ dùng chung memory và context!")
        print("• Không cần hỏi lại thông tin đã có")
        print("• Agents tự động nhớ toàn bộ cuộc trò chuyện")
        print("• Trải nghiệm liền mạch giữa các agents")
        print("\nGõ 'quit' để thoát.")
        print("-" * 50)

        sess = shared_memory.get_or_create_session()
        print(f"Session: {sess}")

        print(f"Hôm nay: {datetime.now().strftime('%d/%m/%Y (%A) - %H:%M')}")

        while True:
            try:
                q = input("\n👤 Bạn: ").strip()
                if q.lower() in {"quit", "exit", "thoát"}:
                    print("\n🏨 Cảm ơn bạn đã sử dụng dịch vụ Ohana Hotel!")
                    break

                if not q:
                    continue

                ans = await rt.ask(q, session_id=sess)
                print(f"\n🤖 Host: {ans}")

                if shared_memory.conversation_history:
                    print(f"\n💾 Memory: {len(shared_memory.conversation_history)} interactions stored")

            except KeyboardInterrupt:
                print("\n\n👋 Tạm biệt!")
                break
            except Exception as e:
                print(f"⚠️ Lỗi: {e}")

    asyncio.run(demo())
