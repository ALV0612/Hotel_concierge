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
BOOKING_URL = os.getenv("BOOKING_AGENT_URL", "http://localhost:9999")
INFO_URL    = os.getenv("INFO_AGENT_URL",    "http://localhost:10002")

AGENT_MAP = {
    "Ohana Booking Agent": BOOKING_URL,
    "Ohana GetInfo Agent": INFO_URL,
}

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ====== Shared Memory System ======
# ====== Shared Memory System với Dynamic Sessions ======
class SharedMemoryService:
    def __init__(self):
        self.conversation_history = []
        self.user_preferences = {}
        self.booking_context = {}
        self.current_session = None
        self.current_booking_id = None
        self.last_activity = None
        self.session_start_time = None
        
    def should_reset_session(self, message: str) -> bool:
        """Detect nếu cần reset session dựa trên message hoặc timeout"""
        if not message:
            return False
            
        msg_lower = message.lower().strip()
        
        # Reset signals - explicit commands
        reset_signals = [
            "xin chào", "chào", "hello", "hi", "chao",
            "tôi muốn đặt phòng mới", "đặt phòng mới",
            "booking mới", "new booking",
            "bắt đầu lại", "start over", "restart",
            "session mới", "conversation mới", "chat mới",
            "reset", "new", "làm lại"
        ]
        
        # Check exact match hoặc start of sentence
        for signal in reset_signals:
            if msg_lower == signal or msg_lower.startswith(signal + " "):
                return True
        
        # Check conversation starters (typical greeting patterns)
        starter_patterns = [
            r'^(xin\s+)?chào\b',
            r'^hello\b', 
            r'^hi\b',
            r'^tôi\s+muốn\s+đặt\s+phòng',
            r'^đặt\s+phòng',
            r'^book\s+room'
        ]
        
        import re
        for pattern in starter_patterns:
            if re.match(pattern, msg_lower):
                # Only reset if we have existing conversation
                if len(self.conversation_history) > 0:
                    return True
        
        # Check timeout - reset sau 1 giờ không hoạt động
        if self.last_activity:
            import time
            if time.time() - self.last_activity > 3600:  # 1 hour
                return True
        
        return False
        
    def get_or_create_session(self, message: str = ""):
        """Tạo session mới hoặc dùng existing, với auto-reset detection"""
        import time
        current_time = time.time()
        
        # Check if should reset based on message or timeout
        if message and self.should_reset_session(message):
            if self.current_session:
                print(f"🔄 Auto-detected session reset signal in: '{message[:30]}...'")
                return self.start_new_conversation()
        
        # Create first session nếu chưa có
        if not self.current_session:
            timestamp = int(datetime.now().timestamp())
            self.current_session = f"ohana-session-{timestamp}"
            self.session_start_time = current_time
            print(f"✨ Created new session: {self.current_session}")
        
        # Update last activity
        self.last_activity = current_time
        return self.current_session
    
    def start_new_conversation(self):
        """Bắt đầu conversation mới với session mới"""
        import time
        timestamp = int(datetime.now().timestamp())
        old_session = self.current_session
        old_history_count = len(self.conversation_history)
        
        # Create new session
        self.current_session = f"ohana-session-{timestamp}"
        self.session_start_time = time.time()
        self.last_activity = time.time()
        
        # Clear context cho conversation mới
        self.conversation_history = []
        self.booking_context = {}
        self.current_booking_id = None
        
        print(f"🆕 New conversation started!")
        print(f"   Previous: {old_session} ({old_history_count} messages)")
        print(f"   Current:  {self.current_session}")
        
        return self.current_session
    
    def manual_reset(self):
        """Reset thủ công - để dùng trong CLI commands"""
        return self.start_new_conversation()
    
    def add_message(self, agent: str, user_message: str, agent_response: str):
        """Thêm message vào lịch sử conversation"""
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "user_message": user_message,
            "agent_response": agent_response,
            "booking_id": self.current_booking_id,
            "session_id": self.current_session
        })
        print(f"📝 Added to memory - {agent}: {user_message[:50]}...")
        
        # Update activity
        import time
        self.last_activity = time.time()
    
    def get_recent_context(self, last_n: int = 3):
        """Lấy n messages gần nhất trong session hiện tại"""
        recent = self.conversation_history[-last_n:] if self.conversation_history else []
        return recent
    
    def update_booking_context(self, context):
        """Cập nhật context cho booking hiện tại"""
        self.booking_context.update(context)
        print(f"📋 Updated booking context: {list(context.keys())}")
    
    def get_full_context_summary(self):
        """Lấy tóm tắt context đầy đủ của cuộc trò chuyện hiện tại"""
        recent = self.get_recent_context(5)
        summary = ""
        
        if recent:
            summary = "Lịch sử cuộc trò chuyện gần đây:\n"
            for msg in recent:
                user_msg = msg['user_message'][:50] + "..." if len(msg['user_message']) > 50 else msg['user_message']
                agent_resp = msg['agent_response'][:100] + "..." if len(msg['agent_response']) > 100 else msg['agent_response']
                summary += f"- {msg['agent']}: {user_msg} → {agent_resp}\n"
        
        if self.booking_context:
            summary += f"\nThông tin đặt phòng hiện tại: {json.dumps(self.booking_context, ensure_ascii=False)}\n"
        
        return summary
    
    def start_new_booking(self):
        """Bắt đầu booking mới trong session hiện tại"""
        self.current_booking_id = f"booking-{uuid.uuid4().hex[:8]}"
        self.booking_context = {"booking_id": self.current_booking_id}
        print(f"🏨 Started new booking {self.current_booking_id} in session {self.current_session}")
        return self.current_booking_id
    
    def complete_booking(self):
        """Hoàn thành booking hiện tại, giữ session"""
        if self.current_booking_id:
            completed_id = self.current_booking_id
            print(f"✅ Completed booking {completed_id}")
            self.current_booking_id = None
            self.booking_context = {}
            return completed_id
        return None
    
    def mark_booking_complete(self):
        """Đánh dấu booking đã hoàn thành"""
        completed = self.complete_booking()
        if completed:
            print(f"🎉 Booking {completed} marked as complete")
    
    def get_session_stats(self):
        """Lấy thống kê session hiện tại"""
        import time
        stats = {
            "session_id": self.current_session,
            "message_count": len(self.conversation_history),
            "current_booking": self.current_booking_id,
            "context_items": len(self.booking_context),
            "uptime_minutes": round((time.time() - self.session_start_time) / 60, 1) if self.session_start_time else 0
        }
        return stats
    
    def reset_session(self):
        """Legacy method - calls start_new_conversation"""
        return self.start_new_conversation()

# Global shared memory instance
shared_memory = SharedMemoryService()
# Global shared memory instance

# ====== A2A helpers ======
async def _post_jsonrpc(base_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=30) as cli:
        r = await cli.post(base_url, json=payload)
        r.raise_for_status()
        return r.json()

def _extract_text_from_task(result: Dict[str, Any]) -> str:
    # Try status.message.parts
    status = result.get("status") or {}
    msg = status.get("message") or {}
    parts = msg.get("parts") or []
    if parts and isinstance(parts[0], dict):
        return parts[0].get("text") or parts[0].get("data") or ""
    # Fallback to result.message
    msg2 = (result.get("message") or {})
    parts2 = msg2.get("parts") or []
    if parts2 and isinstance(parts2[0], dict):
        return parts2[0].get("text") or parts2[0].get("data") or ""
    # Fallback artifacts
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
    
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": text}],
                "messageId": uuid.uuid4().hex,
            },
            "sessionId": session_id,  # QUAN TRỌNG: đảm bảo key đúng
        },
    }
    
    print(f"DEBUG _send_message: Full payload: {json.dumps(payload, indent=2)}")
    
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
        # Đảm bảo session_id được truyền trong payload
        out = await _send_message(base_url, query, session_id)
        print(f"DEBUG _call_a2a: Response: {out}")
        if out.strip():
            if agent_name:
                shared_memory.add_message(agent_name, query, out)
            return out
    except Exception as e:
        print(f"DEBUG _call_a2a: message/send failed: {e}")
    
    # Try tasks/send as fallback
    try:
        out2 = await _send_tasks(base_url, query, session_id)
        if out2.strip():
            if agent_name:
                shared_memory.add_message(agent_name, query, out2)
            return out2
    except Exception as e:
        return f"⛔ Lỗi gọi A2A: {e}"
    
    return "⚠️ Không nhận được phản hồi hợp lệ từ agent A2A."

# ====== Enhanced Tools với Shared Memory ======
async def query_rooms(
    guests: int,
    check_in: Optional[str] = None,
    check_out: Optional[str] = None,
    tool_context: ToolContext = None,
) -> str:
    """
    Hỏi GetInfo Agent về phòng trống với shared memory context.
    """
    session_id = shared_memory.get_or_create_session()
    info_url = AGENT_MAP["Ohana GetInfo Agent"]

    # Tạo câu hỏi với context nếu có
    context_summary = shared_memory.get_full_context_summary()
    
    question = f"Tôi cần tìm phòng cho {guests} người"
    if check_in and check_out:
        question += f" từ ngày {check_in} đến {check_out}"
    elif check_in:
        question += f" từ ngày {check_in}"
    question += ". Bạn có thể cho tôi biết những phòng nào còn trống không?"
    
    # Nếu có context, thêm vào
    if context_summary.strip():
        question = f"Context cuộc trò chuyện:\n{context_summary}\n\nYêu cầu mới: {question}"

    response = await _call_a2a(info_url, question, session_id, "GetInfo Agent")
    
    # Lưu context cho handoff
    shared_memory.update_booking_context({
        "last_room_query": {
            "guests": guests,
            "check_in": check_in,
            "check_out": check_out,
            "response": response
        }
    })
    
    return response

async def book_room(
    room_selection: str,
    tool_context: ToolContext = None,
) -> str:
    session_id = shared_memory.get_or_create_session()  # FIX: Sử dụng FIXED_SESSION thay vì hardcoded "test-fixed-session"
    print(f"DEBUG: Using FIXED session for book_room: {session_id}")  # FIX: Log để check
    
    booking_url = AGENT_MAP["Ohana Booking Agent"]
    
    # FIX: Thêm context summary vào message để subagent parse dễ hơn (handle_context_rich)
    context_summary = shared_memory.get_full_context_summary()
    message = room_selection
    if context_summary.strip():
        message = f"Context cuộc trò chuyện:\n{context_summary}\n\n{room_selection}"
    
    # FIX: Convert date format (tương tự confirm_booking) để đồng bộ với subagent parser
    import re
    def convert_date(text):
        pattern = r'(\d{1,2})/(\d{1,2})/(\d{4})'
        def repl(m):
            day, month, year = m.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        return re.sub(pattern, repl, text)
    message = convert_date(message)
    
    response = await _call_a2a(booking_url, message, session_id, "Booking Agent")
    
    return response

async def confirm_booking(
    confirmation: str,
    tool_context: ToolContext = None,
) -> str:
    session_id = shared_memory.get_or_create_session()
    booking_url = AGENT_MAP["Ohana Booking Agent"]
    
    # Convert date format từ DD/MM/YYYY sang YYYY-MM-DD
    import re
    
    # Tìm và convert dates
    def convert_date(text):
        # Pattern cho DD/MM/YYYY
        pattern = r'(\d{1,2})/(\d{1,2})/(\d{4})'
        def repl(m):
            day, month, year = m.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        return re.sub(pattern, repl, text)
    
    confirmation = convert_date(confirmation)
    
    print(f"DEBUG: Converted confirmation: {confirmation}")
    
    response = await _call_a2a(booking_url, confirmation, session_id, "Booking Agent")
    
    return response

async def ask_info_agent(
    question: str,
    tool_context: ToolContext = None,
) -> str:
    """
    Hỏi GetInfo Agent với shared context.
    """
    session_id = shared_memory.get_or_create_session()
    info_url = AGENT_MAP["Ohana GetInfo Agent"]
    
    # Thêm context nếu liên quan đến booking hiện tại
    context_summary = shared_memory.get_full_context_summary()
    if shared_memory.booking_context and any(word in question.lower() for word in ["phòng", "đặt", "booking", "check"]):
        enhanced_question = f"Context cuộc trò chuyện:\n{context_summary}\n\nCâu hỏi: {question}"
    else:
        enhanced_question = question
    
    response = await _call_a2a(info_url, enhanced_question, session_id, "GetInfo Agent")
    return response
from datetime import date

# ====== Host Agent (ADK) ======
def build_host_agent() -> Agent:
    # ... existing date calculations ...
     # Get current date info
    from datetime import datetime, timedelta

    today = datetime.now()
    weekdays = ['Thứ Hai', 'Thứ Ba', 'Thứ Tư', 'Thứ Năm', 'Thứ Sáu', 'Thứ Bảy', 'Chủ Nhật']
    
    # Calculate common relative dates
    tomorrow = today + timedelta(days=1)
    day_after_tomorrow = today + timedelta(days=2)
    sunday_this_week = today + timedelta(days=(6-today.weekday()))
    instruction = f"""
    Bạn là Ohana Host Agent - trung gian điều hướng thông minh với SHARED MEMORY SYSTEM.
    
    THÔNG TIN NGÀY HIỆN TẠI:
    - Hôm nay: {today.strftime('%Y-%m-%d')} ({weekdays[today.weekday()]}) - {today.strftime('%d/%m/%Y')}
    - Ngày mai: {tomorrow.strftime('%Y-%m-%d')} ({weekdays[tomorrow.weekday()]}) - {tomorrow.strftime('%d/%m/%Y')}
    - Ngày mốt: {day_after_tomorrow.strftime('%Y-%m-%d')} ({weekdays[day_after_tomorrow.weekday()]}) - {day_after_tomorrow.strftime('%d/%m/%Y')}
    - Chủ nhật tuần này: {sunday_this_week.strftime('%Y-%m-%d')} - {sunday_this_week.strftime('%d/%m/%Y')}
    
    SHARED MEMORY SYSTEM:
    - TẤT CẢ agents (Host, GetInfo, Booking) dùng CHUNG session và memory
    - Mỗi agent có thể truy cập toàn bộ lịch sử cuộc trò chuyện
    - Context được tự động truyền giữa các agents
    - Không cần thu thập lại thông tin đã có
    
    CÁC TOOLS AVAILABLE:
    1. `query_rooms(guests, check_in?, check_out?)` - Tìm phòng trống (có context awareness)
    2. `book_room(room_selection)` - Đặt phòng với full context từ shared memory
    3. `confirm_booking(confirmation)` - Xác nhận đặt phòng 
    4. `ask_info_agent(question)` - Hỏi về dịch vụ với context awareness
    
    LUỒNG XỬ LÝ THÔNG MINH:
    1. Agents tự động nhớ thông tin đã thu thập (số khách, ngày, tên, sđt)
    2. Không hỏi lại thông tin đã có trong shared memory
    3. Tự động điền context cho agents khác
    4. Booking Agent nhận đầy đủ context từ GetInfo Agent
    
    PHONG CÁCH:
    - Tận dụng shared memory để tạo trải nghiệm seamless
    - Không lặp lại câu hỏi về thông tin đã có
    - Tự động reference thông tin từ cuộc trò chuyện trước
    - Thông minh trong việc kết nối các steps
    
    VÍ DỤ SMART FLOW:
    User: "Tôi cần phòng cho 4 người cuối tuần"
    → query_rooms() → Lưu context: 4 người, cuối tuần
    
    1) Bất kỳ token là số **3 chữ số** (100–999) đứng độc lập hoặc đi kèm các từ đặt phòng
   (vd: "phòng", "đặt", "lấy", "chốt", "ok", "nhé", "đi") thì coi là **số phòng**.
   → Chuẩn hoá thành "OH" + số đó (có đủ 3 chữ số).
   Ví dụ:
    - "101", "phòng 101", "lấy 201 nhé", "404 đi" → room_id = "OH101"/"OH201"/"OH404".
    2) Nếu đã có prefix "OH" thì giữ nguyên (vd: "OH203" → "OH203").
    3) **Không** coi là số phòng nếu:
    - Có đơn vị tiền/khác ngay sau số: "k", "K", "nghìn", "tr", "%", "cm"… (vd: "101k", "101%").
    - Rõ ràng là **số người**: theo sau bởi "người/khách" (vd: "4 người").
    - Nằm trong định dạng ngày/giờ/phạm vi: có dấu "/" ":" "-" liên quan tới ngày/giờ (vd: "10/1", "10:10", "101-103").
    4) Nếu có **nhiều** số 3 chữ số hợp lệ, ưu tiên số gần các từ "phòng/đặt/lấy/chốt/ok".
    5) Không suy diễn số người từ mã phòng; "404" **không** tự nghĩa là 4 người.
        
    User: "Giờ check-in là mấy giờ?"
    → ask_info_agent() → Có context đang đặt OH404, trả lời chính xác
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
    
    async def _ensure_session(self, session_id: str):
        """Đảm bảo session tồn tại trước khi sử dụng"""
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
                        response = "\n".join([p.text for p in ev.content.parts if p.text])
                        # Add Host Agent interaction to shared memory
                        shared_memory.add_message("Host Agent", text, response)
                        return response
        except Exception as e:
            return f"⛔ Lỗi xử lý: {e}"
            
        return "⚠️ Không có phản hồi."

# ====== Demo CLI ======
if __name__ == "__main__":
    async def demo():
        rt = HostRuntime()
        print("🏨 Ohana Host Agent - Shared Memory System")
        print("Tất cả agents giờ có chung memory và context!")
        print("• Không cần hỏi lại thông tin đã có")
        print("• Agents tự động nhớ toàn bộ cuộc trò chuyện")
        print("• Trải nghiệm seamless giữa các agents")
        print("\nType 'quit' to exit.")
        print("-" * 50)
        
        sess = shared_memory.get_or_create_session()  # FIX: Sử dụng FIXED_SESSION thay vì random
        print(f"Session: {sess}")
        
        # Show current time
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
                
                # Show shared memory status
                if shared_memory.conversation_history:
                    print(f"\n💭 Memory: {len(shared_memory.conversation_history)} interactions stored")
                
            except KeyboardInterrupt:
                print("\n\n👋 Tạm biệt!")
                break
            except Exception as e:
                print(f"⛔ Lỗi: {e}")

    asyncio.run(demo())