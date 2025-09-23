# agents/host_agent/agent_with_shared_memory.py
# Host Agent with Shared Memory Integration

import asyncio
import json
import uuid
import re
from datetime import datetime, timedelta
from typing import Any, AsyncIterable, Dict, List, Optional, Callable

import httpx
import nest_asyncio
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
)
from dotenv import load_dotenv
from google.adk import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types

load_dotenv()
nest_asyncio.apply()

# ====== Shared Memory System ======
class SharedMemory:
    """Thread-safe shared memory system for multi-agent communication"""
    
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
    
    def set_context(self, session_id: str, key: str, value: Any):
        """Set context for a session"""
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = {}
            self._store[session_id][key] = value
            self._store[session_id]['last_updated'] = datetime.now().isoformat()
    
    def update_booking_info(self, session_id: str, info: Dict[str, Any]):
        """Update booking information incrementally"""
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = {}
            
            current_booking = self._store[session_id].get('booking_info', {})
            current_booking.update(info)
            self._store[session_id]['booking_info'] = current_booking
            self._store[session_id]['last_updated'] = datetime.now().isoformat()
    
    def get_booking_info(self, session_id: str) -> Dict[str, Any]:
        """Get complete booking information"""
        with self._lock:
            return self._store.get(session_id, {}).get('booking_info', {})
    
    def export_for_booking_agent(self, session_id: str) -> str:
        """Export context specifically formatted for booking agent"""
        booking_info = self.get_booking_info(session_id)
        if not booking_info:
            return ""
        # Chỉ giữ các key mà Booking Agent biết parse
        keys = ["room_id", "check_in", "check_out", "guest_name", "phone_number", "guests"]
        ctx = {k: booking_info[k] for k in keys if k in booking_info and booking_info[k]}

        # JSON compact nhất để tránh vượt limit
        import json as _json
        compact = _json.dumps(ctx, ensure_ascii=False, separators=(",", ":"))

        # 1 câu context ngắn (regex-friendly) đề phòng JSON bị lỗi hiếm hoi
        line = []
        if all(k in ctx for k in ("room_id","check_in","check_out")):
            line.append(f"Context: Đặt {ctx['room_id']} từ {ctx['check_in']} đến {ctx['check_out']}.")
        if "guest_name" in ctx and "phone_number" in ctx:
            line.append(f"Tên {ctx['guest_name']}, SĐT {ctx['phone_number']}.")
        if "guests" in ctx:
            line.append(f"Cho {ctx['guests']} người.")

        # Trả về: JSON + 1 dòng context, để ĐẦU message khi gửi đi
        return compact + ("\n" + " ".join(line) if line else "")


# Global shared memory instance
import threading
shared_memory = SharedMemory()

# ====== Information Extractor ======
class BookingInfoExtractor:
    """Extract booking information from conversation - Fixed version"""
    
    @staticmethod
    def extract_from_message(message: str) -> Dict[str, Any]:
        """Extract booking info from any message - improved name extraction"""
        info = {}
        
        # Extract room ID
        room_match = re.search(r'\b(OH\d{3})\b', message, re.IGNORECASE)
        if room_match:
            info['room_id'] = room_match.group(1).upper()
        
        # Extract dates (YYYY-MM-DD format)
        date_pattern = r'(\d{4}-\d{2}-\d{2})'
        dates = re.findall(date_pattern, message)
        if len(dates) >= 2:
            info['check_in'] = dates[0]
            info['check_out'] = dates[1]
        
        # Extract guest count
        guest_match = re.search(r'(\d+)\s+(?:người|khách)', message, re.IGNORECASE)
        if guest_match:
            info['guests'] = int(guest_match.group(1))
        
        # IMPROVED: Extract name and phone - multiple patterns with priority
        # Pattern 1: Name,Phone (highest priority for comma-separated format)
        comma_pattern = r'([A-ZÀ-Ỹ][a-zA-ZÀ-ỹ\s]+?),\s*(0\d{9,10})'
        comma_match = re.search(comma_pattern, message)
        if comma_match:
            name_candidate = comma_match.group(1).strip()
            # Validate Vietnamese name (at least 2 words, no digits)
            if len(name_candidate.split()) >= 2 and not re.search(r'\d', name_candidate):
                info['guest_name'] = name_candidate
                info['phone_number'] = comma_match.group(2).strip()
        
        # Pattern 2: Name + space + Phone (if comma pattern didn't work)
        if 'guest_name' not in info:
            space_pattern = r'([A-ZÀ-Ỹ][a-zA-ZÀ-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zA-ZÀ-ỹ]+)+)\s+(0\d{9,10})'
            space_match = re.search(space_pattern, message)
            if space_match:
                name_candidate = space_match.group(1).strip()
                if len(name_candidate.split()) >= 2 and not re.search(r'\d', name_candidate):
                    info['guest_name'] = name_candidate
                    info['phone_number'] = space_match.group(2).strip()
        
        # Pattern 3: Try to find name separately if not found above
        if 'guest_name' not in info:
            # Look for Vietnamese name patterns
            name_patterns = [
                r'[Tt]ên\s+(?:khách\s+)?(?:là\s+)?([A-ZÀ-Ỹ][a-zA-ZÀ-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zA-ZÀ-ỹ]+)+)',
                r'[Tt]ôi\s+là\s+([A-ZÀ-Ỹ][a-zA-ZÀ-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zA-ZÀ-ỹ]+)+)',
                # r'([A-ZÀ-Ỹ][a-zA-ZÀ-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zA-ZÀ-ỹ]+){1,3})(?=\s*[,.\n]|$)',
                # Special pattern for "Name nha" format
                r'([A-ZÀ-Ỹ][a-zA-ZÀ-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zA-ZÀ-ỹ]+)+)\s+nha?'
            ]
            for pattern in name_patterns:
                match = re.search(pattern, message, re.IGNORECASE)
                if match:
                    candidate = match.group(1).strip()
                    # Validate Vietnamese name
                    if (len(candidate.split()) >= 2 and 
                        not re.search(r'\d', candidate) and
                        len(candidate) > 3):  # Minimum length check
                        info['guest_name'] = candidate
                        break
        
        # Pattern 4: Find phone separately if not found
        if 'phone_number' not in info:
            phone_match = re.search(r'\b(0\d{9,10})\b', message)
            if phone_match:
                info['phone_number'] = phone_match.group(1)
        
        return info

# ====== Remote Agent Connection Class ======
class RemoteAgentConnections:
    """A class to hold the connections to the remote agents."""
    
    def __init__(self, agent_card: AgentCard, agent_url: str):
        print(f"Connecting to agent: {agent_card.name}")
        print(f"URL: {agent_url}")
        self._httpx_client = httpx.AsyncClient(timeout=60)
        self.agent_client = A2AClient(self._httpx_client, agent_card, url=agent_url)
        self.card = agent_card
        self.conversation_name = None
        self.conversation = None
        self.pending_tasks = set()

    def get_agent(self) -> AgentCard:
        return self.card

    async def send_message(
        self, message_request: SendMessageRequest
    ) -> SendMessageResponse:
        return await self.agent_client.send_message(message_request)

# ====== Enhanced Host Agent Class ======
class OhanaHostAgent:
    """Enhanced Ohana Hotel Host Agent with shared memory integration."""

    def __init__(self):
        self.remote_agent_connections: Dict[str, RemoteAgentConnections] = {}
        self.cards: Dict[str, AgentCard] = {}
        self.agents: str = ""
        self.shared_memory = shared_memory
        self.extractor = BookingInfoExtractor()
        self._agent = self.create_agent()
        self._user_id = "ohana_host"
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    async def _async_init_components(self, remote_agent_addresses: List[str]):
        """Initialize connections to remote agents using A2A framework."""
        async with httpx.AsyncClient(timeout=60) as client:
            for address in remote_agent_addresses:
                card_resolver = A2ACardResolver(client, address)
                try:
                    card = await card_resolver.get_agent_card()
                    remote_connection = RemoteAgentConnections(
                        agent_card=card, agent_url=address
                    )
                    self.remote_agent_connections[card.name] = remote_connection
                    self.cards[card.name] = card
                    print(f"Successfully connected to {card.name}")
                except httpx.ConnectError as e:
                    print(f"Failed to connect to {address}: {e}")
                except Exception as e:
                    print(f"Failed to initialize {address}: {e}")

        # Build agent info string for instruction
        agent_info = [
            json.dumps({"name": card.name, "description": card.description})
            for card in self.cards.values()
        ]
        self.agents = "\n".join(agent_info) if agent_info else "No backend agents found"
        print(f"Available agents: {len(self.cards)}")

    @classmethod
    async def create(cls, remote_agent_addresses: List[str]):
        """Factory method to create and initialize OhanaHostAgent."""
        instance = cls()
        await instance._async_init_components(remote_agent_addresses)
        return instance

    def create_agent(self) -> Agent:
        """Create the ADK Agent with tools."""
        return Agent(
            model="gemini-2.5-flash",
            name="Ohana_Host_Agent_Enhanced",
            instruction=self.root_instruction,
            description="Enhanced host agent with shared memory for Ohana Hotel booking system.",
            tools=[
                self.send_message,
            ],
        )

    def root_instruction(self, context: ReadonlyContext) -> str:
        """Generate dynamic instruction with enhanced date handling."""
        today = datetime.now()
        weekdays = ['Thứ Hai', 'Thứ Ba', 'Thứ Tư', 'Thứ Năm', 'Thứ Sáu', 'Thứ Bảy', 'Chủ Nhật']
        
        # Calculate this weekend and next weekend
        current_weekday = today.weekday()  # 0=Monday, 6=Sunday
        days_to_saturday = (5 - current_weekday) % 7
        days_to_sunday = (6 - current_weekday) % 7
        
        this_saturday = today + timedelta(days=days_to_saturday)
        this_sunday = today + timedelta(days=days_to_sunday)
        
        next_saturday = this_saturday + timedelta(days=7)
        next_sunday = this_sunday + timedelta(days=7)
        
        return f"""
        **Vai trò:** Bạn là Ohana Assistance - với shared memory system để ghi nhớ thông tin qua session.

        **THÔNG TIN THỜI GIAN:**
        - Ngày hiện tại: {today.strftime('%Y-%m-%d')} ({weekdays[today.weekday()]})
        - Cuối tuần này: {this_saturday.strftime('%Y-%m-%d')} đến {this_sunday.strftime('%Y-%m-%d')}
        - Cuối tuần sau: {next_saturday.strftime('%Y-%m-%d')} đến {next_sunday.strftime('%Y-%m-%d')}

        **XỬ LÝ THỜI GIAN THÔNG MINH:**
        - "cuối tuần này" = {this_saturday.strftime('%Y-%m-%d')} đến {this_sunday.strftime('%Y-%m-%d')}
        - "cuối tuần sau/tới" = {next_saturday.strftime('%Y-%m-%d')} đến {next_sunday.strftime('%Y-%m-%d')}
        - "tuần sau" = từ {(today + timedelta(days=7)).strftime('%Y-%m-%d')} trở đi
        - "hôm nay" = {today.strftime('%Y-%m-%d')}
        - "ngày mai" = {(today + timedelta(days=1)).strftime('%Y-%m-%d')}

        **SHARED MEMORY SYSTEM:**
        - Hệ thống tự động lưu trữ và chia sẻ thông tin booking giữa các agents
        - Thông tin được tích lũy qua cuộc trò chuyện: phòng, ngày, tên khách, SĐT
        - Khi gửi đến Booking Agent, tự động kèm theo context đã thu thập
        - **QUAN TRỌNG:** Luôn chuyển đổi thời gian tương đối thành ngày cụ thể (YYYY-MM-DD)

        **QUY TRÌNH THÔNG MINH:**
        1. **GetInfo Agent:** Tìm kiếm phòng, thông tin khách sạn
        2. **Shared Memory:** Tự động lưu thông tin phòng, ngày được chọn (đã convert sang YYYY-MM-DD)
        3. **Booking Agent:** Nhận context đầy đủ với ngày cụ thể để đặt phòng

        **NHIỆM VỤ CỐT LÕI:**
        - Phân tích user input để trích xuất thông tin booking
        - **CHUYỂN ĐỔI thời gian tương đối thành ngày cụ thể**
        - Chọn agent phù hợp cho từng task
        - Tự động enrichment context khi gửi đến Booking Agent
        - Đảm bảo thông tin được truyền đạt đầy đủ và chính xác

        **Backend Agents:**
        {self.agents}

        **EXAMPLE FLOWS:**
        User: "Tôi cần phòng cho 2 người cuối tuần này"
        → Convert: "cuối tuần này" = {this_saturday.strftime('%Y-%m-%d')} đến {this_sunday.strftime('%Y-%m-%d')}
        → Send to GetInfo Agent: "Tìm phòng cho 2 người từ {this_saturday.strftime('%Y-%m-%d')} đến {this_sunday.strftime('%Y-%m-%d')}"

        User: "Tôi cần phòng cuối tuần sau"  
        → Convert: "cuối tuần sau" = {next_saturday.strftime('%Y-%m-%d')} đến {next_sunday.strftime('%Y-%m-%d')}
        → Send to GetInfo Agent: "Tìm phòng từ {next_saturday.strftime('%Y-%m-%d')} đến {next_sunday.strftime('%Y-%m-%d')}"

        User: "Tôi chọn phòng 102" 
        → Store room_id in memory
        → Send to GetInfo Agent: "Thông tin chi tiết phòng OH102"

        User: "Đặt phòng đi, tên tôi là Khánh Hòa, 0937401803"
        → Store guest info in memory  
        → Send to Booking Agent with FULL CONTEXT: "Context: Khách đã chọn phòng OH102 từ [specific_date] đến [specific_date]. Tên khách Khánh Hòa, số điện thoại 0937401803. Xác nhận đặt phòng."

        **LƯU Ý QUAN TRỌNG:**
        - LUÔN chuyển đổi "cuối tuần này/sau", "tuần tới" thành ngày cụ thể
        - Không để thời gian tương đối trong memory hoặc gửi đến backend agents
        - Xác nhận lại với user về ngày đã convert: "Bạn muốn đặt từ [date] đến [date], đúng không?"
        - Trước khi booking, phải cho người dùng xác nhận lại thông tin có đúng hay không.
        """
    async def stream(self, query: str, session_id: str) -> AsyncIterable[Dict[str, Any]]:
        """Stream the agent's response with memory integration."""
        # Extract and store booking info from user query
        booking_info = self.extractor.extract_from_message(query)
        if booking_info:
            self.shared_memory.update_booking_info(session_id, booking_info)
        
        # Ensure session exists in ADK
        session = await self._runner.session_service.get_session(
            app_name=self._agent.name,
            user_id=self._user_id,
            session_id=session_id,
        )
        
        if session is None:
            session = await self._runner.session_service.create_session(
                app_name=self._agent.name,
                user_id=self._user_id,
                state={"unified_session_id": session_id},
                session_id=session_id,
            )
        else:
            session.state = session.state or {}
            session.state["unified_session_id"] = session_id

        content = types.Content(role="user", parts=[types.Part.from_text(text=query)])
        
        async for event in self._runner.run_async(
            user_id=self._user_id, session_id=session.id, new_message=content
        ):
            if event.is_final_response():
                response = ""
                if (
                    event.content
                    and event.content.parts
                    and event.content.parts[0].text
                ):
                    response = "\n".join([p.text for p in event.content.parts if p.text])
                
                yield {
                    "is_task_complete": True,
                    "content": response,
                }
            else:
                yield {
                    "is_task_complete": False,
                    "updates": "Host agent is processing your request...",
                }

    async def send_message(
        self, 
        agent_name: str, 
        task: str, 
        tool_context: ToolContext
    ) -> str:
        """Enhanced send_message with shared memory integration."""
        
        if agent_name not in self.remote_agent_connections:
            available_agents = list(self.remote_agent_connections.keys())
            return f"Agent '{agent_name}' not found. Available agents: {available_agents}"

        client = self.remote_agent_connections[agent_name]
        
        # Get session ID
        session_id = None
        if tool_context and tool_context.state:
            session_id = tool_context.state.get("unified_session_id")
        
        if not session_id:
            session_id = f"ohana-unified-{int(datetime.now().timestamp())}"
        
        # Extract booking info from current task
        task_booking_info = self.extractor.extract_from_message(task)
        if task_booking_info:
            self.shared_memory.update_booking_info(session_id, task_booking_info)
        
        # Enhance task with context for Booking Agent
        enhanced_task = task
        if "Booking Agent" in agent_name:
            context = self.shared_memory.export_for_booking_agent(session_id)
            if context:
                enhanced_task = f"{context}. {task}"
                print(f"[DEBUG] Enhanced task for Booking Agent: {enhanced_task}")
        
        # Normalize the task message
        clean_task = self._normalize_message(enhanced_task)
        
        # Create A2A message
        message_id = str(uuid.uuid4())
        payload = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": clean_task}],
                "messageId": message_id,
            },
            "sessionId": session_id
        }

        message_request = SendMessageRequest(
            id=message_id, 
            params=MessageSendParams.model_validate(payload)
        )

        try:
            send_response: SendMessageResponse = await client.send_message(message_request)
            
            if hasattr(send_response.root, 'error'):
                error_info = send_response.root.error
                return f"Backend error: {error_info.message} (code: {error_info.code})"
            
            if not isinstance(send_response.root, SendMessageSuccessResponse):
                return f"Unexpected response from {agent_name}"
            
            response_text = self._extract_message_text(send_response.root.result)
            
            # Extract booking info from response
            response_booking_info = self.extractor.extract_from_message(response_text)
            if response_booking_info:
                self.shared_memory.update_booking_info(session_id, response_booking_info)
            
            return response_text

        except Exception as e:
            return f"Error communicating with {agent_name}: {e}"

    def _extract_message_text(self, result) -> str:
        """Extract text from A2A Message object."""
        try:
            if hasattr(result, 'parts') and result.parts:
                all_texts = []
                for part in result.parts:
                    if hasattr(part, 'text') and part.text:
                        all_texts.append(part.text.strip())
                    elif hasattr(part, 'root') and hasattr(part.root, 'text') and part.root.text:
                        all_texts.append(part.root.text.strip())
                
                if all_texts:
                    return "\n".join(all_texts)
            
            if hasattr(result, 'text') and result.text:
                return result.text.strip()
            
            return "Request processed but no text response available"
            
        except Exception as e:
            return f"Error extracting response: {str(e)}"

    def _normalize_message(self, text: str) -> str:
        """Normalize message - convert dates and room IDs."""
        import re
        
        # Convert DD/MM/YYYY to YYYY-MM-DD
        def convert_date(match):
            day, month, year = match.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        
        normalized = re.sub(r'(\d{1,2})/(\d{1,2})/(\d{4})', convert_date, text)
        
        # Convert 3-digit room numbers to OH format
        def convert_room(match):
            number = match.group(1)
            if len(number) == 3 and number.isdigit() and 100 <= int(number) <= 999:
                return f"OH{number}"
            return match.group(0)
        
        booking_pattern = r'\b(\d{3})\b(?=\s*(?:nhé|đi|ok|phòng|room|lấy|chọn|book)?)'
        normalized = re.sub(booking_pattern, convert_room, normalized, flags=re.IGNORECASE)
        
        return normalized.strip()


# ====== Factory Function ======
def _get_initialized_ohana_host_agent_sync():
    """Synchronously creates and initializes the enhanced OhanaHostAgent."""
    
    async def _async_main():
        backend_agent_urls = [
            "http://localhost:10002",  # GetInfo Agent
            "http://localhost:9999",   # Booking Agent
        ]
        
        print("Initializing Enhanced Ohana Host Agent with Shared Memory...")
        host_agent_instance = await OhanaHostAgent.create(
            remote_agent_addresses=backend_agent_urls
        )
        print("Enhanced Ohana Host Agent initialized successfully")
        return host_agent_instance

    try:
        return asyncio.run(_async_main())
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            print("Warning: Event loop already running. Using existing loop.")
            try:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(_async_main())
            except Exception as inner_e:
                print(f"Failed to initialize with existing loop: {inner_e}")
                raise
        else:
            raise

# Create the root agent for export
root_agent = _get_initialized_ohana_host_agent_sync()

# ====== Main Demo ======
if __name__ == "__main__":
    async def demo():
        """Demo CLI with shared memory."""
        print("Ohana Host Agent Enhanced - Shared Memory Version")
        print("Features: Shared Memory, Context Accumulation, Smart Agent Routing")
        print("-" * 80)
        
        try:
            host_agent = await OhanaHostAgent.create([
                "http://localhost:10002",  # GetInfo Agent
                "http://localhost:9999",   # Booking Agent
            ])
        except Exception as e:
            print(f"Failed to initialize: {e}")
            return
        
        session_id = f"ohana-unified-{int(datetime.now().timestamp())}"
        print(f"Session ID: {session_id}")
        print(f"Today: {datetime.now().strftime('%d/%m/%Y (%A) - %H:%M')}")
        print("\nCommands: 'quit' to exit, 'new' for new session, 'memory' to view stored info")
        print("-" * 80)
        
        while True:
            try:
                query = input("\nYou: ").strip()
                
                if query.lower() in {"quit", "exit", "bye", "thoát"}:
                    print("\nThank you for using Ohana Hotel! Have a great day!")
                    break
                
                if query.lower() in {"new", "reset", "restart"}:
                    session_id = f"ohana-unified-{int(datetime.now().timestamp())}"
                    print(f"New session created: {session_id}")
                    continue
                
                if query.lower() == "memory":
                    booking_info = shared_memory.get_booking_info(session_id)
                    print(f"Stored booking info: {json.dumps(booking_info, ensure_ascii=False, indent=2)}")
                    continue
                
                if not query:
                    continue
                
                print("\nHost Agent: ", end="")
                
                async for response in host_agent.stream(query, session_id):
                    if response["is_task_complete"]:
                        print(response["content"])
                    else:
                        print(".", end="", flush=True)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")

    print("Starting Enhanced Ohana Hotel Demo...")
    asyncio.run(demo())