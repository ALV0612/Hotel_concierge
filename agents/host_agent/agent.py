import asyncio
import json
import uuid
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

TaskCallbackArg = Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent
TaskUpdateCallback = Callable[[TaskCallbackArg, AgentCard], Task]


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


# ====== Main Host Agent Class ======
class OhanaHostAgent:
    """The Ohana Hotel Host Agent - coordinates between GetInfo and Booking agents using A2A framework with unified session management."""

    def __init__(self):
        self.remote_agent_connections: Dict[str, RemoteAgentConnections] = {}
        self.cards: Dict[str, AgentCard] = {}
        self.agents: str = ""
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
            name="Ohana_Host_Agent",
            instruction=self.root_instruction,
            description="Host agent for Ohana Hotel booking system using A2A shared context.",
            tools=[
                self.send_message,  # Only tool: communicate with backend agents
            ],
        )

    def root_instruction(self, context: ReadonlyContext) -> str:
        """Generate dynamic instruction with current date and agent info."""
        today = datetime.now()
        weekdays = ['Thứ Hai', 'Thứ Ba', 'Thứ Tư', 'Thứ Năm', 'Thứ Sáu', 'Thứ Bảy', 'Chủ Nhật']
        
        tomorrow = today + timedelta(days=1)
        day_after_tomorrow = today + timedelta(days=2)
        
        return f"""
**Vai trò:** Bạn là Ohana Host Agent - trung gian điều hướng thông minh cho hệ thống đặt phòng khách sạn sử dụng ngữ cảnh chia sẻ A2A.

**Thông tin ngày hiện tại:**
- Hôm nay: {today.strftime('%Y-%m-%d')} ({weekdays[today.weekday()]}) – {today.strftime('%d/%m/%Y')}
- Ngày mai: {tomorrow.strftime('%Y-%m-%d')} ({weekdays[tomorrow.weekday()]}) – {tomorrow.strftime('%d/%m/%Y')}
- Ngày mốt: {day_after_tomorrow.strftime('%Y-%m-%d')} ({weekdays[day_after_tomorrow.weekday()]}) – {day_after_tomorrow.strftime('%d/%m/%Y')}

**Hệ thống ngữ cảnh chia sẻ A2A:**
- Tất cả agents tự động chia sẻ ngữ cảnh qua unified session management
- Backend agents có thể truy cập toàn bộ lịch sử hội thoại qua sessionId
- Không cần lặp lại thông tin - agents tự nhớ các tương tác trước đó
- Handoff liền mạch giữa GetInfo và Booking agents

**Chỉ thị cốt lõi:**
1. **Điều phối thuần túy:** Bạn chỉ điều hướng - dùng tool send_message để ủy quyền cho backend agents
2. **Lựa chọn agent:** Chọn đúng backend agent cho từng nhiệm vụ:
   - **Ohana GetInfo Agent:** Tìm phòng trống, thông tin khách sạn, chính sách, dịch vụ
   - **Ohana Booking Agent:** Đặt phòng, xác nhận, thanh toán, quản lý reservation
3. **Ủy quyền thông minh:** Gửi tin nhắn rõ ràng, có ngữ cảnh đến backend agents
4. **Nhận diện phòng:** Số 3 chữ số (100-999) với từ khóa đặt phòng = số phòng
5. **Xử lý ngày:** Nhận DD/MM/YYYY, chuyển sang YYYY-MM-DD cho backend
6. **Nhận thức ngữ cảnh:** Backend agents chia sẻ ngữ cảnh qua unified sessionId

**Quy trình đặt phòng:**
1. **Thu thập yêu cầu:** Số khách, ngày nhận/trả phòng → send_message tới Ohana GetInfo Agent
2. **Tìm phòng:** "Tìm phòng cho X khách từ ngày Y đến Z"
3. **Lựa chọn:** User chọn phòng → send_message tới Ohana Booking Agent
4. **Xác nhận:** Hoàn tất chi tiết đặt phòng với Ohana Booking Agent
5. **Câu hỏi thêm:** Gửi câu hỏi chung tới Ohana GetInfo Agent

**Ví dụ send_message:**
- Tới Ohana GetInfo Agent: "Tôi cần phòng cho 4 người từ 15/12 đến 17/12"
- Tới Ohana Booking Agent: "Đặt phòng OH101 cho khách Nguyễn Văn A, SĐT 0987654321"
- Tới Ohana GetInfo Agent: "Giờ check-in và check-out là mấy giờ?"

**Công cụ có sẵn:**
- `send_message(agent_name, task)` - Gửi nhiệm vụ tới backend agent được chỉ định

**Phong cách phản hồi:**
- Thân thiện và chuyên nghiệp
- Dùng bullet points để rõ ràng khi cần thiết
- Xác nhận các chi tiết quan trọng với user
- Hướng dẫn user qua quy trình đặt phòng một cách mượt mà

**Backend Agents có sẵn:**
{self.agents}
"""

    async def stream(self, query: str, session_id: str) -> AsyncIterable[Dict[str, Any]]:
        """Stream the agent's response to a query with unified session management."""
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
                state={"unified_session_id": session_id},  # Store unified session_id
                session_id=session_id,
            )
        else:
            # Update existing session with unified session_id
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

    # ====== Tool Method ======
    async def send_message(
        self, 
        agent_name: str, 
        task: str, 
        tool_context: ToolContext
    ) -> str:
        """Send a task to a specified backend agent using A2A with unified session management."""
        
        if agent_name not in self.remote_agent_connections:
            available_agents = list(self.remote_agent_connections.keys())
            return f"Agent '{agent_name}' not found. Available agents: {available_agents}"

        client = self.remote_agent_connections[agent_name]
        
        # CRITICAL: Use HOST AGENT's session_id for shared context
        # Priority 1: Get unified session_id from ADK session state
        session_id = None
        if tool_context and tool_context.state:
            session_id = tool_context.state.get("unified_session_id")
        
        # Priority 2: Try to get session_id from tool_context attributes
        if not session_id and tool_context and hasattr(tool_context, 'session_id'):
            session_id = tool_context.session_id
        
        # Priority 3: Get from general state
        if not session_id and tool_context and tool_context.state:
            session_id = tool_context.state.get("session_id")
        
        # Fallback: Generate unified session if none found
        if not session_id:
            session_id = f"ohana-unified-{int(datetime.now().timestamp())}"
        
        # Normalize the task message (dates, room numbers)
        clean_task = self._normalize_message(task)
        
        # Create unique message ID for A2A
        message_id = str(uuid.uuid4())
        
        # A2A message payload with unified session
        payload = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": clean_task}],
                "messageId": message_id,
            },
            # Add unified sessionId for shared context across all agents
            "sessionId": session_id
        }

        message_request = SendMessageRequest(
            id=message_id, 
            params=MessageSendParams.model_validate(payload)
        )

        try:
            send_response: SendMessageResponse = await client.send_message(message_request)
            
            # Handle different response types
            if hasattr(send_response.root, 'error'):
                error_info = send_response.root.error
                return f"Backend error: {error_info.message} (code: {error_info.code})"
            
            if not isinstance(send_response.root, SendMessageSuccessResponse):
                return f"Unexpected response from {agent_name}"
            
            # Extract response text from backend agent
            response_text = self._extract_message_text(send_response.root.result)
            return response_text

        except Exception as e:
            return f"Error communicating with {agent_name}: {e}"

    # ====== Helper Methods ======
    def _extract_message_text(self, result) -> str:
        """Extract text from A2A Message object (simplified for backend that returns Messages)."""
        try:
            # Handle Message objects (what backend returns)
            if hasattr(result, 'parts') and result.parts:
                all_texts = []
                for part in result.parts:
                    if hasattr(part, 'text') and part.text:
                        all_texts.append(part.text.strip())
                    elif hasattr(part, 'root') and hasattr(part.root, 'text') and part.root.text:
                        all_texts.append(part.root.text.strip())
                
                if all_texts:
                    return "\n".join(all_texts)
            
            # Fallback: direct text attribute
            if hasattr(result, 'text') and result.text:
                return result.text.strip()
            
            return "Request processed but no text response available"
            
        except Exception as e:
            return f"Error extracting response: {str(e)}"

    def _normalize_message(self, text: str) -> str:
        """Normalize message - convert dates and room IDs for backend compatibility."""
        import re
        
        # Convert DD/MM/YYYY to YYYY-MM-DD for backend compatibility
        def convert_date(match):
            day, month, year = match.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        
        normalized = re.sub(r'(\d{1,2})/(\d{1,2})/(\d{4})', convert_date, text)
        
        # Convert 3-digit room numbers to OH format (101 → OH101)
        def convert_room(match):
            number = match.group(1)
            if len(number) == 3 and number.isdigit() and 100 <= int(number) <= 999:
                return f"OH{number}"
            return match.group(0)
        
        # Only convert standalone 3-digit numbers or those with booking keywords
        booking_pattern = r'\b(\d{3})\b(?=\s*(?:nhé|đi|ok|phòng|room|lấy|chọn|book)?)'
        normalized = re.sub(booking_pattern, convert_room, normalized, flags=re.IGNORECASE)
        
        return normalized.strip()


# ====== Factory Function ======
def _get_initialized_ohana_host_agent_sync():
    """Synchronously creates and initializes the OhanaHostAgent."""
    
    async def _async_main():
        # Backend agent URLs for Ohana Hotel system
        backend_agent_urls = [
            "http://localhost:10002",  # GetInfo Agent
            "http://localhost:9999",   # Booking Agent
        ]
        
        print("Initializing Ohana Host Agent with A2A...")
        host_agent_instance = await OhanaHostAgent.create(
            remote_agent_addresses=backend_agent_urls
        )
        print("Ohana Host Agent initialized successfully")
        return host_agent_instance

    try:
        return asyncio.run(_async_main())
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            print("Warning: Event loop already running. Using existing loop.")
            # For Jupyter/existing event loop environments
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
        """Demo CLI for testing the Ohana Host Agent with A2A shared context."""
        print("Ohana Host Agent - A2A Unified Session Version")
        print("Features: A2A Framework, Unified Session Management, Seamless Agent Communication")
        print("Backend agents share conversation context automatically via unified sessionId")
        print("-" * 80)
        
        # Initialize host agent
        try:
            host_agent = await OhanaHostAgent.create([
                "http://localhost:10002",  # GetInfo Agent
                "http://localhost:9999",   # Booking Agent
            ])
        except Exception as e:
            print(f"Failed to initialize: {e}")
            return
        
        # Generate unified session ID
        session_id = f"ohana-unified-{int(datetime.now().timestamp())}"
        print(f"Unified Session ID: {session_id}")
        print(f"Today: {datetime.now().strftime('%d/%m/%Y (%A) - %H:%M')}")
        print("\nCommands: 'quit' to exit, 'new' for new session")
        print("-" * 80)
        
        while True:
            try:
                query = input("\nYou: ").strip()
                
                if query.lower() in {"quit", "exit", "bye", "thoát"}:
                    print("\nThank you for using Ohana Hotel! Have a great day!")
                    break
                
                if query.lower() in {"new", "reset", "restart"}:
                    session_id = f"ohana-unified-{int(datetime.now().timestamp())}"
                    print(f"New unified session created: {session_id}")
                    continue
                
                if not query:
                    continue
                
                print("\nHost Agent: ", end="")
                response_complete = False
                
                async for response in host_agent.stream(query, session_id):
                    if response["is_task_complete"]:
                        print(response["content"])
                        response_complete = True
                    else:
                        print(".", end="", flush=True)
                
                if not response_complete:
                    print("\nNo response received")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! Thanks for using Ohana Hotel!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again or type 'quit' to exit.")

    # Run the demo
    print("Starting Ohana Hotel Demo...")
    asyncio.run(demo())