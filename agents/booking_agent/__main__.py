import os
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from agent_executor import BookingAgentExecutor
from dotenv import load_dotenv

load_dotenv()

def main():
    # Dùng env hoặc mặc định 9999
    try:
        PORT = int(os.getenv("BOOKING_PORT", "9999"))
    except ValueError:
        PORT = 9999
    if not (0 < PORT < 65536):
        raise ValueError(f"Invalid port {PORT}. Use 1-65535.")

    skills = [
        AgentSkill(
            id="ohana.booking",
            name="Ohana: Hotel Booking",
            description="Đặt phòng khách sạn Ohana - thu thập thông tin và xác nhận booking qua MCP",
            tags=["booking", "hotel", "reservation", "mcp", "gemini", "langgraph"],
            examples=[
                "Đặt phòng OH103 cho 2 người ngày mai",
                "Tôi muốn đặt phòng từ 10/09 đến 12/09",
                "Booking phòng Standard cho gia đình 4 người",
                "Đặt phòng cuối tuần, tên Nguyễn Văn A, SĐT 0901234567"
            ],
        )
    ]

    card = AgentCard(
        name="Ohana Booking Agent",
        description="Chuyên viên đặt phòng khách sạn Ohana - LangGraph workflow với Gemini 2.5 Flash và MCP integration",
        url=f"http://localhost:{PORT}/",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=skills,
        version="2.0.0",
        capabilities=AgentCapabilities(
            streaming=False, 
            stateTransitionHistory=True
        ),
    )

    handler = DefaultRequestHandler(
        agent_executor=BookingAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        http_handler=handler,
        agent_card=card,
    )

    print("\n" + "="*60)
    print("🏨 Ohana Hotel Booking Agent (A2A)")
    print("="*60)
    print(f"📍 Server: http://localhost:{PORT}/")
    print(f"🤖 Model: LangGraph + Gemini 2.5 Flash")
    print(f"💾 Storage: MCP + SQLite")
    print(f"🔗 A2A Card: {card.name}")
    print("="*60)
    print("Ready for AgentCard integration!")
    print("Press Ctrl+C to stop\n")

    uvicorn.run(app.build(), host="0.0.0.0", port=PORT)

if __name__ == "__main__":
    main()