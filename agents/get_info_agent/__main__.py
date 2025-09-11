import os
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from agents.get_info_agent.agent_executor import GetInfoAgentExecutor
from dotenv import load_dotenv

load_dotenv()

def pick_port() -> int:
    """
    - Khi chạy dưới Host: dùng INFO_PORT (mặc định 10002), KHÔNG dùng $PORT.
    - Khi chạy độc lập (STANDALONE=1): cho phép dùng $PORT để test ngoài.
    """
    if os.getenv("STANDALONE", "0") == "1":
        p = int(os.getenv("PORT", "10002"))
    else:
        p = int(os.getenv("INFO_PORT", "10002"))

    if not (0 < p < 65536):
        raise ValueError(f"Invalid port {p}. Use 1–65535.")

    # Tránh xung đột với Host khi không ở chế độ standalone
    if os.getenv("STANDALONE", "0") != "1" and str(p) == os.getenv("PORT", ""):
        raise RuntimeError(f"INFO_PORT ({p}) trùng với $PORT của Host. Hãy đặt INFO_PORT khác.")

    return p

def main():
    PORT = pick_port()
    BIND = os.getenv("INFO_BIND", "localhost")  # đổi từ 127.0.0.1 thành localhost
    # URL hiển thị trên AgentCard (đổi từ 127.0.0.1 thành localhost)
    card_url = os.getenv("INFO_CARD_URL", f"http://localhost:{PORT}/")

    skills = [
        AgentSkill(
            id="ohana.getinfo",
            name="Ohana: Booking + Docs",
            description="Đặt phòng; hỏi nội quy/dịch vụ (RAG) qua MCP",
            tags=["booking", "hotel", "rag", "mcp", "gemini"],
            examples=[
                "Đặt OH203 ngày mai cho 2 người",
                "Nội quy hút thuốc là gì?",
                "Check-out mấy giờ?",
            ],
        )
    ]

    card = AgentCard(
        name="Ohana GetInfo Agent",
        description="Concierge: booking + policy/docs via RAG (Gemini 2.5 + MCP)",
        url=card_url,
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=skills,
        version="2.0.0",
        capabilities=AgentCapabilities(streaming=False, stateTransitionHistory=True),
    )

    handler = DefaultRequestHandler(
        agent_executor=GetInfoAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        http_handler=handler,
        agent_card=card,
    )

    print("\n" + "=" * 60)
    print("  Ohana GetInfo Agent (A2A)")
    print("=" * 60)
    print(f"📍 Server:   http://{BIND}:{PORT}/")
    print(f"🔗 Card URL: {card_url}")
    print("Ready for AgentCard integration!")
    print("=" * 60 + "\n")

    uvicorn.run(app.build(), host=BIND, port=PORT)

if __name__ == "__main__":
    main()