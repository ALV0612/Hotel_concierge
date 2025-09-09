import os
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from agents.booking_agent.agent_executor import BookingAgentExecutor
from dotenv import load_dotenv

load_dotenv()

def pick_port() -> int:
    # Chạy độc lập (để test/local): cho phép dùng $PORT nếu muốn
    if os.getenv("STANDALONE", "0") == "1":
        p = int(os.getenv("PORT", "9999"))
    else:
        # Chạy dưới Host: dùng BOOKING_PORT (mặc định 9999)
        p = int(os.getenv("BOOKING_PORT", "9999"))
    if not (0 < p < 65536):
        raise ValueError(f"Invalid port {p}. Use 1–65535.")
    # Tránh đụng $PORT của platform (cổng public của Host)
    if str(p) == os.getenv("PORT", "") and os.getenv("STANDALONE", "0") != "1":
        raise RuntimeError(f"BOOKING_PORT ({p}) trùng với $PORT của Host. Đổi BOOKING_PORT khác.")
    return p

def main():
    PORT = pick_port()
    BIND = os.getenv("BOOKING_BIND", "127.0.0.1")  # bind loopback để nội bộ thôi

    skills = [AgentSkill(
        id="ohana.booking",
        name="Ohana: Hotel Booking",
        description="Đặt phòng khách sạn Ohana",
        tags=["booking","hotel","reservation","mcp","langgraph"],
        examples=[
            "Đặt phòng OH103 cho 2 người ngày mai",
            "Tôi muốn đặt phòng từ 10/09 đến 12/09",
        ],
    )]

    card_url = os.getenv("BOOKING_CARD_URL", f"http://127.0.0.1:{PORT}/")  # tránh localhost
    card = AgentCard(
        name="Ohana Booking Agent",
        description="Booking workflow (LangGraph + Gemini)",
        url=card_url,
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=skills,
        version="2.0.0",
        capabilities=AgentCapabilities(streaming=False, stateTransitionHistory=True),
    )

    handler = DefaultRequestHandler(
        agent_executor=BookingAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(http_handler=handler, agent_card=card)

    print("\n" + "="*60)
    print("🏨 Ohana Hotel Booking Agent (A2A)")
    print("="*60)
    print(f"📍 Server: http://{BIND}:{PORT}/")
    print(f"🔗 AgentCard URL: {card_url}")
    print("="*60)

    uvicorn.run(app.build(), host=BIND, port=PORT)

if __name__ == "__main__":
    main()
