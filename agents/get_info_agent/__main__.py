import os
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from agents.get_info_agent.agent_executor import GetInfoAgentExecutor
from dotenv import load_dotenv
load_dotenv()
def main():
    # Dùng env hoặc mặc định 10001
    try:
        PORT = int(os.getenv("INFO_PORT", "10002"))
    except ValueError:
        PORT = 10002
    if not (0 < PORT < 65536):
        raise ValueError(f"Invalid port {PORT}. Use 1-65535.")

    skills = [
        AgentSkill(
            id="ohana.getinfo",
            name="Ohana: Booking + Docs",
            description="Đặt phòng; hỏi nội quy/dịch vụ (RAG) qua MCP",
            tags=["booking","hotel","rag","mcp","gemini"],
            examples=[
                "Đặt OH203 ngày mai cho 2 người",
                "Nội quy hút thuốc là gì?",
                "Check-out mấy giờ?"
            ],
        )
    ]

    card = AgentCard(
        name="Ohana GetInfo Agent",
        description="Concierge: booking + policy/docs via RAG (Gemini 2.5 + MCP)",
        url=f"http://localhost:{PORT}/",              # 🔹 khớp đúng với cổng server
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

    uvicorn.run(app.build(), host="0.0.0.0", port=PORT)

if __name__ == "__main__":
    main()
