import os, sys, time, subprocess
from multiprocessing import Process
import uvicorn
from dotenv import load_dotenv

load_dotenv()

def start_booking_agent():
    # Đảm bảo sub-agent nghe trên localhost:9999 (nội bộ)
    env = os.environ.copy()
    env["PORT"] = "9999"  # nếu __main__ của agent đọc PORT
    # Hoặc nếu agent đọc biến riêng thì đặt BOOKING_AGENT_PORT=...
    subprocess.Popen(
        [sys.executable, "-u", "-X", "utf8", "-m", "agents.booking_agent"],
        env=env,
        cwd=os.getcwd(),
    )

def start_info_agent():
    env = os.environ.copy()
    env["PORT"] = "10002"
    subprocess.Popen(
        [sys.executable, "-u", "-X", "utf8", "-m", "agents.get_info_agent"],
        env=env,
        cwd=os.getcwd(),
    )

def start_host_agent():
    from railwaymain import app
    port = int(os.getenv("PORT", "8000"))  # PORT do Railway cấp
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)

def wait_for_agents_ready():
    import requests
    for name, url in [("Booking", "http://127.0.0.1:9999"),
                      ("GetInfo", "http://127.0.0.1:10002")]:
        for i in range(30):
            try:
                r = requests.get(f"{url}/health", timeout=2)
                if r.status_code == 200:
                    print(f"✅ {name} ready at {url}")
                    break
            except Exception:
                print(f"⏳ Waiting for {name}... ({i+1}/30)")
                time.sleep(1)
        else:
            print(f"⚠️ {name} not ready, continuing...")

if __name__ == "__main__":
    print("🚀 Starting Ohana Multi-Agent System on Railway...")
    # Host public URL sẽ là $PORT; sub-agents chỉ nội bộ
    os.environ["BOOKING_AGENT_URL"] = "http://127.0.0.1:9999"
    os.environ["INFO_AGENT_URL"]    = "http://127.0.0.1:10002"

    # chạy sub-agents
    Process(target=start_booking_agent, daemon=True).start()
    Process(target=start_info_agent, daemon=True).start()

    wait_for_agents_ready()
    print("🎯 Starting Host Agent...")
    start_host_agent()
