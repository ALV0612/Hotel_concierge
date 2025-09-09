import os, sys, time, subprocess
from multiprocessing import Process
import uvicorn
from dotenv import load_dotenv

load_dotenv()

def start_booking_agent():
    # Không cần env PORT nữa vì không phải web service
    subprocess.Popen(
        [sys.executable, "-u", "-X", "utf8", "-m", "agents.booking_agent"],
        cwd=os.getcwd(),
    )

def start_info_agent():
    # Không cần env PORT nữa
    subprocess.Popen(
        [sys.executable, "-u", "-X", "utf8", "-m", "agents.get_info_agent"],
        cwd=os.getcwd(),
    )

def start_host_agent():
    from railwaymain import app
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)

def wait_for_agents_ready():
    import socket
    
    def check_port(host, port, timeout=2):
        """Check if port is open using socket connection"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    # Wait for agents to be ready by checking if ports are listening
    for name, port in [("Booking", 9999), ("GetInfo", 10002)]:
        for i in range(30):
            if check_port("127.0.0.1", port):
                print(f"✅ {name} ready on port {port}")
                break
            print(f"⏳ Waiting for {name}... ({i+1}/30)")
            time.sleep(1)
        else:
            print(f"⚠️ {name} not ready, continuing...")

# Alternative: Skip health checks entirely and just wait
def simple_wait():
    print("⏳ Giving agents time to start...")
    time.sleep(5)  # Simple wait instead of health checks
    print("✅ Proceeding with host agent startup")

if __name__ == "__main__":
    print("🚀 Starting Ohana Multi-Agent System on Railway...")
    
    os.environ["BOOKING_AGENT_URL"] = "http://127.0.0.1:9999"
    os.environ["INFO_AGENT_URL"] = "http://127.0.0.1:10002"

    # Start sub-agents
    Process(target=start_booking_agent, daemon=True).start()
    Process(target=start_info_agent, daemon=True).start()

    # Choose one of these approaches:
    # wait_for_agents_ready()  # Port-based check
    simple_wait()              # Simple time-based wait
    
    print("🎯 Starting Host Agent...")
    start_host_agent()