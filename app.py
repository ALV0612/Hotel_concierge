import os
import asyncio
import threading
import time
from multiprocessing import Process
import uvicorn
from dotenv import load_dotenv

load_dotenv()

def start_booking_agent():
    """Start Booking Agent process"""
    try:
        print("🏨 Starting Booking Agent...")
        os.chdir("agents/booking_agent")  # Change to booking agent directory
        
        # Import and run booking agent
        from __main__ import main as booking_main
        booking_main()
        
    except Exception as e:
        print(f"❌ Error starting Booking Agent: {e}")

def start_info_agent():
    """Start GetInfo Agent process"""
    try:
        print("ℹ️ Starting GetInfo Agent...")
        os.chdir("agents/getinfo_agent")  # Change to getinfo agent directory
        
        # Import and run info agent
        from __main__ import main as info_main
        info_main()
        
    except Exception as e:
        print(f"❌ Error starting GetInfo Agent: {e}")

def start_host_agent():
    """Start Host Agent (main FastAPI app)"""
    try:
        print("🤖 Starting Host Agent (Main App)...")
        
        # Import main FastAPI app
        from railwaymain import app
        
        port = int(os.getenv("PORT", 8000))
        uvicorn.run(app, host="0.0.0.0", port=port)
        
    except Exception as e:
        print(f"❌ Error starting Host Agent: {e}")

def wait_for_agents_ready():
    """Wait for sub agents to be ready before starting host"""
    import requests
    
    booking_url = os.getenv("BOOKING_AGENT_URL", "http://localhost:9999")
    info_url = os.getenv("INFO_AGENT_URL", "http://localhost:10002")
    
    max_attempts = 30  # 30 seconds timeout
    
    for agent_name, url in [("Booking", booking_url), ("GetInfo", info_url)]:
        for attempt in range(max_attempts):
            try:
                response = requests.get(f"{url}/health", timeout=2)
                if response.status_code == 200:
                    print(f"✅ {agent_name} Agent ready at {url}")
                    break
            except:
                if attempt < max_attempts - 1:
                    print(f"⏳ Waiting for {agent_name} Agent... ({attempt+1}/{max_attempts})")
                    time.sleep(1)
                else:
                    print(f"⚠️ {agent_name} Agent not ready, but continuing...")

if __name__ == "__main__":
    print("🚀 Starting Ohana Multi-Agent System on Railway...")
    
    # Set environment variables for internal communication
    os.environ["BOOKING_AGENT_URL"] = "http://localhost:9999"
    os.environ["INFO_AGENT_URL"] = "http://localhost:10002"
    
    # Start sub agents in separate processes
    booking_process = Process(target=start_booking_agent, daemon=True)
    info_process = Process(target=start_info_agent, daemon=True)
    
    print("🔄 Starting sub agents...")
    booking_process.start()
    info_process.start()
    
    # Wait for sub agents to be ready
    wait_for_agents_ready()
    
    # Start main host agent
    print("🎯 Starting Host Agent...")
    start_host_agent()