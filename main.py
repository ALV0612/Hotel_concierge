import os
import uvicorn
import subprocess
import time
import signal
import atexit
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from typing import Dict, Any, Optional, List
import json
from datetime import datetime
import requests
from dotenv import load_dotenv

# Host agent
from agents.host_agent.agent import HostRuntime, shared_memory

load_dotenv()

class MCPServerManager:
    """Quản lý MCP servers"""
    
    def __init__(self):
        self.mcp_servers = []
        
    def start_mcp_server(self, script_path: str, server_name: str):
        """Start một MCP server"""
        try:
            print(f"🔌 Starting MCP server: {server_name}")
            
            # Get directory of script for proper working directory
            script_dir = os.path.dirname(script_path)
            script_name = os.path.basename(script_path)
            
            # Set up environment with UTF-8 support
            env = os.environ.copy()
            env['PYTHONPATH'] = os.getcwd() + os.pathsep + env.get('PYTHONPATH', '')
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONUTF8'] = '1'
            
            process = subprocess.Popen([
                "python", "-X", "utf8", script_name
            ], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            env=env,
            cwd=script_dir if script_dir else os.getcwd(),
            encoding='utf-8',
            errors='replace'
            )
            
            self.mcp_servers.append({
                "name": server_name,
                "process": process,
                "script_path": script_path,
                "pid": process.pid
            })
            
            print(f"✅ MCP {server_name} started with PID: {process.pid}")
            return True
            
        except Exception as e:
            print(f"❌ Error starting MCP {server_name}: {e}")
            return False
    
    def start_all_mcp_servers(self):
        """Start tất cả MCP servers"""
        print("🔌 Starting MCP servers...")
        
        mcp_configs = [
            {"script": "mcp_server/server_booking_mcp.py", "name": "Booking MCP"},
            {"script": "mcp_server/server_info_mcp.py", "name": "Info MCP"}
        ]
        
        started_count = 0
        for config in mcp_configs:
            if os.path.exists(config["script"]):
                if self.start_mcp_server(config["script"], config["name"]):
                    started_count += 1
            else:
                print(f"⚠️ MCP script not found: {config['script']}")
        
        # Đợi MCP servers khởi động với better health check
        if started_count > 0:
            print(f"⏳ Waiting for {started_count} MCP servers to initialize...")
            time.sleep(5)  # Tăng wait time để handle Unicode init
            
            # Check if processes are still running after init
            running_count = 0
            for server in self.mcp_servers:
                if server["process"].poll() is None:
                    running_count += 1
                    print(f"✅ MCP {server['name']} running successfully")
                else:
                    print(f"⚠️ MCP {server['name']} exited during initialization")
            
            print(f"✅ {running_count}/{started_count} MCP servers are ready")
        
        return started_count > 0
    
    def cleanup_mcp_servers(self):
        """Cleanup MCP servers khi thoát"""
        if self.mcp_servers:
            print("\n🧹 Cleaning up MCP servers...")
            
            for server in self.mcp_servers:
                try:
                    server["process"].terminate()
                    server["process"].wait(timeout=5)
                    print(f"✅ MCP {server['name']} stopped")
                except:
                    try:
                        server["process"].kill()
                        print(f"🔪 MCP {server['name']} killed")
                    except:
                        pass
    
    def get_mcp_status(self):
        """Lấy status của MCP servers"""
        status = {}
        for server in self.mcp_servers:
            if server["process"].poll() is None:  # Still running
                status[server["name"]] = {
                    "status": "running",
                    "pid": server["pid"],
                    "script": server["script_path"]
                }
            else:
                status[server["name"]] = {
                    "status": "stopped",
                    "pid": server["pid"],
                    "script": server["script_path"]
                }
        return status

class SubprocessA2AServers:
    """Quản lý A2A agent servers"""
    
    def __init__(self):
        self.booking_port = 9999
        self.info_port = 10002
        self.booking_process = None
        self.info_process = None
        self.servers_ready = False

    def start_booking_agent(self):
        """Chạy Booking Agent như subprocess"""
        try:
            print(f"🤖 Starting Booking Agent subprocess on port {self.booking_port}")
            
            env = os.environ.copy()
            env["BOOKING_PORT"] = str(self.booking_port)
            
            self.booking_process = subprocess.Popen([
                "python", "-m", "agents.booking_agent.__main__"
            ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            print(f"✅ Booking Agent subprocess started with PID: {self.booking_process.pid}")
            return True
            
        except Exception as e:
            print(f"❌ Error starting Booking Agent subprocess: {e}")
            return False

    def start_info_agent(self):
        """Chạy Info Agent như subprocess"""
        try:
            print(f"🤖 Starting Info Agent subprocess on port {self.info_port}")
            
            env = os.environ.copy()
            env["INFO_PORT"] = str(self.info_port)
            
            self.info_process = subprocess.Popen([
                "python", "-m", "agents.get_info_agent.__main__"
            ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            print(f"✅ Info Agent subprocess started with PID: {self.info_process.pid}")
            return True
            
        except Exception as e:
            print(f"❌ Error starting Info Agent subprocess: {e}")
            return False

    def check_server_health(self, url, timeout=3):
        """Kiểm tra server health"""
        try:
            endpoints = ["/", "/.well-known/agent"]
            for endpoint in endpoints:
                try:
                    resp = requests.get(f"{url}{endpoint}", timeout=timeout)
                    if resp.status_code in [200, 405, 404]:  # Server responding
                        return True
                except:
                    continue
            return False
        except Exception:
            return False

    def wait_for_servers(self, max_wait=60):
        """Đợi A2A servers sẵn sàng"""
        print(f"⏳ Waiting for A2A subprocess servers to be ready (max {max_wait}s)...")
        start_time = time.time()
        
        booking_ready = False
        info_ready = False
        
        while time.time() - start_time < max_wait:
            if not booking_ready:
                booking_ready = self.check_server_health(f"http://localhost:{self.booking_port}")
                if booking_ready:
                    print("✅ Booking Agent subprocess ready")
            
            if not info_ready:
                info_ready = self.check_server_health(f"http://localhost:{self.info_port}")
                if info_ready:
                    print("✅ Info Agent subprocess ready")
            
            if booking_ready and info_ready:
                self.servers_ready = True
                print("✅ All A2A subprocess servers are ready!")
                return True
            
            time.sleep(2)
        
        print("⚠️ Warning: Some A2A subprocess servers may not be ready yet")
        return booking_ready or info_ready

    def start_all(self):
        """Khởi động tất cả A2A subprocess servers"""
        print("🤖 Starting A2A subprocess servers...")
        
        booking_started = self.start_booking_agent()
        info_started = self.start_info_agent()
        
        if not (booking_started and info_started):
            print("❌ Failed to start some A2A subprocess servers")
            return False
        
        return self.wait_for_servers()

    def cleanup(self):
        """Cleanup A2A subprocess khi thoát"""
        print("\n🧹 Cleaning up A2A subprocess servers...")
        
        if self.booking_process:
            try:
                self.booking_process.terminate()
                self.booking_process.wait(timeout=5)
                print("✅ Booking Agent subprocess stopped")
            except:
                try:
                    self.booking_process.kill()
                    print("🔪 Booking Agent subprocess killed")
                except:
                    pass
        
        if self.info_process:
            try:
                self.info_process.terminate()
                self.info_process.wait(timeout=5)
                print("✅ Info Agent subprocess stopped")
            except:
                try:
                    self.info_process.kill()
                    print("🔪 Info Agent subprocess killed")
                except:
                    pass

    def get_status(self):
        """Lấy status của A2A subprocess"""
        booking_status = "stopped"
        info_status = "stopped"
        
        if self.booking_process:
            if self.booking_process.poll() is None:
                booking_status = "running" if self.check_server_health(f"http://localhost:{self.booking_port}") else "starting"
            else:
                booking_status = "crashed"
        
        if self.info_process:
            if self.info_process.poll() is None:
                info_status = "running" if self.check_server_health(f"http://localhost:{self.info_port}") else "starting"
            else:
                info_status = "crashed"
        
        return {
            "booking_agent": booking_status,
            "info_agent": info_status,
            "booking_pid": self.booking_process.pid if self.booking_process else None,
            "info_pid": self.info_process.pid if self.info_process else None
        }

# Global instances
mcp_manager = None
subprocess_servers = None
host_runtime = None
user_sessions = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI"""
    # Startup
    global mcp_manager, subprocess_servers, host_runtime
    
    print("🚀 Starting Ohana Facebook Bot with MCP + A2A Architecture...")
    print("="*70)
    
    # Step 1: Start MCP servers first
    print("STEP 1: Starting MCP servers...")
    mcp_manager = MCPServerManager()
    mcp_started = mcp_manager.start_all_mcp_servers()
    
    if mcp_started:
        print("✅ MCP servers started successfully")
    else:
        print("⚠️ No MCP servers found or failed to start")
    
    # Step 2: Start A2A subprocess servers
    print("\nSTEP 2: Starting A2A subprocess servers...")
    subprocess_servers = SubprocessA2AServers()
    a2a_ready = subprocess_servers.start_all()
    
    if a2a_ready:
        print("✅ A2A subprocess servers ready")
    else:
        print("⚠️ Some A2A subprocess servers may not be ready")
    
    # Step 3: Initialize host runtime
    print("\nSTEP 3: Initializing Host Agent runtime...")
    os.environ["BOOKING_AGENT_URL"] = f"http://localhost:{subprocess_servers.booking_port}"
    os.environ["INFO_AGENT_URL"] = f"http://localhost:{subprocess_servers.info_port}"
    
    host_runtime = HostRuntime()
    print("✅ Host Agent runtime initialized")
    
    # Register cleanup handlers
    atexit.register(cleanup_all_services)
    
    print("\n" + "="*70)
    print("🎯 OHANA FACEBOOK BOT - FULLY OPERATIONAL")
    print("="*70)
    print(f"📡 Main API: http://localhost:8000 (exposed)")
    print(f"🔌 MCP Servers: {len(mcp_manager.mcp_servers)} running")
    print(f"🤖 A2A Agents: booking@{subprocess_servers.booking_port}, info@{subprocess_servers.info_port}")
    print(f"🧠 Host Agent: Connected with shared memory")
    print(f"📱 Facebook Webhook: Ready for integration")
    print("="*70)
    
    yield
    
    # Shutdown
    print("\n🧹 Shutting down all services...")
    cleanup_all_services()

# FastAPI app with lifespan
app = FastAPI(
    title="Ohana Facebook Bot - MCP + A2A", 
    version="1.0.0",
    lifespan=lifespan
)

# Facebook configuration
VERIFY_TOKEN = os.getenv("FB_VERIFY_TOKEN", "ohana_verify_token_2025")
PAGE_ACCESS_TOKEN = os.getenv("FB_PAGE_ACCESS_TOKEN", "")

def get_user_session(fb_user_id: str) -> str:
    """Tạo hoặc lấy session cho Facebook user"""
    if fb_user_id not in user_sessions:
        user_sessions[fb_user_id] = f"fb-user-{fb_user_id}"
    return user_sessions[fb_user_id]

def cleanup_all_services():
    """Cleanup tất cả services khi thoát"""
    global mcp_manager, subprocess_servers
    
    if mcp_manager:
        mcp_manager.cleanup_mcp_servers()
    
    if subprocess_servers:
        subprocess_servers.cleanup()

@app.get("/")
async def root():
    """API status endpoint"""
    return {
        "message": "Ohana Facebook Bot - MCP + A2A Architecture", 
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "architecture": "mcp_servers + subprocess_a2a_servers",
        "exposed_port": "8000_only",
        "components": {
            "mcp_servers": len(mcp_manager.mcp_servers) if mcp_manager else 0,
            "a2a_agents": {
                "booking": f"localhost:{subprocess_servers.booking_port}" if subprocess_servers else "not_started",
                "info": f"localhost:{subprocess_servers.info_port}" if subprocess_servers else "not_started"
            }
        },
        "agent_cards": "preserved",
        "protocol": "A2A_compliant"
    }

@app.get("/webhook")
async def verify_webhook(request: Request):
    """Facebook webhook verification"""
    try:
        verify_token = request.query_params.get('hub.verify_token')
        challenge = request.query_params.get('hub.challenge')
        
        if verify_token == VERIFY_TOKEN:
            print("✅ Webhook verification successful!")
            return int(challenge)
        else:
            print("❌ Invalid verification token")
            raise HTTPException(status_code=403, detail="Invalid verify token")
            
    except Exception as e:
        print(f"❌ Webhook verification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook")
async def receive_message(request: Request):
    """Nhận và xử lý tin nhắn từ Facebook Messenger"""
    try:
        data = await request.json()
        print(f"📨 Received Facebook webhook")
        
        for entry in data.get('entry', []):
            for messaging in entry.get('messaging', []):
                await process_messaging_event(messaging)
        
        return {"status": "EVENT_RECEIVED"}
        
    except Exception as e:
        print(f"❌ Error processing webhook: {e}")
        return {"status": "ERROR", "message": str(e)}

async def process_messaging_event(messaging: Dict[str, Any]):
    """Xử lý messaging event từ Facebook"""
    try:
        sender_id = messaging.get('sender', {}).get('id')
        
        if not sender_id:
            return
        
        if 'message' in messaging:
            message = messaging['message']
            message_text = message.get('text', '').strip()
            
            if message_text:
                print(f"👤 User {sender_id}: {message_text}")
                response = await process_with_host_agent(message_text, sender_id)
                
                if response:
                    await send_facebook_message(sender_id, response)
                else:
                    await send_facebook_message(sender_id, "Xin lỗi, tôi không hiểu yêu cầu của bạn.")
        
        elif 'postback' in messaging:
            postback = messaging['postback']
            payload = postback.get('payload', '')
            print(f"🔘 User {sender_id} clicked: {payload}")
            
            response = await process_with_host_agent(payload, sender_id)
            if response:
                await send_facebook_message(sender_id, response)
        
    except Exception as e:
        print(f"❌ Error processing messaging event: {e}")

async def process_with_host_agent(message: str, fb_user_id: str) -> Optional[str]:
    """Xử lý tin nhắn qua Host Agent"""
    try:
        session_id = get_user_session(fb_user_id)
        
        # Tắt debug logs để clean output
        # print(f"🤖 Processing via Host Agent:")
        # print(f"   Message: {message}")
        # print(f"   Session: {session_id}")
        
        shared_memory.get_or_create_session(message)
        response = await host_runtime.ask(message, session_id=session_id)
        
        # Chỉ in response cuối cùng
        print(f"📤 Final Response: {response}")
        return response
        
    except Exception as e:
        print(f"❌ Error calling Host Agent: {e}")
        return f"Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau ít phút."

async def send_facebook_message(recipient_id: str, message_text: str):
    """Gửi tin nhắn về Facebook Messenger"""
    if not PAGE_ACCESS_TOKEN:
        print("❌ Cannot send message: PAGE_ACCESS_TOKEN not configured")
        return
    
    url = "https://graph.facebook.com/v18.0/me/messages"
    
    # Chia tin nhắn dài
    max_length = 1900
    messages = []
    
    if len(message_text) <= max_length:
        messages = [message_text]
    else:
        lines = message_text.split('\n')
        current_msg = ""
        
        for line in lines:
            if len(current_msg + line + '\n') <= max_length:
                current_msg += line + '\n'
            else:
                if current_msg:
                    messages.append(current_msg.strip())
                current_msg = line + '\n'
        
        if current_msg:
            messages.append(current_msg.strip())
    
    for i, msg in enumerate(messages):
        try:
            payload = {
                "recipient": {"id": recipient_id},
                "message": {"text": msg}
            }
            
            headers = {
                "Authorization": f"Bearer {PAGE_ACCESS_TOKEN}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 200:
                print(f"📤 Sent message {i+1}/{len(messages)} to Facebook")
            else:
                print(f"❌ Facebook API error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error sending message {i+1} to Facebook: {e}")

@app.get("/health")
async def health_check():
    """Health check với thông tin chi tiết tất cả components"""
    try:
        # MCP servers status
        mcp_status = mcp_manager.get_mcp_status() if mcp_manager else {}
        
        # A2A subprocess status
        a2a_status = subprocess_servers.get_status() if subprocess_servers else {}
        
        # Host agent status
        host_status = "connected" if host_runtime else "not_initialized"
        
        # Shared memory status
        memory_stats = shared_memory.get_session_stats() if shared_memory.current_session else {}
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "architecture": "mcp_servers + subprocess_a2a_servers",
            "components": {
                "mcp_servers": mcp_status,
                "a2a_subprocess_agents": a2a_status,
                "host_agent": host_status,
                "shared_memory": memory_stats
            },
            "active_sessions": len(user_sessions),
            "facebook_config": {
                "verify_token": "configured" if VERIFY_TOKEN else "missing",
                "page_token": "configured" if PAGE_ACCESS_TOKEN else "missing"
            },
            "protocols": {
                "mcp": "active",
                "agent_cards": "preserved",
                "a2a_protocol": "compliant"
            }
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/test-chat")
async def test_chat(request: Request):
    """Test endpoint"""
    try:
        data = await request.json()
        message = data.get("message", "")
        user_id = data.get("user_id", "test-user")
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        print(f"🧪 Test chat request: {message}")
        
        response = await process_with_host_agent(message, user_id)
        
        return {
            "user_message": message,
            "bot_response": response,
            "session_id": get_user_session(user_id),
            "timestamp": datetime.now().isoformat(),
            "architecture": "mcp + subprocess_a2a_with_agent_cards"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Starting Ohana Facebook Bot with MCP + A2A Servers...")
    print(f"   Architecture: MCP Servers + Subprocess A2A with AgentCard Protocol")
    print(f"   Verify Token: {'✅ Set' if VERIFY_TOKEN else '❌ Missing'}")
    print(f"   Page Token: {'✅ Set' if PAGE_ACCESS_TOKEN else '❌ Missing'}")
    print(f"   Exposed Port: 8000 (only)")
    print(f"   MCP Servers: Auto-start server_booking_mcp.py, server_info_mcp.py")
    print(f"   A2A Agents: Auto-start on localhost:9999, localhost:10002")
    
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)