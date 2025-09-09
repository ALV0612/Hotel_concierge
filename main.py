import os
import uvicorn
import subprocess
import time
import signal
import atexit
from fastapi import FastAPI, Request, HTTPException
from typing import Dict, Any, Optional
import json
from datetime import datetime
import requests
from dotenv import load_dotenv

# Host agent
from agents.host_agent.agent import HostRuntime, shared_memory

load_dotenv()

class SubprocessA2AServers:
    """Chạy booking và info agents như subprocess riêng biệt"""
    
    def __init__(self):
        self.booking_port = 9999
        self.info_port = 10002
        self.booking_process = None
        self.info_process = None
        self.servers_ready = False
        
        # Register cleanup on exit
        atexit.register(self.cleanup)

    def start_booking_agent(self):
        """Chạy Booking Agent như subprocess"""
        try:
            print(f"Starting Booking Agent subprocess on port {self.booking_port}")
            
            # Set environment variables for subprocess
            env = os.environ.copy()
            env["BOOKING_PORT"] = str(self.booking_port)
            
            # Start subprocess
            self.booking_process = subprocess.Popen([
                "python", "-m", "agents.booking_agent.__main__"
            ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            print(f"Booking Agent subprocess started with PID: {self.booking_process.pid}")
            return True
            
        except Exception as e:
            print(f"Error starting Booking Agent subprocess: {e}")
            return False

    def start_info_agent(self):
        """Chạy Info Agent như subprocess"""
        try:
            print(f"Starting Info Agent subprocess on port {self.info_port}")
            
            # Set environment variables for subprocess
            env = os.environ.copy()
            env["INFO_PORT"] = str(self.info_port)
            
            # Start subprocess
            self.info_process = subprocess.Popen([
                "python", "-m", "agents.get_info_agent.__main__"
            ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            print(f"Info Agent subprocess started with PID: {self.info_process.pid}")
            return True
            
        except Exception as e:
            print(f"Error starting Info Agent subprocess: {e}")
            return False

    def check_server_health(self, url, timeout=3):
        """Kiểm tra server health"""
        try:
            # Try different endpoints
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
        """Đợi servers sẵn sàng"""
        print(f"Waiting for subprocess servers to be ready (max {max_wait}s)...")
        start_time = time.time()
        
        booking_ready = False
        info_ready = False
        
        while time.time() - start_time < max_wait:
            # Check booking agent
            if not booking_ready:
                booking_ready = self.check_server_health(f"http://localhost:{self.booking_port}")
                if booking_ready:
                    print("✅ Booking Agent subprocess ready")
            
            # Check info agent  
            if not info_ready:
                info_ready = self.check_server_health(f"http://localhost:{self.info_port}")
                if info_ready:
                    print("✅ Info Agent subprocess ready")
            
            # Both ready?
            if booking_ready and info_ready:
                self.servers_ready = True
                print("✅ All subprocess A2A servers are ready!")
                return True
            
            time.sleep(2)
        
        print("⚠️ Warning: Some subprocess servers may not be ready yet")
        print(f"   Booking Agent: {'✅' if booking_ready else '❌'}")
        print(f"   Info Agent: {'✅' if info_ready else '❌'}")
        
        return booking_ready or info_ready  # At least one working

    def start_all(self):
        """Khởi động tất cả subprocess servers"""
        print("Starting subprocess A2A servers...")
        
        booking_started = self.start_booking_agent()
        info_started = self.start_info_agent()
        
        if not (booking_started and info_started):
            print("❌ Failed to start some subprocess servers")
            return False
        
        # Wait for servers to be ready
        return self.wait_for_servers()

    def cleanup(self):
        """Cleanup subprocess khi thoát"""
        print("\n🧹 Cleaning up subprocess servers...")
        
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
        """Lấy status của các subprocess"""
        booking_status = "stopped"
        info_status = "stopped"
        
        if self.booking_process:
            if self.booking_process.poll() is None:  # Still running
                booking_status = "running" if self.check_server_health(f"http://localhost:{self.booking_port}") else "starting"
            else:
                booking_status = "crashed"
        
        if self.info_process:
            if self.info_process.poll() is None:  # Still running
                info_status = "running" if self.check_server_health(f"http://localhost:{self.info_port}") else "starting"
            else:
                info_status = "crashed"
        
        return {
            "booking_agent": booking_status,
            "info_agent": info_status,
            "booking_pid": self.booking_process.pid if self.booking_process else None,
            "info_pid": self.info_process.pid if self.info_process else None
        }

# FastAPI app
app = FastAPI(title="Ohana Facebook Bot - Subprocess A2A", version="1.0.0")

# Facebook configuration
VERIFY_TOKEN = os.getenv("FB_VERIFY_TOKEN", "ohana_verify_token_2025")
PAGE_ACCESS_TOKEN = os.getenv("FB_PAGE_ACCESS_TOKEN", "")

# Global instances
subprocess_servers = None
host_runtime = None
user_sessions = {}

def get_user_session(fb_user_id: str) -> str:
    """Tạo hoặc lấy session cho Facebook user"""
    if fb_user_id not in user_sessions:
        user_sessions[fb_user_id] = f"fb-user-{fb_user_id}"
    return user_sessions[fb_user_id]

@app.on_event("startup")
async def startup_event():
    """Khởi động subprocess servers khi FastAPI start"""
    global subprocess_servers, host_runtime
    
    print("🚀 Starting Ohana Facebook Bot with Subprocess A2A...")
    
    # Start subprocess A2A servers
    subprocess_servers = SubprocessA2AServers()
    servers_ready = subprocess_servers.start_all()
    
    if servers_ready:
        print("✅ Subprocess servers ready")
    else:
        print("⚠️ Some subprocess servers may not be ready")
    
    # Initialize host runtime
    os.environ["BOOKING_AGENT_URL"] = f"http://localhost:{subprocess_servers.booking_port}"
    os.environ["INFO_AGENT_URL"] = f"http://localhost:{subprocess_servers.info_port}"
    
    host_runtime = HostRuntime()
    print("✅ Host Agent runtime initialized")
    
    print(f"🎯 Main API ready on port 8000 (only exposed port)")
    print(f"🔧 Subprocess Booking Agent: localhost:{subprocess_servers.booking_port}")
    print(f"🔧 Subprocess Info Agent: localhost:{subprocess_servers.info_port}")

@app.get("/")
async def root():
    """API status endpoint"""
    return {
        "message": "Ohana Facebook Bot - Subprocess A2A Architecture", 
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "architecture": "subprocess_a2a_servers",
        "exposed_port": "8000_only",
        "subprocess_agents": {
            "booking": f"localhost:{subprocess_servers.booking_port if subprocess_servers else 'starting'}",
            "info": f"localhost:{subprocess_servers.info_port if subprocess_servers else 'starting'}"
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
        
        print(f"🤖 Processing via Host Agent:")
        print(f"   Message: {message}")
        print(f"   Session: {session_id}")
        
        shared_memory.get_or_create_session(message)
        response = await host_runtime.ask(message, session_id=session_id)
        
        print(f"🤖 Host Agent response: {response}")
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
    """Health check với thông tin chi tiết"""
    try:
        subprocess_status = subprocess_servers.get_status() if subprocess_servers else {}
        host_status = "connected" if host_runtime else "not_initialized"
        memory_stats = shared_memory.get_session_stats() if shared_memory.current_session else {}
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "architecture": "subprocess_a2a_servers",
            "host_agent": host_status,
            "subprocess_agents": subprocess_status,
            "shared_memory": memory_stats,
            "active_sessions": len(user_sessions),
            "facebook_config": {
                "verify_token": "configured" if VERIFY_TOKEN else "missing",
                "page_token": "configured" if PAGE_ACCESS_TOKEN else "missing"
            },
            "agent_cards": "preserved",
            "protocol": "A2A_compliant"
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
            "architecture": "subprocess_a2a_with_agent_cards"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Starting Ohana Facebook Bot with Subprocess A2A Servers...")
    print(f"   Architecture: Subprocess A2A with AgentCard Protocol")
    print(f"   Verify Token: {'✅ Set' if VERIFY_TOKEN else '❌ Missing'}")
    print(f"   Page Token: {'✅ Set' if PAGE_ACCESS_TOKEN else '❌ Missing'}")
    print(f"   Exposed Port: 8000 (only)")
    print(f"   Subprocess Agents: Will start on localhost:9999, localhost:10002")
    
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)