from fastapi import FastAPI, Request, HTTPException
import os
import requests
import asyncio
from typing import Dict, Any, Optional
import json
from datetime import datetime
import uvicorn
from dotenv import load_dotenv

# Import Host Agent từ file của bạn
from agent import HostRuntime, shared_memory

load_dotenv()

app = FastAPI(title="Ohana Facebook Bot API", version="1.0.0")

# Facebook configuration
VERIFY_TOKEN = os.getenv("FB_VERIFY_TOKEN", "ohana_verify_token_2025")
PAGE_ACCESS_TOKEN = os.getenv("FB_PAGE_ACCESS_TOKEN", "")

# Kiểm tra config
if not PAGE_ACCESS_TOKEN:
    print("WARNING: FB_PAGE_ACCESS_TOKEN not set!")

# Khởi tạo Host Agent
host_runtime = HostRuntime()

# Lưu trữ sessions cho từng Facebook user
user_sessions = {}

def get_user_session(fb_user_id: str) -> str:
    """Tạo hoặc lấy session cho Facebook user"""
    if fb_user_id not in user_sessions:
        user_sessions[fb_user_id] = f"fb-user-{fb_user_id}"
    return user_sessions[fb_user_id]

@app.get("/")
async def root():
    """API status endpoint"""
    return {
        "message": "Ohana Facebook Bot API", 
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "host_agent": "connected",
        "facebook_integration": "active"
    }

@app.get("/webhook")
async def verify_webhook(request: Request):
    """Facebook webhook verification endpoint"""
    try:
        # Lấy parameters từ Facebook
        verify_token = request.query_params.get('hub.verify_token')
        challenge = request.query_params.get('hub.challenge')
        
        print(f"🔍 Webhook verification request:")
        print(f"   Token received: {verify_token}")
        print(f"   Expected token: {VERIFY_TOKEN}")
        print(f"   Challenge: {challenge}")
        
        # Verify token
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
        # Parse Facebook webhook data
        data = await request.json()
        print(f"📨 Received Facebook webhook:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        
        # Xử lý từng entry trong webhook
        for entry in data.get('entry', []):
            for messaging in entry.get('messaging', []):
                await process_messaging_event(messaging)
        
        return {"status": "EVENT_RECEIVED"}
        
    except Exception as e:
        print(f"❌ Error processing webhook: {e}")
        return {"status": "ERROR", "message": str(e)}

async def process_messaging_event(messaging: Dict[str, Any]):
    """Xử lý một messaging event từ Facebook"""
    try:
        sender_id = messaging.get('sender', {}).get('id')
        
        if not sender_id:
            print("⚠️ No sender ID found in messaging event")
            return
        
        # Xử lý tin nhắn text
        if 'message' in messaging:
            message = messaging['message']
            message_text = message.get('text', '').strip()
            
            if message_text:
                print(f"👤 User {sender_id}: {message_text}")
                
                # Gọi Host Agent để xử lý
                response = await process_with_host_agent(message_text, sender_id)
                
                if response:
                    # Gửi phản hồi về Facebook
                    await send_facebook_message(sender_id, response)
                else:
                    await send_facebook_message(sender_id, "Xin lỗi, tôi không hiểu yêu cầu của bạn.")
        
        # Xử lý postback (button clicks)
        elif 'postback' in messaging:
            postback = messaging['postback']
            payload = postback.get('payload', '')
            print(f"🔘 User {sender_id} clicked: {payload}")
            
            # Xử lý postback như tin nhắn text
            response = await process_with_host_agent(payload, sender_id)
            if response:
                await send_facebook_message(sender_id, response)
        
    except Exception as e:
        print(f"❌ Error processing messaging event: {e}")

async def process_with_host_agent(message: str, fb_user_id: str) -> Optional[str]:
    """Xử lý tin nhắn qua Host Agent với session management"""
    try:
        # Lấy session cho user này
        session_id = get_user_session(fb_user_id)
        
        print(f"🤖 Processing via Host Agent:")
        print(f"   Message: {message}")
        print(f"   FB User: {fb_user_id}")
        print(f"   Session: {session_id}")
        
        # Auto-detect session reset cho user này
        shared_memory.get_or_create_session(message)
        
        # Gọi Host Agent để xử lý
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
    
    # Chia tin nhắn dài thành nhiều phần (Facebook limit ~2000 chars)
    max_length = 1900
    messages = []
    
    if len(message_text) <= max_length:
        messages = [message_text]
    else:
        # Chia theo dòng để giữ format
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
    
    # Gửi từng tin nhắn
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
                print(f"📤 Sent message {i+1}/{len(messages)} to Facebook: {msg[:50]}...")
            else:
                print(f"❌ Facebook API error: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Error sending message {i+1} to Facebook: {e}")

@app.get("/health")
async def health_check():
    """Health check với thông tin chi tiết"""
    try:
        # Check Host Agent
        host_status = "connected"
        
        # Check shared memory
        memory_stats = shared_memory.get_session_stats() if shared_memory.current_session else {}
        
        # Check sub agents (nếu có thể)
        sub_agents_status = {
            "booking_agent": "unknown",  # Có thể ping BOOKING_URL
            "info_agent": "unknown"      # Có thể ping INFO_URL  
        }
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "host_agent": host_status,
            "shared_memory": memory_stats,
            "sub_agents": sub_agents_status,
            "active_sessions": len(user_sessions),
            "facebook_config": {
                "verify_token": "configured" if VERIFY_TOKEN else "missing",
                "page_token": "configured" if PAGE_ACCESS_TOKEN else "missing"
            }
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/test-chat")
async def test_chat(request: Request):
    """Test endpoint để thử nghiệm không qua Facebook"""
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
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset-session/{user_id}")
async def reset_user_session(user_id: str):
    """Reset session cho một user cụ thể"""
    try:
        if user_id in user_sessions:
            del user_sessions[user_id]
        
        # Reset shared memory nếu cần
        shared_memory.start_new_conversation()
        
        return {
            "message": f"Session reset for user {user_id}",
            "new_session": get_user_session(user_id)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Starting Ohana Facebook Bot API...")
    print(f"   Verify Token: {'✅ Set' if VERIFY_TOKEN else '❌ Missing'}")
    print(f"   Page Token: {'✅ Set' if PAGE_ACCESS_TOKEN else '❌ Missing'}")
    
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)