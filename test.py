# -*- coding: utf-8 -*-
"""
Chat Test CLI cho Host Agent
Chạy độc lập để test conversation
"""

import asyncio
import os
from datetime import datetime
from agents.host_agent.agent import HostAgentExecutor, shared_memory

async def chat_cli():
    """CLI chat interface cho Host Agent"""
    
    print("🏨 Ohana Host Agent - Chat Test")
    print("=" * 50)
    print("Host Agent sẽ giao tiếp với:")
    print(f"  📋 GetInfo Agent:  http://localhost:{os.getenv('INFO_PORT', '10002')}")
    print(f"  🏨 Booking Agent:  http://localhost:{os.getenv('BOOKING_PORT', '9999')}")
    print()
    print("Hãy đảm bảo 2 agents này đã chạy trước!")
    print("Gõ 'quit' để thoát, 'reset' để reset session")
    print("=" * 50)
    
    # Khởi tạo Host Agent Executor
    executor = HostAgentExecutor()
    session_id = shared_memory.get_or_create_session()
    
    print(f"Session: {session_id}")
    print(f"Hôm nay: {datetime.now().strftime('%d/%m/%Y (%A) - %H:%M')}")
    
    while True:
        try:
            # Input từ user
            user_input = input("\n👤 Bạn: ").strip()
            
            if user_input.lower() in ["quit", "exit", "thoát"]:
                print("\n🏨 Cảm ơn bạn đã sử dụng dịch vụ Ohana Hotel!")
                break
                
            if user_input.lower() in ["reset", "làm lại"]:
                session_id = shared_memory.start_new_conversation()
                print(f"🔄 Đã reset! Session mới: {session_id}")
                continue
                
            if not user_input:
                continue
                
            # Gửi tới Host Agent
            print("\n🤖 Host Agent đang xử lý...")
            response = await executor.execute(user_input, session_id)
            
            print(f"\n🎯 Host: {response}")
            
            # Show memory stats
            stats = shared_memory.get_session_stats()
            print(f"\n💾 Session: {stats['message_count']} messages, {stats['uptime_minutes']}min")
            
        except KeyboardInterrupt:
            print("\n\n👋 Tạm biệt!")
            break
        except Exception as e:
            print(f"\n⚠️ Lỗi: {e}")

if __name__ == "__main__":
    # Check dependencies
    try:
        from agents.host_agent.agent import HostAgentExecutor, shared_memory
        print("✅ Host Agent modules loaded successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Hãy đảm bảo file host_agent_main.py đã được tạo!")
        exit(1)
    
    # Run chat
    asyncio.run(chat_cli())