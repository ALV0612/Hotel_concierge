# mcp_server/server_info_mcp.py
# Ohana Info MCP Server (PostgreSQL) - FastMCP stdio server

import os
import sys
import signal
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional

import asyncpg
import fastmcp
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool

# ====== ENV & LOGGING ======
load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stderr)],  # GIỮ stdout sạch cho JSON-RPC
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("ohana.info.mcp")

# Optional: RAG
try:
    from hotel_local_rag import HotelLocalRAG
    RAG_AVAILABLE = True
except Exception:
    RAG_AVAILABLE = False

# ====== DB MANAGER ======
class DatabaseManager:
    def __init__(self):
        self.connection_pool = psycopg2.pool.SimpleConnectionPool(
            1, 20,  # min=1, max=20 connections
            host=os.environ['DB_HOST'],
            port=os.environ.get('DB_PORT', 5432),
            database=os.environ['DB_NAME'],
            user=os.environ['DB_USER'],
            password=os.environ['DB_PASSWORD']
        )
    
    def get_connection(self):
        return self.connection_pool.getconn()
    
    def put_connection(self, conn):
        self.connection_pool.putconn(conn)
    
    def execute_query(self, query: str, params=None) -> List[Dict]:
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                if cur.description:  # SELECT query
                    return [dict(row) for row in cur.fetchall()]
                else:  # INSERT/UPDATE/DELETE
                    conn.commit()
                    return []
        finally:
            self.put_connection(conn)

# Initialize database manager
db = DatabaseManager()
mcp = fastmcp.FastMCP("Ohana Info MCP Server")
# Replace the complex RAG initialization with this simple version:
try:
    from hotel_local_rag import HotelLocalRAG
    hotel_rag = HotelLocalRAG(
        vector_db_path=".hotel_vector_db",
        embedding_model_name="all-MiniLM-L6-v2",
    )
    RAG_AVAILABLE = hotel_rag.is_ready()
    print(f"RAG initialized: {RAG_AVAILABLE}")
except Exception as e:
    print(f"RAG initialization failed: {e}")
    hotel_rag = None
    RAG_AVAILABLE = False

# ====== TOOLS ======
@mcp.tool()
def get_room_types() -> List[str]:
    """Lấy danh sách tất cả các loại phòng có sẵn"""
    query = "SELECT DISTINCT room_type FROM rooms WHERE status = 'active' ORDER BY room_type"
    results = db.execute_query(query)
    return [row['room_type'] for row in results]

@mcp.tool()
def check_availability(
    check_in: str, 
    check_out: str, 
    guests: int = 1, 
    room_type: Optional[str] = None
) -> List[Dict]:
    """
    Kiểm tra phòng trống theo ngày và tiêu chí
    
    Args:
        check_in: Ngày check-in (YYYY-MM-DD)
        check_out: Ngày check-out (YYYY-MM-DD) 
        guests: Số khách (mặc định: 1)
        room_type: Loại phòng (standard/deluxe/family/suite)
    """
    
    query = """
    SELECT r.room_id, r.room_type, r.capacity, r.base_price, r.description
    FROM rooms r
    WHERE r.status = 'active' 
    AND r.capacity >= %s
    AND NOT EXISTS (
        SELECT 1 FROM bookings b 
        WHERE b.room_id = r.room_id 
        AND b.status IN ('confirmed', 'paid', 'hold', 'reserved')
        AND (b.check_in < %s AND b.check_out > %s)
    )
    """
    
    params = [guests, check_out, check_in]
    
    # Add room type filter - exact match for standard/deluxe/family/suite
    if room_type:
        room_type_lower = room_type.lower().strip()
        valid_types = ['standard', 'deluxe', 'family', 'suite']
        
        if room_type_lower in valid_types:
            query += " AND LOWER(r.room_type) = %s"
            params.append(room_type_lower)
    
    query += " ORDER BY r.room_type, r.base_price"
    
    return db.execute_query(query, params)

@mcp.tool()
def get_room_info(room_id: str) -> Dict:
    """Lấy thông tin chi tiết của một phòng"""
    query = "SELECT * FROM rooms WHERE room_id = %s"
    results = db.execute_query(query, [room_id])
    
    if results:
        return results[0]
    else:
        return {"error": f"Không tìm thấy phòng {room_id}"}

@mcp.tool()
def get_bookings_by_room(room_id: str) -> List[Dict]:
    """Lấy tất cả booking của một phòng"""
    query = """
    SELECT b.*, c.guest_name, c.guest_phone 
    FROM bookings b
    JOIN customers c ON b.customer_id = c.customer_id
    WHERE b.room_id = %s
    ORDER BY b.check_in DESC
    """
    return db.execute_query(query, [room_id])

@mcp.tool()
def search_guest(guest_name: str) -> List[Dict]:
    """Tìm kiếm booking theo tên khách"""
    query = """
    SELECT b.*, c.guest_name, c.guest_phone, r.room_type
    FROM bookings b
    JOIN customers c ON b.customer_id = c.customer_id
    JOIN rooms r ON b.room_id = r.room_id
    WHERE c.guest_name ILIKE %s
    ORDER BY b.check_in DESC
    """
    return db.execute_query(query, [f"%{guest_name}%"])

@mcp.tool()
def get_bookings_by_status(status: str) -> List[Dict]:
    """Lấy booking theo trạng thái"""
    query = """
    SELECT b.*, c.guest_name, c.guest_phone, r.room_type
    FROM bookings b
    JOIN customers c ON b.customer_id = c.customer_id
    JOIN rooms r ON b.room_id = r.room_id
    WHERE b.status = %s
    ORDER BY b.check_in DESC
    """
    return db.execute_query(query, [status])

@mcp.tool()
def hotel_summary() -> Dict:
    """Tổng quan về khách sạn"""
    
    # Room summary
    room_query = """
    SELECT 
        COUNT(*) as total_rooms,
        SUM(capacity) as total_capacity,
        room_type,
        COUNT(*) as count_by_type
    FROM rooms 
    WHERE status = 'active'
    GROUP BY room_type
    """
    
    # Booking summary
    booking_query = """
    SELECT 
        status,
        COUNT(*) as count
    FROM bookings
    GROUP BY status
    """
    
    room_results = db.execute_query(room_query)
    booking_results = db.execute_query(booking_query)
    
    return {
        "total_rooms": sum(r['count_by_type'] for r in room_results),
        "total_capacity": sum(r['total_capacity'] for r in room_results) if room_results else 0,
        "room_types": {r['room_type']: r['count_by_type'] for r in room_results},
        "booking_status": {b['status']: b['count'] for b in booking_results}
    }

@mcp.tool()
def refresh_data() -> str:
    """Test database connection"""
    try:
        query = "SELECT COUNT(*) as room_count FROM rooms"
        room_count = db.execute_query(query)[0]['room_count']
        
        query = "SELECT COUNT(*) as booking_count FROM bookings"
        booking_count = db.execute_query(query)[0]['booking_count']
        
        return f"Database connected! {room_count} phòng, {booking_count} booking"
    except Exception as e:
        return f"Database connection error: {str(e)}"

hotel_rag = HotelLocalRAG(
    vector_db_path="D:\\test\\hotel_vector_db",
    embedding_model_name="all-MiniLM-L6-v2"
)

# Add this line that was missing:
RAG_AVAILABLE = hotel_rag.is_ready()

@mcp.tool()
def query_hotel_docs(question: str, top_k: int = 3) -> List[Dict]:
    """Tìm kiếm thông tin trong tài liệu khách sạn"""
    if not hotel_rag or not RAG_AVAILABLE:
        return [{"error": "RAG system not available"}]
    return hotel_rag.query(question, k=top_k)
# @mcp.tool()
# def setup_hotel_documents(txt_folder: str = None, pdf_folder: str = None) -> Dict:
#     """Setup tài liệu khách sạn từ thư mục TXT hoặc PDF"""
#     results = {}
#     if txt_folder:
#         results["txt"] = hotel_rag.create_db_from_txt_folder(txt_folder)
#     if pdf_folder:
#         results["pdf"] = hotel_rag.create_db_from_pdf_folder(pdf_folder)
#     return results

@mcp.tool()
def hotel_rag_summary() -> Dict:
    """Thông tin về hệ thống RAG"""
    return hotel_rag.get_summary()


# ====== ENTRYPOINT ======
if __name__ == "__main__":
    import os, sys, logging
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("ohana")

    log.info("Starting MCP server...")  # -> stderr, an toàn
    mcp.run()  # chỉ dòng này là cần để chạy server
