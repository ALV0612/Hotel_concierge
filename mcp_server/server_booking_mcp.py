# mcp_server/server_booking_mcp.py
# Ohana Booking MCP Server (PostgreSQL) - FastMCP stdio server

import os
import sys
import signal
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional

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
logger = logging.getLogger("ohana.booking.mcp")

try:
    from zoneinfo import ZoneInfo  # Py3.9+
    _VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
except Exception:
    _VN_TZ = None

def _now_vn():
    try:
        return datetime.now(_VN_TZ) if _VN_TZ else datetime.now()
    except Exception:
        return datetime.now()

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
    
    def execute_query_returning(self, query: str, params=None) -> List[Dict]:
        """Execute query with RETURNING clause"""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                conn.commit()
                if cur.description:
                    return [dict(row) for row in cur.fetchall()]
                return []
        finally:
            self.put_connection(conn)

# Initialize database manager and FastMCP
db = DatabaseManager()
booking_agent = fastmcp.FastMCP("Ohana Booking Agent - PostgreSQL")

# ====== HELPER FUNCTIONS ======
def generate_booking_id() -> str:
    """BKG-YYYYMMDD-XXXX (đếm trong ngày)."""
    today = datetime.now().strftime("%Y%m%d")
    
    # Count existing bookings today
    query = """
    SELECT COUNT(*) as count 
    FROM bookings 
    WHERE booking_id LIKE %s
    """
    results = db.execute_query(query, [f"BKG-{today}-%"])
    count = results[0]['count'] if results else 0
    
    return f"BKG-{today}-{(count + 1):04d}"

# ====== BOOKING TOOLS ======
@booking_agent.tool()
def create_booking(
    room_id: str,
    check_in: str,
    check_out: str,
    guest_name: str,
    phone_number: Optional[str] = None,
    status: str = "confirmed"
) -> Dict:
    """
    Tạo booking mới với tính toán giá tự động
    
    Args:
        room_id: Mã phòng
        check_in: Ngày check-in (YYYY-MM-DD)
        check_out: Ngày check-out (YYYY-MM-DD)
        guest_name: Tên khách hàng
        phone_number: Số điện thoại (tùy chọn)
        status: Trạng thái booking (confirmed/paid/hold/reserved)
    """
    now_vn = _now_vn()
    today_vn = now_vn.date().isoformat()

    # Validate required fields
    if not all([room_id, check_in, check_out, guest_name]):
        return {
            "success": False,
            "message": "Thiếu thông tin bắt buộc",
            "required_fields": ["room_id", "check_in", "check_out", "guest_name"],
            "server_time": {
                "today_vn": today_vn, 
                "now_vn": now_vn.isoformat(timespec="seconds"), 
                "timezone": "Asia/Ho_Chi_Minh" if _VN_TZ else "local"
            }
        }

    try:
        # 1. Check room exists and get price info
        room_check_query = """
        SELECT room_id, room_type, capacity, base_price, status 
        FROM rooms 
        WHERE room_id = %s AND status = 'active'
        """
        room_results = db.execute_query(room_check_query, [room_id])
        
        if not room_results:
            return {
                "success": False,
                "message": f"Phòng {room_id} không tồn tại hoặc không khả dụng"
            }

        # 2. Calculate total amount IMMEDIATELY after getting room info
        room_info = room_results[0]
        room_price = room_info['base_price']
        
        # Parse dates and calculate nights
        ci_date = datetime.strptime(check_in, "%Y-%m-%d")
        co_date = datetime.strptime(check_out, "%Y-%m-%d")
        nights = (co_date - ci_date).days
        
        if nights <= 0:
            return {
                "success": False,
                "message": "Ngày check-out phải sau ngày check-in"
            }
        
        # Calculate total amount
        total_amount = room_price * nights

        # 3. Generate booking ID
        booking_id = generate_booking_id()
        
        # 4. Handle customer - check if exists or create new
        customer_query = """
        SELECT customer_id FROM customers 
        WHERE guest_name = %s AND guest_phone = %s
        """
        customer_results = db.execute_query(
            customer_query, 
            [guest_name.strip(), (phone_number or "").strip()]
        )
        
        if customer_results:
            customer_id = customer_results[0]['customer_id']
        else:
            # Create new customer
            insert_customer_query = """
            INSERT INTO customers (guest_name, guest_phone)
            VALUES (%s, %s)
            RETURNING customer_id
            """
            customer_results = db.execute_query_returning(
                insert_customer_query, 
                [guest_name.strip(), (phone_number or "").strip()]
            )
            customer_id = customer_results[0]['customer_id']

        # 5. Create booking with calculated total_amount
        insert_booking_query = """
        INSERT INTO bookings (
            booking_id, room_id, customer_id, check_in, check_out, total_amount, status
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING *
        """
        
        booking_results = db.execute_query_returning(
            insert_booking_query,
            [booking_id, room_id, customer_id, check_in, check_out, total_amount, status]
        )

        if booking_results:
            # Format price display
            formatted_amount = f"{total_amount:,}".replace(",", ".")
            
            return {
                "success": True,
                "booking_id": booking_id,
                "message": "✅ Booking thành công!",
                "server_time": {
                    "today_vn": today_vn,
                    "now_vn": now_vn.isoformat(timespec="seconds"),
                    "timezone": "Asia/Ho_Chi_Minh" if _VN_TZ else "local"
                },
                "booking_details": {
                    "booking_id": booking_id,
                    "room_id": room_id,
                    "room_type": room_info['room_type'],
                    "check_in": check_in,
                    "check_out": check_out,
                    "nights": nights,
                    "guest_name": guest_name,
                    "phone_number": phone_number or "N/A",
                    "status": status,
                    "customer_id": customer_id,
                    "base_price": room_price,
                    "total_amount": total_amount,
                    "formatted_amount": f"{formatted_amount} VND"
                }
            }
        else:
            return {
                "success": False,
                "message": "Lỗi khi tạo booking trong database"
            }

    except ValueError as ve:
        logger.error(f"Date parsing error: {ve}")
        return {
            "success": False,
            "error": str(ve),
            "message": "Lỗi định dạng ngày tháng"
        }
    except Exception as e:
        logger.error(f"Error creating booking: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": "Lỗi khi tạo booking"
        }
@booking_agent.tool()
def update_booking_status(booking_id: str, new_status: str) -> Dict:
    """Cập nhật trạng thái booking"""
    valid_statuses = ["confirmed", "paid", "hold", "reserved", "cancelled"]
    
    if new_status.lower() not in valid_statuses:
        return {
            "success": False,
            "message": f"Status không hợp lệ",
            "valid_statuses": valid_statuses
        }

    try:
        # Check if booking exists
        check_query = "SELECT booking_id, status FROM bookings WHERE booking_id = %s"
        existing = db.execute_query(check_query, [booking_id])
        
        if not existing:
            return {
                "success": False,
                "message": f"Không tìm thấy booking {booking_id}"
            }

        # Update status (no updated_at column in schema)
        update_query = """
        UPDATE bookings 
        SET status = %s 
        WHERE booking_id = %s
        RETURNING booking_id, status
        """
        
        results = db.execute_query_returning(
            update_query, 
            [new_status.lower(), booking_id]
        )

        if results:
            return {
                "success": True,
                "message": f"Đã update status thành '{new_status}'",
                "booking_id": booking_id,
                "new_status": new_status.lower(),
                "old_status": existing[0]['status']
            }
        else:
            return {
                "success": False,
                "message": "Lỗi khi update status"
            }

    except Exception as e:
        logger.error(f"Error updating booking status: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": "Lỗi khi update status"
        }

@booking_agent.tool()
def get_booking_by_id(booking_id: str) -> Dict:
    """Tra cứu booking theo mã"""
    try:
        query = """
        SELECT 
            b.*,
            c.guest_name,
            c.guest_phone,
            r.room_type,
            r.capacity,
            r.base_price
        FROM bookings b
        JOIN customers c ON b.customer_id = c.customer_id
        JOIN rooms r ON b.room_id = r.room_id
        WHERE b.booking_id = %s
        """
        
        results = db.execute_query(query, [booking_id])
        
        if results:
            return {
                "success": True,
                "booking": results[0]
            }
        else:
            return {
                "success": False,
                "message": f"Không tìm thấy booking {booking_id}"
            }

    except Exception as e:
        logger.error(f"Error getting booking by ID: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@booking_agent.tool()
def search_bookings_by_guest(guest_name: str) -> List[Dict]:
    """Tìm booking theo tên khách"""
    try:
        query = """
        SELECT 
            b.*,
            c.guest_name,
            c.guest_phone,
            r.room_type,
            r.capacity
        FROM bookings b
        JOIN customers c ON b.customer_id = c.customer_id
        JOIN rooms r ON b.room_id = r.room_id
        WHERE c.guest_name ILIKE %s
        ORDER BY b.check_in DESC
        """
        
        results = db.execute_query(query, [f"%{guest_name}%"])
        return results

    except Exception as e:
        logger.error(f"Error searching bookings by guest: {str(e)}")
        return [{"error": str(e), "message": "Lỗi khi search bookings"}]

@booking_agent.tool()
def get_recent_bookings(limit: int = 10) -> List[Dict]:
    """Xem booking gần đây"""
    try:
        query = """
        SELECT 
            b.*,
            c.guest_name,
            c.guest_phone,
            r.room_type,
            r.capacity
        FROM bookings b
        JOIN customers c ON b.customer_id = c.customer_id
        JOIN rooms r ON b.room_id = r.room_id
        WHERE b.status != 'cancelled'
        ORDER BY b.created_at DESC
        LIMIT %s
        """
        
        results = db.execute_query(query, [max(1, int(limit))])
        return results

    except Exception as e:
        logger.error(f"Error getting recent bookings: {str(e)}")
        return [{"error": str(e), "message": "Lỗi khi lấy recent bookings"}]

@booking_agent.tool()
def get_booking_stats() -> Dict:
    """Thống kê booking"""
    try:
        # Total bookings by status
        status_query = """
        SELECT status, COUNT(*) as count
        FROM bookings
        GROUP BY status
        """
        status_results = db.execute_query(status_query)
        status_breakdown = {row['status']: row['count'] for row in status_results}
        
        # Total bookings
        total_query = "SELECT COUNT(*) as total FROM bookings"
        total_results = db.execute_query(total_query)
        total_bookings = total_results[0]['total'] if total_results else 0
        
        # Today's check-ins and check-outs
        today = datetime.now().strftime("%Y-%m-%d")
        
        checkin_query = """
        SELECT COUNT(*) as count
        FROM bookings
        WHERE check_in = %s AND status IN ('confirmed', 'paid')
        """
        checkin_results = db.execute_query(checkin_query, [today])
        today_checkins = checkin_results[0]['count'] if checkin_results else 0
        
        checkout_query = """
        SELECT COUNT(*) as count
        FROM bookings
        WHERE check_out = %s AND status IN ('confirmed', 'paid')
        """
        checkout_results = db.execute_query(checkout_query, [today])
        today_checkouts = checkout_results[0]['count'] if checkout_results else 0
        
        return {
            "total_bookings": total_bookings,
            "status_breakdown": status_breakdown,
            "today": {
                "check_ins": today_checkins,
                "check_outs": today_checkouts,
                "date": today
            }
        }

    except Exception as e:
        logger.error(f"Error getting booking stats: {str(e)}")
        return {"error": str(e), "message": "Lỗi khi lấy statistics"}

@booking_agent.tool()
def get_agent_capabilities() -> Dict:
    """Thông tin về khả năng của agent"""
    return {
        "agent_name": "Ohana Booking Agent",
        "version": "2.0.0 - PostgreSQL",
        "capabilities": [
            "create_booking - Tạo booking mới",
            "update_booking_status - Cập nhật trạng thái booking",
            "get_booking_by_id - Tra cứu booking theo mã",
            "search_bookings_by_guest - Tìm booking theo tên khách",
            "get_recent_bookings - Xem booking gần đây",
            "get_booking_stats - Thống kê booking",
        ],
        "required_inputs_for_booking": {
            "room_id": "Mã phòng (từ Room Search Agent)",
            "check_in": "YYYY-MM-DD",
            "check_out": "YYYY-MM-DD",
            "guest_name": "Tên khách",
            "phone_number": "Số ĐT (optional)"
        },
        "integration": {
            "storage": "PostgreSQL Database",
            "tables": ["bookings", "customers", "rooms"]
        }
    }

@booking_agent.tool()
def refresh_booking_data() -> str:
    """Test database connection và hiển thị thống kê"""
    try:
        # Test connection
        query = "SELECT COUNT(*) as booking_count FROM bookings"
        booking_count = db.execute_query(query)[0]['booking_count']
        
        query = "SELECT COUNT(*) as customer_count FROM customers"
        customer_count = db.execute_query(query)[0]['customer_count']
        
        # Recent booking
        query = """
        SELECT b.booking_id, c.guest_name, b.status, b.created_at
        FROM bookings b
        JOIN customers c ON b.customer_id = c.customer_id
        ORDER BY b.created_at DESC
        LIMIT 1
        """
        recent = db.execute_query(query)
        recent_info = ""
        if recent:
            recent_booking = recent[0]
            recent_info = f", booking gần nhất: {recent_booking['booking_id']} ({recent_booking['guest_name']})"
        
        return f"Database connected! {booking_count} booking, {customer_count} khách hàng{recent_info}"
        
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        return f"Database connection error: {str(e)}"

# ====== ENTRYPOINT ======
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("ohana.booking")
    
    log.info("Starting Booking MCP server...")
    booking_agent.run()  # FastMCP stdio