# pip install fastmcp gspread google-auth tzdata

import os, time, random
import fastmcp
import gspread
from google.oauth2.service_account import Credentials
from typing import Dict, Optional, List
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

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

print(f"Current GOOGLE_API_KEY: {os.environ.get('GOOGLE_API_KEY','NOT_SET')[:10]}...")

booking_agent = fastmcp.FastMCP("Ohana Booking Agent - A2A Protocol")

SHEET_ID = os.environ.get("OHANA_SHEET_ID")
CREDS_FILE = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE")

# --- ENV VALIDATION (NEW) ---
if not SHEET_ID:
    raise RuntimeError("Missing OHANA_SHEET_ID")
if not CREDS_FILE or not os.path.exists(CREDS_FILE):
    raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_FILE or path not found")

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
creds = Credentials.from_service_account_file(CREDS_FILE, scopes=SCOPES)
gc = gspread.authorize(creds)
sh = gc.open_by_key(SHEET_ID)

# --- ENSURE WORKSHEET + HEADERS (NEW) ---
HEADERS = ["booking_id","room_id","check_in","check_out","status","guest_name","phone_number"]

def _ensure_sheet():
    try:
        ws = sh.worksheet("Bookings")
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="Bookings", rows=1000, cols=len(HEADERS))
        ws.append_row(HEADERS)
        return ws
    # ensure headers
    first_row = ws.row_values(1)
    if [h.lower() for h in first_row] != HEADERS:
        # rewrite header (idempotent)
        ws.resize(1)
        ws.update("A1", [HEADERS])
    return ws

bookings_ws = _ensure_sheet()

# --- small backoff helper (NEW) ---
def _gs_append_row(ws, row, tries=3):
    for i in range(tries):
        try:
            return ws.append_row(row)
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(0.4 + 0.2 * i + random.random()*0.2)

# Helpers
def generate_booking_id() -> str:
    """BKG-YYYYMMDD-XXXX (đếm trong ngày)."""
    today = datetime.now().strftime("%Y%m%d")
    existing_bookings = bookings_ws.get_all_records()
    today_bookings = [b.get("booking_id","") for b in existing_bookings if str(b.get("booking_id","")).startswith(f"BKG-{today}")]
    last_num = 0
    for b in today_bookings:
        try:
            last_num = max(last_num, int(str(b).split("-")[-1]))
        except Exception:
            pass
    return f"BKG-{today}-{(last_num+1):04d}"

def add_booking_to_sheet(booking_data: Dict) -> Dict:
    try:
        booking_id = generate_booking_id()
        new_row = [
            booking_id,
            booking_data["room_id"],
            booking_data["check_in"],
            booking_data["check_out"],
            booking_data.get("status", "confirmed"),
            booking_data["guest_name"],
            booking_data.get("phone_number", ""),
        ]
        _gs_append_row(bookings_ws, new_row)  # FIX: retry
        return {"success": True, "booking_id": booking_id, "message": f"Booking {booking_id} đã được lưu vào Google Sheets"}
    except Exception as e:
        return {"success": False, "error": str(e), "message": "Lỗi khi thêm booking vào Google Sheets"}

@booking_agent.tool()
def create_booking(
    room_id: str,
    check_in: str,
    check_out: str,
    guest_name: str,
    phone_number: Optional[str] = None,
    status: str = "confirmed"
) -> Dict:
    now_vn = _now_vn()
    today_vn = now_vn.date().isoformat()

    if not all([room_id, check_in, check_out, guest_name]):
        return {
            "success": False,
            "message": "Thiếu thông tin bắt buộc",
            "required_fields": ["room_id", "check_in", "check_out", "guest_name"],
            "server_time": {"today_vn": today_vn, "now_vn": now_vn.isoformat(timespec="seconds"), "timezone": "Asia/Ho_Chi_Minh" if _VN_TZ else "local"}
        }

    booking_data = {
        "room_id": room_id,
        "check_in": check_in,
        "check_out": check_out,
        "guest_name": guest_name.strip(),
        "phone_number": (phone_number or "").strip(),
        "status": status
    }

    result = add_booking_to_sheet(booking_data)
    if result.get("success"):
        ci_date = datetime.strptime(check_in, "%Y-%m-%d")
        co_date = datetime.strptime(check_out, "%Y-%m-%d")
        nights = (co_date - ci_date).days
        return {
            "success": True,
            "booking_id": result["booking_id"],
            "message": "✅ Booking thành công!",
            "server_time": {"today_vn": today_vn, "now_vn": now_vn.isoformat(timespec="seconds"), "timezone": "Asia/Ho_Chi_Minh" if _VN_TZ else "local"},
            "booking_details": {
                "booking_id": result["booking_id"],
                "room_id": room_id, "check_in": check_in, "check_out": check_out, "nights": nights,
                "guest_name": guest_name, "phone_number": phone_number or "N/A", "status": status
            }
        }
    else:
        result.setdefault("server_time", {"today_vn": today_vn, "now_vn": now_vn.isoformat(timespec="seconds"), "timezone": "Asia/Ho_Chi_Minh" if _VN_TZ else "local"})
        return result

@booking_agent.tool()
def update_booking_status(booking_id: str, new_status: str) -> Dict:
    valid_statuses = ["confirmed", "paid", "hold", "reserved", "cancelled"]
    if new_status.lower() not in valid_statuses:
        return {"success": False, "message": f"Status không hợp lệ", "valid_statuses": valid_statuses}
    try:
        all_values = bookings_ws.get_all_values()
        if not all_values:  # không có dữ liệu
            return {"success": False, "message": f"Không tìm thấy booking {booking_id}"}
        header_row = [h.lower() for h in all_values[0]]
        try:
            status_col = header_row.index("status") + 1
        except ValueError:
            return {"success": False, "message": "Thiếu cột 'status' trong sheet"}
        booking_row_index = None
        for i, row in enumerate(all_values[1:], start=2):
            if row and len(row) > 0 and row[0] == booking_id:
                booking_row_index = i
                break
        if not booking_row_index:
            return {"success": False, "message": f"Không tìm thấy booking {booking_id}"}
        bookings_ws.update_cell(booking_row_index, status_col, new_status.lower())
        return {"success": True, "message": f"Đã update status thành '{new_status}'", "booking_id": booking_id, "new_status": new_status}
    except Exception as e:
        return {"success": False, "error": str(e), "message": "Lỗi khi update status"}

@booking_agent.tool()
def get_booking_by_id(booking_id: str) -> Dict:
    try:
        for b in bookings_ws.get_all_records():
            if str(b.get("booking_id","")) == booking_id:
                return {"success": True, "booking": b}
        return {"success": False, "message": f"Không tìm thấy booking {booking_id}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@booking_agent.tool()
def search_bookings_by_guest(guest_name: str) -> List[Dict]:
    try:
        bookings = bookings_ws.get_all_records()
        q = [w for w in guest_name.lower().split() if w]
        out = []
        for b in bookings:
            g = str(b.get("guest_name","")).lower().strip()
            if g and all(w in g for w in q):
                out.append(b)
        return sorted(out, key=lambda x: x.get("check_in",""), reverse=True)
    except Exception as e:
        return [{"error": str(e), "message": "Lỗi khi search bookings"}]

@booking_agent.tool()
def get_recent_bookings(limit: int = 10) -> List[Dict]:
    try:
        bookings = bookings_ws.get_all_records()
        active = [b for b in bookings if str(b.get("status","")).lower() != "cancelled"]
        return sorted(active, key=lambda x: x.get("check_in",""), reverse=True)[:max(1, int(limit))]
    except Exception as e:
        return [{"error": str(e), "message": "Lỗi khi lấy recent bookings"}]

@booking_agent.tool()
def get_booking_stats() -> Dict:
    try:
        bookings = bookings_ws.get_all_records()
        status_counts: Dict[str,int] = {}
        for b in bookings:
            status = str(b.get("status","unknown")).lower()
            status_counts[status] = status_counts.get(status, 0) + 1
        today = datetime.now().strftime("%Y-%m-%d")
        today_checkins  = [b for b in bookings if b.get("check_in")  == today and str(b.get("status","")).lower() in ("confirmed","paid")]
        today_checkouts = [b for b in bookings if b.get("check_out") == today and str(b.get("status","")).lower() in ("confirmed","paid")]
        return {"total_bookings": len(bookings), "status_breakdown": status_counts, "today": {"check_ins": len(today_checkins), "check_outs": len(today_checkouts), "date": today}}
    except Exception as e:
        return {"error": str(e), "message": "Lỗi khi lấy statistics"}

@booking_agent.tool()
def get_agent_capabilities() -> Dict:
    return {
        "agent_name": "Ohana Booking Agent",
        "version": "1.0.0",
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
        "integration": {"storage": "Google Sheets", "sheet_name": "Bookings"}
    }

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("booking_agent").info("Starting Booking Agent MCP server...")
    booking_agent.run()  # FastMCP stdio
