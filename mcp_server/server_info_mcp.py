# pip install fastmcp gspread google-auth

import os
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, date
from typing import List, Dict, Optional
import fastmcp
from hotel_local_rag import HotelLocalRAG

# --- Thêm ở đầu file (gần import) ---
import re, unicodedata
from dotenv import load_dotenv
load_dotenv()
def _slug(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    return " ".join(s.split())

# Từ khoá tiếng Việt/Anh → capacity
_CAP_ALIASES = {
    1: {"1", "một", "đơn", "single", "1 nguoi", "1 pax"},
    2: {"2", "hai", "đôi", "double", "twin", "2 nguoi", "2 pax"},
    3: {"3", "ba", "triple", "3 nguoi", "3 pax"},
    4: {"4", "bốn", "quad", "quadruple", "4 nguoi", "4 pax"},
}

def _parse_room_hint(text: Optional[str]) -> tuple[Optional[str], Optional[int]]:
    """
    Trả về (type_filter, cap_hint) từ chuỗi tự nhiên như:
    'phòng đôi', '2 người', 'deluxe', 'family 4 người', ...
    """
    if not text:
        return None, None
    s = _slug(text)

    # Ưu tiên bắt số người nếu có '2/3/4 (nguoi|pax)'
    m = re.search(r"(\d+)\s*(nguoi|pax)?", s)
    if m:
        try:
            n = int(m.group(1))
            if n in (1, 2, 3, 4):
                return None, n
        except:
            pass

    for cap, words in _CAP_ALIASES.items():
        if any(w in s for w in words):
            return None, cap

    return None, None

import os, json, base64

def load_sa_credentials(scopes: List[str] = None) -> Credentials:
    """
    Hỗ trợ: FILE PATH / RAW JSON / BASE64 JSON.
    Ưu tiên:
      1) GOOGLE_SERVICE_ACCOUNT_JSON
      2) GOOGLE_SERVICE_ACCOUNT_FILE
      3) GOOGLE_APPLICATION_CREDENTIALS (path chuẩn của Google SDK)
    """
    raw = (
        os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        or os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    )
    if not raw:
        raise RuntimeError("Missing service account credentials (JSON or file path).")

    raw = raw.strip()

    # (1) Nếu là JSON trực tiếp
    if raw.startswith("{"):
        try:
            info = json.loads(raw)
            return Credentials.from_service_account_info(info, scopes=SCOPES)
        except Exception as e:
            raise RuntimeError(f"Invalid inline JSON for service account: {e}")

    # (2) Nếu là BASE64 của JSON
    try:
        decoded = base64.b64decode(raw).decode("utf-8")
        if decoded.strip().startswith("{"):
            info = json.loads(decoded)
            return Credentials.from_service_account_info(info, scopes=SCOPES)
    except Exception:
        pass  # không phải base64 → thử như file path

    # (3) Xử lý như FILE PATH
    if not os.path.exists(raw):
        raise RuntimeError(f"Service account file not found: {raw}")
    return Credentials.from_service_account_file(raw, scopes=SCOPES)

# --- Khởi tạo FastMCP ---
mcp = fastmcp.FastMCP("Ohana Hotel Booking")

SHEET_ID = os.environ["OHANA_SHEET_ID"]
CREDS_FILE = os.environ["GOOGLE_SERVICE_ACCOUNT_FILE"]

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
creds = Credentials.load_sa_credentials(SCOPES)
gc = gspread.authorize(creds)

sh = gc.open_by_key(SHEET_ID)
rooms_ws = sh.worksheet("Rooms")
bookings_ws = sh.worksheet("Bookings")

rooms = rooms_ws.get_all_records()
bookings = bookings_ws.get_all_records()

# --- Helpers ---
def d(s: str) -> date:
    return datetime.fromisoformat(str(s)).date()

BLOCK = {"confirmed", "paid", "hold", "reserved"}

def overlap(a1: date, a2: date, b1: date, b2: date) -> bool:
    return not (a2 <= b1 or a1 >= b2)

# --- Convert thành MCP Tools ---

@mcp.tool()
def get_room_types() -> List[str]:
    """Lấy danh sách tất cả các loại phòng có sẵn"""
    return _get_room_types()

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
        room_type: Loại phòng (tùy chọn, ví dụ: "Deluxe")
    """
    return _check_availability(check_in, check_out, guests, room_type)

@mcp.tool()
def refresh_data() -> str:
    """Refresh dữ liệu từ Google Sheets"""
    global rooms, bookings
    try:
        rooms = rooms_ws.get_all_records()
        bookings = bookings_ws.get_all_records()
        return f"Đã refresh thành công! {len(rooms)} phòng, {len(bookings)} booking"
    except Exception as e:
        return f"Lỗi refresh: {str(e)}"

@mcp.tool()
def get_room_info(room_id: str) -> Dict:
    """Lấy thông tin chi tiết của một phòng"""
    for r in rooms:
        if r["room_id"] == room_id:
            return r
    return {"error": f"Không tìm thấy phòng {room_id}"}

@mcp.tool()
def get_bookings_by_room(room_id: str) -> List[Dict]:
    """Lấy tất cả booking của một phòng"""
    room_bookings = [b for b in bookings if b["room_id"] == room_id]
    return sorted(room_bookings, key=lambda x: x.get("check_in", ""))

@mcp.tool()
def search_guest(guest_name: str) -> List[Dict]:
    """Tìm kiếm booking theo tên khách (khớp từng từ)"""
    results = []
    search_words = guest_name.lower().strip().split()
    
    for b in bookings:
        guest_full_name = str(b.get("guest_name", "")).lower().strip()
        if not guest_full_name:
            continue
            
        guest_words = guest_full_name.split()
        
        # Tất cả từ trong search phải có trong guest_words
        if all(search_word in guest_words for search_word in search_words):
            results.append(b)
    
    return sorted(results, key=lambda x: x.get("check_in", ""))


@mcp.tool()
def get_bookings_by_status(status: str) -> List[Dict]:
    """Lấy booking theo trạng thái (confirmed, paid, hold, reserved, etc.)"""
    results = [b for b in bookings if str(b.get("status", "")).strip().lower() == status.lower()]
    return sorted(results, key=lambda x: x.get("check_in", ""))

@mcp.tool()
def check_room_busy_periods(room_id: str) -> List[Dict]:
    """Kiểm tra các khoảng thời gian phòng bị chiếm"""
    busy_periods = []
    for b in bookings:
        if b["room_id"] == room_id and str(b.get("status","")).strip().lower() in BLOCK:
            busy_periods.append({
                "booking_id": b["booking_id"],
                "check_in": b["check_in"],
                "check_out": b["check_out"],
                "status": b["status"]
            })
    return sorted(busy_periods, key=lambda x: x["check_in"])



@mcp.tool() 
def hotel_summary() -> Dict:
    """Tổng quan về khách sạn"""
    # Room summary
    room_types = {}
    total_capacity = 0
    for r in rooms:
        room_type = r["type"]
        capacity = int(r["capacity"])
        room_types[room_type] = room_types.get(room_type, 0) + 1
        total_capacity += capacity
    
    # Booking summary
    status_counts = {}
    for b in bookings:
        status = b.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    
    return {
        "total_rooms": len(rooms),
        "total_capacity": total_capacity,
        "room_types": room_types,
        "total_bookings": len(bookings),
        "booking_status": status_counts
    }

# Initialize RAG system (Local Embeddings - FREE)
hotel_rag = HotelLocalRAG(
    vector_db_path=".hotel_vector_db",
    embedding_model_name="all-MiniLM-L6-v2",  # Free local model
)

@mcp.tool()
def query_hotel_docs(question: str, top_k: int = 3) -> List[Dict]:
    """Tìm kiếm thông tin trong tài liệu khách sạn"""
    return hotel_rag.query(question, k=top_k)

@mcp.tool()
def setup_hotel_documents(txt_folder: str = None, pdf_folder: str = None) -> Dict:
    """Setup tài liệu khách sạn từ thư mục TXT hoặc PDF"""
    results = {}
    if txt_folder:
        results["txt"] = hotel_rag.create_db_from_txt_folder(txt_folder)
    if pdf_folder:
        results["pdf"] = hotel_rag.create_db_from_pdf_folder(pdf_folder)
    return results

@mcp.tool()
def hotel_rag_summary() -> Dict:
    """Thông tin về hệ thống RAG"""
    return hotel_rag.get_summary()

def _get_room_types():
    return sorted({r["type"] for r in rooms})

def _check_availability(check_in: str, check_out: str, guests: int = 1, room_type: Optional[str] = None):
    type_filter, cap_hint = _parse_room_hint(room_type)
    try:
        guests = max(1, int(guests))
    except Exception:
        guests = 1
    if cap_hint:  # nếu user nói 'phòng đôi/ba/bốn' → ép guests tương ứng
        guests = max(guests, cap_hint)

    BLOCK = {"confirmed", "paid", "hold", "reserved"}
    def d(s: str): 
        return datetime.strptime(s, "%Y-%m-%d").date()
    def overlap(a1, a2, b1, b2):
        return a1 < b2 and b1 < a2

    ci, co = d(check_in), d(check_out)
    by_room: dict[str, list[tuple[date, date]]] = {}
    for b in bookings:
        if str(b.get("status","")).strip().lower() in BLOCK:
            by_room.setdefault(b["room_id"], []).append((d(b["check_in"]), d(b["check_out"])))

    avail = []
    for r in rooms:
        # capacity theo số người
        try:
            if int(r["capacity"]) < guests:
                continue
        except Exception:
            continue

        # nếu người dùng có yêu cầu hạng phòng (Standard/Deluxe/Family/Suite)
        if type_filter and _slug(r["type"]) != _slug(type_filter):
            continue

        busy = any(overlap(ci, co, s, e) for (s, e) in by_room.get(r["room_id"], []))
        if not busy:
            avail.append({
                "room_id": r["room_id"],
                "type": r["type"],
                "capacity": r["capacity"],
                "base_price": r["base_price"],
            })
    return avail

if __name__ == "__main__":
    import os, sys, logging
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("ohana")

    log.info("Starting MCP server...")  # -> stderr, an toàn
    mcp.run()  # chỉ dòng này là cần để chạy server