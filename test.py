from mcp_server.hotel_local_rag import setup_hotel_rag
import os

# Create documents folder if it doesn't exist
os.makedirs("hotel_docs", exist_ok=True)

# Create a basic policy file
with open("hotel_docs/noi_quy.txt", "w", encoding="utf-8") as f:
    f.write("""NỘI QUY KHÁCH SẠN OHANA

1. THỜI GIAN CHECK-IN/CHECK-OUT
- Check-in: 14:00
- Check-out: 12:00

2. QUY ĐỊNH TRONG PHÒNG
- Không hút thuốc trong phòng
- Giữ yên lặng từ 22:00 - 06:00

3. THANH TOÁN & HỦY PHÒNG
- Miễn phí hủy trước 24h
- Chấp nhận tiền mặt, thẻ tín dụng

4. LIÊN HỆ
- Lễ tân: 24/7
- Hotline: 1900-OHANA""")

print("Created hotel_docs/noi_quy.txt")

# Setup RAG system
rag = setup_hotel_rag(
    txt_folder="hotel_docs",
    vector_db_path=".hotel_vector_db"
)

if rag and rag.is_ready():
    print("✅ RAG system setup successfully!")
    result = rag.query("nội quy khách sạn")
    print("Test query result:", result)
else:
    print("❌ RAG setup failed")