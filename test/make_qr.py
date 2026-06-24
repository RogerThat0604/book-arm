import qrcode
import json

books = [
    {
        "id": "BOOK_001",
        "title": "데미안",
        "category": "문학"
    },
    {
        "id": "BOOK_002",
        "title": "코스모스",
        "category": "과학"
    }
]

for book in books:
    qr_data = json.dumps(book, ensure_ascii=False)

    img = qrcode.make(qr_data)

    file_name = f"{book['id']}_{book['title']}.png"
    img.save(file_name)

    print(f"저장 완료: {file_name}")