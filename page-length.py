import fitz

def extract_text_by_page(pdf_path):
    doc = fitz.open(pdf_path)
    page_info = {}

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text()
        page_info[page_num + 1] = page_text

    return page_info

# Example usage
pdf_path = r'C:\\Users\\Utki\\Desktop\\code\\project\\apug data\\MATH136-notes.pdf'
pages = extract_text_by_page(pdf_path)
total_wc = 0
max_wc = 0
for i in pages:
    total_wc += len(pages[i])
    if max_wc < len(pages[i]):
        max_wc = len(pages[i])
average_wc = total_wc / len(pages)
print(max_wc, average_wc)