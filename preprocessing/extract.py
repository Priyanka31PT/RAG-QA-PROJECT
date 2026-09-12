import pymupdf as fitz
import os
from collections import Counter


def extract_pages_with_formatting(pdf_path):
    """
    Extract text line-by-line with font size and bold info, per page.
    Returns: list of dicts: [{"page_number": 1, "lines": [{"text":.., "size":.., "bold":..}, ...]}, ...]
    """
    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_dict = page.get_text("dict")
        lines_data = []

        for block in page_dict["blocks"]:
            if "lines" not in block:
                continue  # skip image blocks etc.
            for line in block["lines"]:
                line_text = ""
                max_size = 0
                is_bold = False

                for span in line["spans"]:
                    line_text += span["text"]
                    max_size = max(max_size, span["size"])
                    if "bold" in span["font"].lower():
                        is_bold = True

                line_text = line_text.strip()
                if line_text:
                    lines_data.append({
                        "text": line_text,
                        "size": round(max_size, 1),
                        "bold": is_bold
                    })

        pages.append({
            "page_number": page_num + 1,  # human-friendly, 1-indexed
            "lines": lines_data
        })

    doc.close()
    return pages


def get_body_font_size(pages):
    """
    Determine the most common font size across all lines in a paper — this is the body text size.
    """
    sizes = []
    for page in pages:
        for line in page["lines"]:
            sizes.append(line["size"])

    if not sizes:
        return 10.0  # fallback default, shouldn't normally happen

    most_common_size = Counter(sizes).most_common(1)[0][0]
    return most_common_size


def is_heading(line_info, body_size, size_threshold=1.5, max_heading_length=80):
    """
    Determine if a line is likely a heading based on font size/boldness,
    rather than matching against a fixed keyword list.
    """
    text = line_info["text"]
    size = line_info["size"]
    bold = line_info["bold"]

    if len(text) > max_heading_length:
        return False  # headings are short; long lines are body text even if bold

    if size >= body_size + size_threshold:
        return True  # noticeably larger than body text

    if bold and size >= body_size:
        return True  # bold and at least body-size (catches bold headings same size as body)

    return False


def extract_all_papers_with_formatting(papers_folder="papers"):
    """
    Loop through all PDFs in the papers folder, extracting formatted lines
    (with font size/bold info) plus the detected body font size per paper.
    Returns: {paper_title: {"pages": [...], "body_size": float}, ...}
    """
    all_papers = {}

    for filename in os.listdir(papers_folder):
        if filename.lower().endswith(".pdf"):
            paper_title = os.path.splitext(filename)[0]
            pdf_path = os.path.join(papers_folder, filename)
            print(f"Extracting: {paper_title}")

            pages = extract_pages_with_formatting(pdf_path)
            body_size = get_body_font_size(pages)

            all_papers[paper_title] = {
                "pages": pages,
                "body_size": body_size
            }

    return all_papers


if __name__ == "__main__":
    # Quick manual test — run this file directly to sanity-check heading detection
    papers = extract_all_papers_with_formatting()

    for title, data in papers.items():
        print(f"\n--- {title} ---")
        print(f"Detected body font size: {data['body_size']}")
        print(f"Total pages: {len(data['pages'])}")

        # show detected headings on first 3 pages, as a sanity check
        for page in data["pages"][:3]:
            for line in page["lines"]:
                if is_heading(line, data["body_size"]):
                    print(f"  [Page {page['page_number']}] HEADING: {line['text']}")