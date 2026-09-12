if __name__ == "__main__":
    pages = extract_pages_with_formatting("papers/Attention Is All You Need.pdf")
    body_size = get_body_font_size(pages)
    print(f"Detected body font size: {body_size}")

    for page in pages[:3]:
        print(f"\n--- Page {page['page_number']} ---")
        for line in page["lines"]:
            if is_heading(line, body_size):
                print(f"  HEADING: {line}")