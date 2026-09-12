import tiktoken
import json


def split_into_sections(lines, body_size):
    """
    Splits a page's lines into segments, each tagged with the section
    that was active at that point, using font-based heading detection.
    Returns: list of (section_name, segment_text) tuples.
    """
    from extract import is_heading

    segments = []
    current_section = None
    current_lines = []

    for line_info in lines:
        if is_heading(line_info, body_size):
            # save whatever we've accumulated under the previous section
            if current_lines:
                segments.append((current_section, "\n".join(current_lines)))
            current_section = line_info["text"]
            current_lines = []  # don't include the heading line itself in the body text
        else:
            current_lines.append(line_info["text"])

    # save the final segment after the loop ends
    if current_lines:
        segments.append((current_section, "\n".join(current_lines)))

    return segments


def count_tokens(text, encoding_name="cl100k_base"):
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def chunk_text_by_tokens(text, max_tokens=400, overlap_tokens=60, encoding_name="cl100k_base"):
    """
    Split a block of text into overlapping chunks based on token count.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)

        if end == len(tokens):
            break
        start = end - overlap_tokens  # step back for overlap

    return chunks


def chunk_paper(paper_title, pages, body_size, max_tokens=400, overlap_tokens=60):
    """
    Takes a paper's formatted pages (from extract_pages_with_formatting) and produces
    a list of chunk dicts with metadata, using font-based section detection.
    """
    all_chunks = []
    current_section = "Unknown"  # carries over across pages
    chunk_counter = 0

    for page in pages:
        page_num = page["page_number"]
        lines = page["lines"]

        segments = split_into_sections(lines, body_size)

        for section, segment_text in segments:
            if section is not None:
                current_section = section  # update running section tracker

            if len(segment_text.strip()) < 20:
                continue  # skip empty/near-empty segments

            segment_chunks = chunk_text_by_tokens(segment_text, max_tokens, overlap_tokens)

            for chunk in segment_chunks:
                if len(chunk.strip()) < 20:
                    continue
                chunk_counter += 1
                all_chunks.append({
                    "chunk_id": f"{paper_title.replace(' ', '_')}_{chunk_counter:03d}",
                    "paper_title": paper_title,
                    "section": current_section,
                    "page_number": page_num,
                    "text": chunk.strip()
                })

    return all_chunks


def chunk_all_papers(all_papers_dict, max_tokens=400, overlap_tokens=60):
    """
    all_papers_dict: {paper_title: {"pages": [...], "body_size": float}, ...}
    Returns: list of all chunks across all papers
    """
    all_chunks = []
    for paper_title, data in all_papers_dict.items():
        chunks = chunk_paper(paper_title, data["pages"], data["body_size"], max_tokens, overlap_tokens)
        all_chunks.extend(chunks)
        print(f"{paper_title}: {len(chunks)} chunks created")
    return all_chunks


if __name__ == "__main__":
    from extract import extract_all_papers_with_formatting

    papers = extract_all_papers_with_formatting()
    chunks = chunk_all_papers(papers)

    print(f"\nTotal chunks across all papers: {len(chunks)}")

    # print first 8 chunks in order, to visually verify section labeling looks correct
    print("\n--- First 8 chunks (sequence check) ---")
    for c in chunks[:8]:
        print(f"[{c['chunk_id']}] Section: {c['section']} | Page: {c['page_number']}")
        print(f"  {c['text'][:100]}...\n")

    # Save to disk so downstream steps (embedding) don't need to re-run extraction+chunking
    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(chunks)} chunks to chunks.json")