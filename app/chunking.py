from pathlib import Path


def load_document(file_path: str) -> str:
    path = Path(file_path)
    return path.read_text(encoding="utf-8")


def parse_markdown_sections(text: str) -> list[dict]:
    sections = []

    current_title = None
    current_content = []

    for line in text.splitlines():
        line = line.strip()

        # Ignore empty lines
        if not line:
            continue

        # Document title
        if line.startswith("# "):
            continue

        # Section heading
        if line.startswith("## "):

            # Save previous section
            if current_title is not None:
                sections.append({
                    "title": current_title,
                    "content": " ".join(current_content)
                })

            # Start new section
            current_title = line[3:].strip()
            current_content = []

        else:
            current_content.append(line)

    # Save the last section
    if current_title is not None:
        sections.append({
            "title": current_title,
            "content": " ".join(current_content)
        })

    return sections


def create_chunks(
    sections: list[dict],
    chunk_size: int = 300,
    overlap_sentences: int = 1
) -> list[dict]:

    chunks = []

    for section in sections:

        sentences = [
            sentence.strip()
            for sentence in section["content"].split(". ")
            if sentence.strip()
        ]

        current_sentences = []

        for sentence in sentences:

            candidate = ". ".join(
                current_sentences + [sentence]
            )

            if len(candidate) <= chunk_size:
                current_sentences.append(sentence)

            else:

                if current_sentences:
                    chunks.append({
                        "text": ". ".join(current_sentences),
                        "metadata": {
                            "document": "company_handbook.md",
                            "section": section["title"]
                        }
                    })

                # Keep the last sentence for overlap
                overlap = current_sentences[-overlap_sentences:]

                current_sentences = overlap + [sentence]

        # Add remaining sentences
        if current_sentences:
            chunks.append({
                "text": ". ".join(current_sentences),
                "metadata": {
                    "document": "company_handbook.md",
                    "section": section["title"]
                }
            })

    return chunks


# --------------------------------------------------
# Main
# --------------------------------------------------

document = load_document(
    "data/company_handbook.md"
)

sections = parse_markdown_sections(document)

chunks = create_chunks(
    sections,
    chunk_size=300,
    overlap_sentences=1
)

for index, chunk in enumerate(chunks):

    print(f"\n--- Chunk {index + 1} ---")

    print(f"Document: {chunk['metadata']['document']}")

    print(
        f"Section: {chunk['metadata']['section']}"
    )

    print(
        f"Text: {chunk['text']}"
    )