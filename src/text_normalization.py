import re

def time_to_seconds(t_str: str) -> float:
    """Converts timestamp format (HH:MM:SS.mmm or HH:MM:SS,mmm) to float seconds."""
    t_str = t_str.replace(",", ".")
    parts = t_str.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return float(h) * 3600 + float(m) * 60 + float(s)
    return 0.0


def parse_subtitles(filepath: str):
    """Parses VTT or SRT caption files into a list of (start_time, end_time, text) tuples.
    
    Skips exact-duplicate consecutive caption blocks defensively.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to match caption blocks with timestamps: e.g., 00:00:01.120 --> 00:00:04.500
    # Captures text until the next timestamp or a double newline.
    pattern = r"(\d{2}:\d{2}:\d{2}[\.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[\.,]\d{3})\r?\n(.*?)(?=\n\d{2}:\d{2}:\d{2}[\.,]\d{3}|\r?\n\r?\n|\Z)"
    matches = re.finditer(pattern, content, re.DOTALL)

    raw_segments = []
    for match in matches:
        start_str, end_str, text_block = match.groups()
        
        start_sec = time_to_seconds(start_str)
        end_sec = time_to_seconds(end_str)
        
        # Strip webvtt inline styling tags (e.g. <i>, <c>, <c.color>)
        text_block = re.sub(r"<[^>]+>", "", text_block)
        lines = [line.strip() for line in text_block.split('\n') if line.strip()]
        text = " ".join(lines).strip()
        
        if text:
            raw_segments.append((start_sec, end_sec, text))

    # Filter out exact-duplicate consecutive caption blocks
    cleaned_segments = []
    for start, end, text in raw_segments:
        if cleaned_segments and cleaned_segments[-1][2] == text:
            continue
        cleaned_segments.append((start, end, text))

    return cleaned_segments


def normalize_azerbaijani(text: str) -> str:
    """Applies Azerbaijani-specific ASR text normalization:
    
    1. Turkic case-folding: maps 'İ' -> 'i' and 'I' -> 'ı' before low-casing.
    2. Strips punctuation but preserves Azerbaijani-specific letters (ə ğ ı ö ü ş ç) and digits.
    3. Collapses multiple whitespace.
    """
    # 1. Turkic case-folding mapping
    text = text.replace("İ", "i").replace("I", "ı")
    text = text.lower()
    
    # 2. Strip punctuation but preserve letters, numbers, and spaces
    # Alphanumeric character matching (\w) handles unicode letters in Python
    text = re.sub(r"[^\w\s]|_", " ", text)
    
    # 3. Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text
