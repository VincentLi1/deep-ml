import re

SPLIT_CHARS = [' ', '\t', '\n', ',', '.', ':', ';', '?', '_', '!', '(', ')', '--', "'"]

def tokenize_text(text: str) -> list:
    whitespace = [' ', '\t', '\n']
    punctuation = [s for s in SPLIT_CHARS if s not in whitespace]

    pattern = r'\s+|(' + '|'.join(
        re.escape(s) for s in sorted(punctuation, key=len, reverse=True)
    ) + r')'

    tokens = re.split(pattern, text)

    return [t.strip() for t in tokens if t and t.strip()]