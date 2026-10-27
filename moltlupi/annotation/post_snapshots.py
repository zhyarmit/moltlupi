import re

def candidates(text):
    sentences = {}
    start = 0
    for match in re.finditer('(?<=[.!?。！？])\\s+|\\n+', text):
        end = match.start()
        left = start
        while left < end and text[left].isspace():
            left += 1
        if left < end:
            sid = f's{len(sentences):04d}'
            sentences[sid] = dict(text=text[left:end], start=left, end=end)
        start = match.end()
    left = start
    while left < len(text) and text[left].isspace():
        left += 1
    end = len(text.rstrip())
    if left < end:
        sentences[f's{len(sentences):04d}'] = dict(text=text[left:end], start=left, end=end)
    spans = set()
    patterns = ['(?<!\\w)[A-Z][\\w]*', '(?:https?://|ipfs://|www\\.)[^\\s<>"\\x27]+', '(?<!\\w)(?:mbc-20|A2A|MCP)(?!\\w)', '(?<!\\w)[\\w]+(?:[-./][\\w]+)+', '(?<!\\w)[A-Z][\\w]*(?:[ \\t]+[A-Z][\\w]*){0,4}', '(?i)(?:protocol|skill|product|project|tool|service)[ \\t]+(?:named[ \\t]+|called[ \\t]+)?([\\w.-]+)', '`([^`\\n]+)`', '"([^"\\n]{1,100})"']
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            spans.add(m.span(1) if m.lastindex else m.span())
    artifacts = {}
    seen = set()
    for start, end in sorted(spans):
        value = text[start:end]
        if value not in seen:
            artifacts[f'a{len(artifacts):04d}'] = dict(text=value, start=start, end=end)
            seen.add(value)
    return (artifacts, sentences)
