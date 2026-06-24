import hashlib


def content_hash(*contents: str) -> str:
    """Short stable hash of several strings, e.g. for naming experiments."""
    combined = "&?!@&".join(contents)
    return hashlib.md5(combined.encode('utf-8')).hexdigest()[:8]
