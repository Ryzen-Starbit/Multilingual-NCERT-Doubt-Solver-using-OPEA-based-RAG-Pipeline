def explain(docs):
    """
    Returns chapter-level NCERT citations.
    """
    seen = set()
    citations = []
    for d in docs:
        m = d.metadata
        key = (m["grade"], m["subject"], m["book"], m["source"])
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            f"Class {m['grade']} | "
            f"{m['subject'].title()} | "
            f"{m['book'].replace('_',' ').title()} | "
            f"{m['chapter']} | "
            f"{m['source']}"
        )
    return citations[:3]
