import hashlib
import re
import xml.etree.ElementTree as ET


def Element(parent: ET.Element | None, tag: str, text: str | None = None, **attrs) -> ET.Element:
    """Create an XML element with case-preserved attributes and optional text content."""
    attrs = {k.replace("_", "-"): format_float(v) if isinstance(v, float) else str(v) for k, v in attrs.items()}
    elem = ET.SubElement(parent, tag, attrs) if parent is not None else ET.Element(tag, attrs)
    if text is not None:
        elem.text = text
    return elem


def format_float(v: float):
    return f"{v:.4g}"


_LOCAL_REF = re.compile(r"url\(#([^)]*)\)")


def content_tag(root: ET.Element, length: int = 8) -> str:
    """A short digest of *root*'s markup, for :func:`stamp_ids` to qualify its ids with."""
    return hashlib.sha256(ET.tostring(root, encoding="unicode").encode()).hexdigest()[:length]


def stamp_ids(root: ET.Element, tag: str) -> None:
    """Append ``-{tag}`` to every ``id`` in *root*, and to the ``url(#…)`` references that reach them.

    An inline SVG shares the host page's id namespace, so a document holding several of
    them needs each one's defs named apart. Taking *tag* from the finished markup
    (:func:`content_tag`) does that without drawing on randomness, which is what lets the
    same figure render the same bytes on every run — so an export only changes when the
    figure does, and two renders of one report can be compared directly.
    """
    for el in root.iter():
        if (ident := el.get("id")) is not None:
            el.set("id", f"{ident}-{tag}")
        for key, value in list(el.items()):
            if "url(#" in value:
                el.set(key, _LOCAL_REF.sub(lambda m: f"url(#{m[1]}-{tag})", value))
