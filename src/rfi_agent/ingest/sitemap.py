from __future__ import annotations

from urllib.parse import urlparse

from rfi_agent.ingest.ssrf import UnsafeURL, assert_public_https


def parse_sitemap_locs(xml_bytes: bytes, origin: str) -> list[str]:
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_bytes)
    locs = [el.text.strip() for el in root.findall(".//{*}loc") if el.text]
    keep = []
    for loc in locs:
        try:
            if assert_public_https(loc) and loc.startswith(origin):
                keep.append(loc)
        except UnsafeURL:
            continue
    return keep


def is_sitemap_index(xml_bytes: bytes) -> bool:
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_bytes)
    return root.tag.rsplit("}", 1)[-1].lower() == "sitemapindex"


def expand_page_locs(fetch_xml, start_url: str, origin: str, *, max_pages: int, max_sitemaps: int = 25) -> tuple[list[str], dict]:
    """Follow sitemap indexes and return page URLs, not child sitemap URLs."""
    queue = [start_url]
    seen_maps: set[str] = set()
    pages: list[str] = []
    seen_pages: set[str] = set()
    truncated = False
    while queue and len(seen_maps) < max_sitemaps:
        current = queue.pop(0)
        if current in seen_maps:
            continue
        seen_maps.add(current)
        xml_bytes = fetch_xml(current)
        locs = parse_sitemap_locs(xml_bytes, origin)
        nested = is_sitemap_index(xml_bytes)
        for loc in locs:
            if nested or looks_like_sitemap(loc):
                if loc not in seen_maps:
                    queue.append(loc)
                continue
            if loc in seen_pages:
                continue
            if len(pages) >= max_pages:
                truncated = True
                break
            seen_pages.add(loc)
            pages.append(loc)
        if truncated:
            break
    return pages, {
        "sitemaps": list(seen_maps),
        "truncated": truncated,
        "listed_pages": len(pages),
    }


def title_from_loc(loc: str) -> str:
    path = loc.rstrip("/").split("/")[-1]
    return path.replace("-", " ").replace("_", " ") if path else urlparse(loc).netloc


def origin_of(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def looks_like_sitemap(url: str) -> bool:
    path = urlparse(url).path.lower()
    return "sitemap" in path and path.endswith(".xml")


def page_fetch_urls(url: str) -> list[str]:
    """Every page is fetched the same way: GitBook markdown first, then the URL itself."""
    bare = url.split("?", 1)[0].split("#", 1)[0].rstrip("/")
    ordered: list[str] = []
    if not bare.lower().endswith(".md"):
        ordered.append(f"{bare}.md")
    if url not in ordered:
        ordered.append(url)
    return ordered


def choose_sitemap(candidates: list[tuple[str, bytes | None]], origin: str) -> str | None:
    """Pick one sitemap for any URL on a host.

    Priority order is the caller's. A sitemap index wins over a single urlset,
    so a site root, a page, and a child sitemap file all expand the same index.
    """
    urlset: str | None = None
    for url, body in candidates:
        if not body:
            continue
        try:
            locs = parse_sitemap_locs(body, origin)
        except Exception:
            continue
        if not locs:
            continue
        try:
            indexed = is_sitemap_index(body)
        except Exception:
            continue
        if indexed:
            return url
        if urlset is None:
            urlset = url
    return urlset
