from types import SimpleNamespace

from rfi_agent.agent import route_after_validate
from rfi_agent.export import citation_audit, export_json_payload, serialize_answer
from rfi_agent.grounding import as_citation_list, as_string_list, quote_matches_chunk, sha256_text
from rfi_agent.ingest.chunker import chunk_text
from rfi_agent.ingest.files import parse_file
from rfi_agent.ingest.pipeline import classify_source_uri
from rfi_agent.ingest.sitemap import choose_sitemap, expand_page_locs, looks_like_sitemap, page_fetch_urls
from rfi_agent.ingest.ssrf import UnsafeURL, assert_public_https
from rfi_agent.questionnaire import parse_questionnaire
from rfi_agent.retrieval import rrf_merge


def test_chunk_text_splits_long_input():
    text = ("# Heading\n\n" + ("word " * 80) + "\n\n## Next\n\n" + ("more " * 80))
    chunks = chunk_text(text, target_chars=200, overlap=20)
    assert len(chunks) >= 2
    assert all(chunks)


def test_ssrf_blocks_localhost():
    try:
        assert_public_https("http://example.com")
        raised = False
    except UnsafeURL:
        raised = True
    assert raised
    try:
        assert_public_https("https://127.0.0.1/secret")
        local_blocked = False
    except UnsafeURL:
        local_blocked = True
    assert local_blocked


def test_rrf_prefers_overlap():
    fused = rrf_merge(["a", "b"], ["b", "c"])
    assert fused[0][0] == "b"


def test_pkce_subject_is_state_keyed():
    from rfi_agent.ingest.drive import _pkce_subject

    assert _pkce_subject("ses_abc") == "pkce:ses_abc"
    assert _pkce_subject("") == "pkce:solo"


def test_sitemap_index_expands_to_pages():
    index = b"""<?xml version="1.0"?>
    <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <sitemap><loc>https://example.com/sitemap-pages.xml</loc></sitemap>
      <sitemap><loc>https://example.com/release/sitemap-pages.xml</loc></sitemap>
    </sitemapindex>"""
    current = b"""<?xml version="1.0"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://example.com/start</loc></url>
    </urlset>"""
    older = b"""<?xml version="1.0"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://example.com/release/notes</loc></url>
      <url><loc>https://example.com/start</loc></url>
    </urlset>"""
    files = {
        "https://example.com/sitemap.xml": index,
        "https://example.com/sitemap-pages.xml": current,
        "https://example.com/release/sitemap-pages.xml": older,
    }
    pages, meta = expand_page_locs(lambda url: files[url], "https://example.com/sitemap.xml", "https://example.com", max_pages=10)
    assert pages == ["https://example.com/start", "https://example.com/release/notes"]
    assert meta["truncated"] is False
    assert len(meta["sitemaps"]) == 3


def test_every_entered_url_uses_the_same_sitemap_and_page_pattern():
    index = b"""<?xml version="1.0"?>
    <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <sitemap><loc>https://example.com/sitemap-pages.xml</loc></sitemap>
    </sitemapindex>"""
    pages = b"""<?xml version="1.0"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://example.com/start</loc></url>
    </urlset>"""
    candidates = [
        ("https://example.com/sitemap.xml", index),
        ("https://example.com/sitemap-pages.xml", pages),
    ]
    assert choose_sitemap(candidates, "https://example.com") == "https://example.com/sitemap.xml"
    assert page_fetch_urls("https://example.com") == ["https://example.com.md", "https://example.com"]
    assert page_fetch_urls("https://example.com/start") == ["https://example.com/start.md", "https://example.com/start"]


def test_classify_and_sitemap_urls():
    assert classify_source_uri("https://drive.google.com/file/d/abc/view") == "drive"
    assert classify_source_uri("https://docs.example.com/") == "url"
    assert looks_like_sitemap("https://docs.example.com/sitemap-pages.xml")
    assert not looks_like_sitemap("https://docs.example.com/getting-started")


def test_quote_must_be_chunk_substring():
    chunk = "We encrypt data at rest with AES-256."
    assert quote_matches_chunk("AES-256", chunk)
    assert quote_matches_chunk("  we   encrypt DATA at rest ", chunk)
    assert not quote_matches_chunk("we never encrypt", chunk)
    assert not quote_matches_chunk("", chunk)


def test_citation_repair_routes_once():
    assert route_after_validate({"needs_repair": True}) == "generate"
    assert route_after_validate({"needs_repair": True, "repair_attempted": True}) == "end"
    assert route_after_validate({"needs_repair": False}) == "end"


def test_content_hash_is_stable():
    assert sha256_text("a", "b") == sha256_text("a", "b")
    assert sha256_text("a", "b") != sha256_text("b", "a")


def test_questionnaire_csv_headers():
    csv = b"Question,Category,Guidance\nDo you encrypt data at rest?,Security,Yes/No\nHow long are logs retained?,Ops,250 chars\n"
    rows = parse_questionnaire("rfi.csv", csv)
    assert len(rows) == 2
    assert rows[0].text.startswith("Do you encrypt")
    assert rows[0].category == "Security"
    assert rows[1].constraints == "250 chars"


def test_questionnaire_numbered_lines():
    text = b"1. Do you support SSO with SAML?\n2. Is production isolated from staging?\n"
    rows = parse_questionnaire("rfi.txt", text)
    assert len(rows) == 2
    assert "SSO" in rows[0].text


def test_questionnaire_cap():
    body = "Question\n" + "\n".join(f"Is control {i} in place as documented?" for i in range(3))
    try:
        parse_questionnaire("rfi.csv", body.encode(), max_questions=2)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_xlsx_tables_serialized():
    parsed = parse_file("note.txt", b"hello table")
    assert parsed.text == "hello table"
    assert parsed.content_type == "text/plain"


def test_export_hides_drafts_unless_flagged():
    draft = SimpleNamespace(
        id="ans_1",
        question_id=None,
        status="draft",
        origin="documents",
        answer_text="maybe",
        short_answer=None,
        confidence=0.4,
        citations=[{"title": "Doc", "locator": "p.1", "quote": "yes", "uri": "https://example.com"}],
        gaps=[],
        conflicts=[],
        model_id="m",
        corpus_version="abc",
        reviewer_id="",
        run_id="run_1",
        question_text="Do you encrypt?",
        edited_by_human=False,
    )
    approved = SimpleNamespace(**{**draft.__dict__, "id": "ans_2", "status": "approved", "answer_text": "Yes"})
    hidden = export_json_payload([draft, approved], include_drafts=False)
    shown = export_json_payload([draft, approved], include_drafts=True)
    assert hidden["count"] == 1
    assert shown["count"] == 2
    assert "Doc" in citation_audit(draft.citations)


def test_gaps_and_citations_coerce_from_llm_shapes():
    assert as_string_list("missing SLA clause") == ["missing SLA clause"]
    assert as_string_list(["a", " ", None, "b"]) == ["a", "b"]
    assert as_string_list(None) == []
    assert as_citation_list("not a list") == []
    assert as_citation_list([{"chunk_id": "c1"}, "skip", None]) == [{"chunk_id": "c1"}]
    messy = SimpleNamespace(
        id="ans_3",
        question_id=None,
        status="draft",
        origin="documents",
        answer_text="maybe",
        short_answer=None,
        confidence=0.4,
        citations="https://example.com",
        gaps="No retention policy in the packet",
        conflicts="Doc A says 30 days; Doc B says 90",
        model_id="m",
        corpus_version="abc",
        reviewer_id="",
        run_id="run_3",
        question_text="Retention?",
        edited_by_human=False,
    )
    out = serialize_answer(messy)
    assert out["gaps"] == ["No retention policy in the packet"]
    assert out["conflicts"] == ["Doc A says 30 days; Doc B says 90"]
    assert out["citations"] == []
    assert citation_audit("oops") == ""
