from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from io import BytesIO

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from sqlalchemy.orm import Session as DBSession

from rfi_agent.config import get_settings
from rfi_agent.ids import new_id
from rfi_agent.models import OAuthToken

os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")
os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
PKCE_COOKIE = "rfi_google_pkce"

FILE_RE = re.compile(r"/file/d/([a-zA-Z0-9_-]+)")
FOLDER_RE = re.compile(r"/folders/([a-zA-Z0-9_-]+)")
DOC_RE = re.compile(r"/document/d/([a-zA-Z0-9_-]+)")
SHEET_RE = re.compile(r"/spreadsheets/d/([a-zA-Z0-9_-]+)")
SLIDE_RE = re.compile(r"/presentation/d/([a-zA-Z0-9_-]+)")


def google_configured() -> bool:
    s = get_settings()
    return bool(s.google_client_id.strip() and s.google_client_secret.strip())


def oauth_start_url() -> str:
    from urllib.parse import urlsplit

    parts = urlsplit(get_settings().google_redirect_uri)
    origin = f"{parts.scheme}://{parts.netloc}" if parts.scheme and parts.netloc else "http://localhost:8000"
    return f"{origin}/oauth/google/start"


def build_flow(*, autogenerate_code_verifier: bool = True) -> Flow:
    s = get_settings()
    return Flow.from_client_config(
        {
            "web": {
                "client_id": s.google_client_id.strip(),
                "client_secret": s.google_client_secret.strip(),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [s.google_redirect_uri.strip()],
            }
        },
        scopes=SCOPES,
        redirect_uri=s.google_redirect_uri.strip(),
        autogenerate_code_verifier=autogenerate_code_verifier,
    )


def _pkce_subject(state: str) -> str:
    return f"pkce:{state or 'solo'}"


def _store_pkce(db: DBSession, state: str, code_verifier: str) -> None:
    subject = _pkce_subject(state)
    existing = db.query(OAuthToken).filter_by(provider="google", subject=subject).first()
    payload = {"code_verifier": code_verifier, "state": state}
    if existing:
        existing.access_token = "pending"
        existing.token_json = payload
        return
    db.add(
        OAuthToken(
            id=new_id("tok"),
            provider="google",
            subject=subject,
            access_token="pending",
            token_json=payload,
        )
    )


def _load_pkce(db: DBSession, state: str) -> str | None:
    row = db.query(OAuthToken).filter_by(provider="google", subject=_pkce_subject(state)).first()
    if row and isinstance(row.token_json, dict):
        verifier = row.token_json.get("code_verifier")
        if isinstance(verifier, str) and verifier:
            return verifier
    pending = (
        db.query(OAuthToken)
        .filter(OAuthToken.provider == "google", OAuthToken.subject.like("pkce:%"), OAuthToken.access_token == "pending")
        .order_by(OAuthToken.id.desc())
        .first()
    )
    if pending and isinstance(pending.token_json, dict):
        verifier = pending.token_json.get("code_verifier")
        if isinstance(verifier, str) and verifier:
            return verifier
    return None


def begin_google_login(db: DBSession, session_id: str) -> tuple[str, str]:
    flow = build_flow(autogenerate_code_verifier=True)
    url, _state = flow.authorization_url(
        access_type="offline",
        prompt="consent",
        state=session_id or "solo",
    )
    verifier = getattr(flow, "code_verifier", None) or ""
    if not verifier:
        raise RuntimeError("Google OAuth did not produce a PKCE code_verifier.")
    _store_pkce(db, session_id or "solo", verifier)
    return url, verifier


def authorization_url(db: DBSession, session_id: str) -> str:
    url, _verifier = begin_google_login(db, session_id)
    return url


def exchange_code(db: DBSession, code: str, state: str = "", code_verifier: str | None = None) -> OAuthToken:
    verifier = (code_verifier or "").strip() or _load_pkce(db, state)
    if not verifier:
        raise RuntimeError("Google OAuth is missing the PKCE code verifier. Click Connect Google Drive again.")
    flow = build_flow(autogenerate_code_verifier=False)
    flow.code_verifier = verifier
    flow.fetch_token(code=code, code_verifier=verifier)
    creds = flow.credentials
    pending = db.query(OAuthToken).filter_by(provider="google", subject=_pkce_subject(state)).first()
    if pending and pending.access_token == "pending":
        db.delete(pending)
    existing = db.query(OAuthToken).filter_by(provider="google", subject="solo").first()
    payload = {
        "token": creds.token,
        "refresh_token": creds.refresh_token or "",
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes or SCOPES),
        "expiry": creds.expiry.isoformat() if creds.expiry else None,
    }
    expires_at = creds.expiry.replace(tzinfo=timezone.utc) if creds.expiry else None
    if existing:
        existing.access_token = creds.token or ""
        existing.refresh_token = creds.refresh_token or existing.refresh_token
        existing.token_json = payload
        existing.expires_at = expires_at
        return existing
    row = OAuthToken(
        id=new_id("tok"),
        provider="google",
        subject="solo",
        access_token=creds.token or "",
        refresh_token=creds.refresh_token or "",
        token_json=payload,
        expires_at=expires_at,
    )
    db.add(row)
    return row


def load_credentials(db: DBSession) -> Credentials | None:
    row = db.query(OAuthToken).filter_by(provider="google", subject="solo").first()
    if not row or row.access_token in {"", "pending"}:
        return None
    s = get_settings()
    info = dict(row.token_json or {})
    if not info.get("token"):
        return None
    info.setdefault("client_id", s.google_client_id.strip())
    info.setdefault("client_secret", s.google_client_secret.strip())
    info.setdefault("token_uri", "https://oauth2.googleapis.com/token")
    try:
        creds = Credentials.from_authorized_user_info(info, SCOPES)
    except (ValueError, KeyError):
        return None
    try:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            row.access_token = creds.token or row.access_token
            row.token_json = {
                **info,
                "token": creds.token,
                "expiry": creds.expiry.isoformat() if creds.expiry else None,
            }
    except Exception:
        return creds
    return creds


def parse_drive_ref(url: str) -> tuple[str, str]:
    for regex, kind in (
        (FOLDER_RE, "folder"),
        (FILE_RE, "file"),
        (DOC_RE, "gdoc"),
        (SHEET_RE, "gsheet"),
        (SLIDE_RE, "gslides"),
    ):
        match = regex.search(url)
        if match:
            return kind, match.group(1)
    raise ValueError("Could not parse a Google Drive file or folder id from that URL.")


def change_feed_moved(payload: dict) -> bool:
    return bool(payload.get("changes"))


def change_feed_token(payload: dict, current: str) -> str:
    token = payload.get("newStartPageToken") or payload.get("nextPageToken") or current
    return str(token or current)


def start_change_token(creds: Credentials) -> str:
    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    token = service.changes().getStartPageToken(supportsAllDrives=True).execute().get("startPageToken")
    return str(token or "")


def folder_changed(creds: Credentials, page_token: str) -> tuple[bool, str]:
    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    payload = (
        service.changes()
        .list(
            pageToken=page_token,
            pageSize=20,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            fields="newStartPageToken,nextPageToken,changes(fileId,removed)",
        )
        .execute()
    )
    return change_feed_moved(payload), change_feed_token(payload, page_token)


def download_drive_items(creds: Credentials, url: str, max_files: int = 200) -> list[tuple[str, bytes]]:
    kind, file_id = parse_drive_ref(url)
    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    if kind == "folder":
        files = _list_folder(service, file_id, max_files)
        return [_download_one(service, f) for f in files]
    meta = service.files().get(fileId=file_id, fields="id,name,mimeType", supportsAllDrives=True).execute()
    return [_download_one(service, meta)]


def _list_folder(service, folder_id: str, max_files: int) -> list[dict]:
    out: list[dict] = []
    page_token = None
    while len(out) < max_files:
        resp = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="nextPageToken, files(id,name,mimeType)",
                pageToken=page_token,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
                pageSize=100,
            )
            .execute()
        )
        for f in resp.get("files", []):
            if f.get("mimeType") == "application/vnd.google-apps.folder":
                out.extend(_list_folder(service, f["id"], max_files - len(out)))
            else:
                out.append(f)
            if len(out) >= max_files:
                break
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return out[:max_files]


def _download_one(service, meta: dict) -> tuple[str, bytes]:
    mime = meta.get("mimeType") or ""
    name = meta.get("name") or meta["id"]
    export_map = {
        "application/vnd.google-apps.document": ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx"),
        "application/vnd.google-apps.spreadsheet": ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx"),
        "application/vnd.google-apps.presentation": ("application/pdf", ".pdf"),
    }
    fh = BytesIO()
    if mime in export_map:
        export_mime, ext = export_map[mime]
        if not name.lower().endswith(ext):
            name = name + ext
        request = service.files().export_media(fileId=meta["id"], mimeType=export_mime)
    else:
        request = service.files().get_media(fileId=meta["id"], supportsAllDrives=True)
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return name, fh.getvalue()
