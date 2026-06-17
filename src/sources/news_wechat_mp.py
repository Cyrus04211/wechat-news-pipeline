from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from src.sources.news_base import NewsSourceAdapter, RawNewsItem
from src.sources.wechat_article_content import (
    is_fetchable_article_type,
    is_valid_wechat_article_url,
    normalize_wechat_article_url,
)

logger = logging.getLogger(__name__)
UTC = timezone.utc

MP_BASE = "https://mp.weixin.qq.com"

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://mp.weixin.qq.com/",
}

# MP backend ret codes that benefit from retry/backoff
RETRYABLE_RET_CODES = {200013, 200018, -1}
SESSION_EXPIRED_RET = 200003

# ---------------------------------------------------------------------------
# Session manager – shared by all adapters
# ---------------------------------------------------------------------------


class SessionExpiredError(Exception):
    """Raised when the MP session cookie/token is no longer valid."""


class WeChatMPSessionManager:
    """Persist and refresh a WeChat MP backend login session.

    Stores cookies as a flat dict on disk.  On load, they are injected into a
    requests Session cookie jar so that Set-Cookie from API responses
    automatically refresh the session.
    """

    def __init__(self, session_path: str = "data/wechat_mp_session.json"):
        self._path = Path(session_path)
        self._cookies: dict[str, str] = {}
        self._http: Optional[requests.Session] = None

    # ---- public API --------------------------------------------------------

    @property
    def session_path(self) -> Path:
        return self._path

    def load(self) -> bool:
        """Load saved cookies from disk.  Returns True on success."""
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._cookies = data.get("cookies", {})
                logger.info("Loaded MP session from %s", self._path)
                return bool(self._cookies)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load MP session: %s", exc)
        return False

    def save(self, cookies: dict[str, str]) -> None:
        """Persist cookies to disk (called from login script)."""
        self._cookies = dict(cookies)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"cookies": self._cookies, "updated": datetime.now(UTC).isoformat()}
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os_mode = self._path.stat().st_mode
        self._path.chmod(os_mode & 0o777 | 0o600)
        logger.info("MP session saved to %s", self._path)
        # Reset HTTP session so next request picks up fresh cookies
        self._http = None

    def probe_status(self) -> tuple[str, str]:
        """Probe whether the saved MP session is usable.

        Returns ``(status, detail)`` where status is one of:
        ``valid``, ``missing_token``, ``expired``, ``invalid``, or ``unreachable``.
        """
        token = self.get_token()
        if not token:
            return "missing_token", "missing token"
        try:
            resp = self.get(
                f"{MP_BASE}/cgi-bin/searchbiz",
                params={
                    "action": "search_biz",
                    "query": "test",
                    "token": token,
                    "begin": "0",
                    "count": "1",
                    "lang": "zh_CN",
                    "f": "json",
                    "ajax": "1",
                },
            )
            data = resp.json()
        except Exception as exc:
            return "unreachable", str(exc)

        ret = (data.get("base_resp") or {}).get("ret", -1)
        if ret == 0:
            return "valid", ""
        if ret == SESSION_EXPIRED_RET:
            return "expired", f"ret={ret}"
        return "invalid", f"ret={ret}"

    def is_valid(self) -> bool:
        """Quick liveness probe via searchbiz."""
        status, _ = self.probe_status()
        return status == "valid"

    def get_token(self) -> Optional[str]:
        return self._cookies.get("token")

    def get(self, url: str, params: Optional[dict] = None, timeout: int = 30) -> requests.Response:
        """Issue a GET with session cookies via the internal HTTP session.

        Stores cookies in the requests.Session cookie jar so Set-Cookie
        responses are handled automatically.  The jar is refreshed from
        ``self._cookies`` on every call so external ``save()`` calls are
        picked up without re-creating the session.
        """
        http = self._http
        if http is None:
            http = requests.Session()
            http.headers.update(BROWSER_HEADERS)
            http.trust_env = False  # ignore system proxy (ClashX etc.)
            self._http = http

        # Sync flat dict into requests cookie jar
        http.cookies.clear()
        for k, v in self._cookies.items():
            if k == "token":
                continue  # token is a param, not a cookie
            http.cookies.set(k, v, domain=".weixin.qq.com")

        return http.get(url, params=params, timeout=timeout, allow_redirects=False)

    def refresh_from_response(self, resp: requests.Response) -> None:
        """Update the flat cookie dict from the response's cookie jar."""
        changed = False
        for cookie in resp.cookies:
            if cookie.value and cookie.value != self._cookies.get(cookie.name):
                self._cookies[cookie.name] = cookie.value
                changed = True
        # Also persist to disk so future sessions pick up refreshed cookies
        if changed:
            self.save(self._cookies)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class WeChatMPAdapter(NewsSourceAdapter):
    """Fetch recent articles from a WeChat public account via the MP backend API."""

    def __init__(
        self,
        account_name: str,
        session_manager: WeChatMPSessionManager,
        fakeid: Optional[str] = None,
        rate_limit: float = 5.0,
        timeout: int = 30,
        max_retries: int = 2,
        page_delay: float = 3.0,
    ):
        super().__init__(rate_limit=rate_limit)
        self.account_name = account_name
        self._session_mgr = session_manager
        self._fakeid = fakeid
        self.timeout = timeout
        self.max_retries = max_retries
        self.page_delay = page_delay
        self._resolved_fakeid: Optional[str] = None
        self._skip_stats: dict[str, int] = {
            "invalid_url": 0,
            "non_article_type": 0,
            "duplicate_url": 0,
        }

    # ---- NewsSourceAdapter interface ---------------------------------------

    def fetch(
        self,
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> list[RawNewsItem]:
        self._rate_limit()
        self._skip_stats = {k: 0 for k in self._skip_stats}

        fakeid = self._resolve_fakeid()
        if not fakeid:
            logger.warning("[%s] Cannot resolve fakeid – skipping", self._source_name)
            return []

        token = self._session_mgr.get_token()
        if not token:
            logger.warning("[%s] No MP session token – skipping", self._source_name)
            return []

        articles = self._paginate(fakeid, token, since, limit)
        items = self._build_items(articles, since, limit)

        skipped = {k: v for k, v in self._skip_stats.items() if v}
        if skipped:
            logger.debug("[%s] Skipped articles: %s", self._source_name, skipped)
        return items

    # ---- internals ---------------------------------------------------------

    def _request_json(
        self,
        endpoint: str,
        params: dict,
    ) -> Optional[dict]:
        """GET an MP JSON endpoint with retry and exponential backoff."""
        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._session_mgr.get(
                    f"{MP_BASE}{endpoint}",
                    params=params,
                    timeout=self.timeout,
                )
                data = resp.json()
            except Exception as exc:
                last_error = str(exc)
                if attempt < self.max_retries:
                    time.sleep(2.0 * (2 ** attempt))
                    continue
                logger.warning(
                    "[%s] %s request failed: %s",
                    self._source_name,
                    endpoint,
                    last_error,
                )
                return None

            self._session_mgr.refresh_from_response(resp)
            ret = (data.get("base_resp") or {}).get("ret", -1)
            if ret == SESSION_EXPIRED_RET:
                raise SessionExpiredError(
                    f"MP session expired for [{self._source_name}]"
                )
            if ret == 0:
                return data
            if ret in RETRYABLE_RET_CODES and attempt < self.max_retries:
                sleep_s = 2.0 * (2 ** attempt)
                logger.debug(
                    "[%s] MP ret=%s, retry in %.1fs",
                    self._source_name,
                    ret,
                    sleep_s,
                )
                time.sleep(sleep_s)
                continue

            logger.warning(
                "[%s] %s returned ret=%s: %s",
                self._source_name,
                endpoint,
                ret,
                json.dumps(data, ensure_ascii=False)[:300],
            )
            return None

        logger.warning("[%s] %s exhausted retries: %s", self._source_name, endpoint, last_error)
        return None

    @staticmethod
    def _pick_fakeid_from_search(account_name: str, biz_list: list[dict]) -> Optional[str]:
        """Match the configured account name against search results."""
        if not biz_list:
            return None

        target = account_name.strip().lower()
        for biz in biz_list:
            nickname = str(biz.get("nickname", "")).strip().lower()
            alias = str(biz.get("alias", "")).strip().lower()
            if nickname == target or alias == target:
                return biz.get("fakeid") or None

        # Fallback: first result with a warning in logs
        return biz_list[0].get("fakeid") or None

    def _resolve_fakeid(self) -> Optional[str]:
        """Return the fakeid for this account, resolving via search if needed."""
        if self._resolved_fakeid:
            return self._resolved_fakeid

        # Pre-configured fakeid takes priority
        if self._fakeid:
            self._resolved_fakeid = self._fakeid
            return self._resolved_fakeid

        # Dynamic resolution via searchbiz
        token = self._session_mgr.get_token()
        if not token:
            return None

        logger.info("[%s] Searching for fakeid via searchbiz …", self._source_name)
        data = self._request_json(
            "/cgi-bin/searchbiz",
            {
                "action": "search_biz",
                "query": self.account_name,
                "token": token,
                "begin": "0",
                "count": "5",
                "lang": "zh_CN",
                "f": "json",
                "ajax": "1",
            },
        )
        if not data:
            return None

        biz_list = data.get("list", [])
        fakeid = self._pick_fakeid_from_search(self.account_name, biz_list)
        if fakeid:
            if biz_list and str(biz_list[0].get("fakeid", "")) != fakeid:
                logger.info(
                    "[%s] Matched fakeid=%s by nickname (not first search hit)",
                    self._source_name,
                    fakeid,
                )
            else:
                logger.info("[%s] Resolved fakeid=%s", self._source_name, fakeid)
            self._resolved_fakeid = fakeid
            return self._resolved_fakeid

        return None

    def _paginate(
        self,
        fakeid: str,
        token: str,
        since: Optional[datetime],
        limit: Optional[int],
    ) -> list[dict]:
        """Fetch article pages until we run past *since* or exhaust API pages.

        When *limit* is set (e.g. in tests), stop after that many articles.
        Production collection relies on the *since* time window only.
        """
        collected: list[dict] = []
        begin = 0
        page_size = 20

        while limit is None or len(collected) < limit:
            data = self._request_json(
                "/cgi-bin/appmsgpublish",
                {
                    "fakeid": fakeid,
                    "token": token,
                    "begin": str(begin),
                    "count": str(page_size),
                    "lang": "zh_CN",
                    "f": "json",
                    "ajax": "1",
                },
            )
            if not data:
                break

            page_articles = self._parse_publish_page(data)
            if not page_articles:
                logger.debug(
                    "[%s] appmsgpublish: empty page at begin=%s",
                    self._source_name,
                    begin,
                )
                break

            for art in page_articles:
                if limit is not None and len(collected) >= limit:
                    return collected

                ts = art.get("create_time", 0)
                art_date = (
                    datetime.fromtimestamp(int(ts), tz=UTC) if ts else None
                )
                if since and art_date and art_date < since:
                    return collected
                collected.append(art)

            begin += page_size
            if self.page_delay > 0:
                time.sleep(self.page_delay)

        return collected

    @staticmethod
    def _parse_publish_page(data: dict) -> list[dict]:
        """Unpack the double-JSON-encoded publish_page response."""
        publish_page_raw = data.get("publish_page", "")
        if not publish_page_raw:
            return []

        try:
            publish_page = json.loads(publish_page_raw)
        except (json.JSONDecodeError, TypeError):
            logger.debug("Failed to parse publish_page JSON")
            return []

        articles: list[dict] = []
        for item in publish_page.get("publish_list", []):
            info_raw = item.get("publish_info", "")
            if not info_raw:
                continue
            try:
                info = json.loads(info_raw)
            except (json.JSONDecodeError, TypeError):
                continue

            # appmsg_info is the active field; appmsgex is deprecated/empty
            for art in info.get("appmsg_info", []):
                # Inject create_time from sent_info if not present on the article
                if "create_time" not in art:
                    art["create_time"] = (info.get("sent_info") or {}).get("time", 0)
                articles.append(art)

        return articles

    def _build_items(
        self,
        articles: list[dict],
        since: Optional[datetime],
        limit: Optional[int],
    ) -> list[RawNewsItem]:
        items: list[RawNewsItem] = []
        seen_urls: set[str] = set()

        for art in articles:
            if limit is not None and len(items) >= limit:
                break

            if not is_fetchable_article_type(art):
                self._skip_stats["non_article_type"] += 1
                continue

            title = (art.get("title") or "").strip()
            raw_link = ((art.get("link") or art.get("content_url") or "")).strip()
            link = normalize_wechat_article_url(raw_link)
            if not title or not link:
                continue
            if not is_valid_wechat_article_url(link):
                self._skip_stats["invalid_url"] += 1
                continue
            if link in seen_urls:
                self._skip_stats["duplicate_url"] += 1
                continue
            seen_urls.add(link)

            ts = art.get("create_time", 0)
            event_date = (
                datetime.fromtimestamp(int(ts), tz=UTC)
                if ts
                else datetime.now(UTC)
            )

            if since and event_date < since:
                continue

            digest = (art.get("digest") or art.get("cover") or "").strip()
            item_id = art.get("appmsgid", "")
            author = (art.get("author") or art.get("author_name") or "").strip()

            items.append(
                RawNewsItem(
                    source_name=self._source_name,
                    source_tier=self._source_tier,
                    title=title,
                    event_date=event_date,
                    source_url=link,
                    summary=digest,
                    source_item_id=str(item_id) if item_id else None,
                    raw_payload={
                        "article_json": art,
                        "fakeid": self._resolved_fakeid,
                        "author": author,
                        "item_show_type": art.get("item_show_type"),
                    },
                )
            )

        return items
