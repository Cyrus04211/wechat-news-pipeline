"""WeChat public-account article URL helpers and content extraction.

Inspired by practices in WeSpy, WeChat-Article-Crawler, and wechat-reader:
- reject expired / preview links (tempkey=)
- unwrap captcha wrapper URLs
- fetch article HTML with browser-like headers and retry/backoff
- extract title, author, and body from standard MP article DOM
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, unquote, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

WECHAT_ARTICLE_HOSTS = {"mp.weixin.qq.com", "mp.weixin.qq.com.cn"}
WECHAT_ARTICLE_PATH_PREFIX = "/s/"
WECHAT_CDN_IMAGE_HOSTS = ("mmbiz.qpic.cn", "wx.qlogo.cn", "mmecoa.qpic.cn")
_MMBIZ_URL_RE = re.compile(
    r"https?://[^\s\"'<>]*mmbiz\.qpic\.cn[^\s\"'<>]*",
    re.IGNORECASE,
)

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://mp.weixin.qq.com/",
}

# Non-article publish types from MP backend (video/image share cards, etc.)
NON_ARTICLE_ITEM_SHOW_TYPES = {5, 7, 8, 10, 11, 16}


@dataclass
class WeChatArticleContent:
    url: str
    title: str = ""
    author: str = ""
    publish_time: str = ""
    content_text: str = ""
    content_html: str = ""
    image_urls: list[str] = None  # type: ignore[assignment]
    status: str = "ok"  # ok / invalid_url / fetch_failed / parse_failed / captcha

    def __post_init__(self) -> None:
        if self.image_urls is None:
            self.image_urls = []


def normalize_wechat_article_url(url: str) -> str:
    """Return the canonical article URL, unwrapping captcha redirect links."""
    url = (url or "").strip()
    if not url:
        return ""

    parsed = urlparse(url)
    if "wappoc_appmsgcaptcha" in parsed.path or "appmsgcaptcha" in parsed.path:
        qs = parse_qs(parsed.query)
        target = (qs.get("target_url") or qs.get("url") or [""])[0]
        if target:
            return unquote(target)

    if parsed.scheme in {"", "http"}:
        url = "https://" + url.lstrip("/")
        parsed = urlparse(url)

    if parsed.netloc in WECHAT_ARTICLE_HOSTS and parsed.path.startswith(WECHAT_ARTICLE_PATH_PREFIX):
        return f"https://{parsed.netloc}{parsed.path.split('?')[0]}"
    return url


def is_valid_wechat_article_url(url: str) -> bool:
    """Return False for blank, non-article, or expired preview links."""
    url = normalize_wechat_article_url(url)
    if not url:
        return False

    lower = url.lower()
    if "tempkey=" in lower:
        return False
    if "javascript:" in lower:
        return False

    parsed = urlparse(url)
    if parsed.netloc not in WECHAT_ARTICLE_HOSTS:
        return False
    if not parsed.path.startswith(WECHAT_ARTICLE_PATH_PREFIX):
        return False
    return True


def is_fetchable_article_type(article: dict) -> bool:
    """Skip deleted entries and non-text publish cards when metadata is available."""
    if not article:
        return False
    if article.get("is_deleted"):
        return False

    show_type = article.get("item_show_type")
    if show_type is not None:
        try:
            if int(show_type) in NON_ARTICLE_ITEM_SHOW_TYPES:
                return False
        except (TypeError, ValueError):
            pass
    return True


def _clean_text(text: str) -> str:
    text = unescape(text or "")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _extract_element_text(soup: BeautifulSoup, selectors: tuple[str, ...]) -> str:
    for selector in selectors:
        node = soup.select_one(selector)
        if node:
            return _clean_text(node.get_text("\n", strip=True))
    return ""


def is_wechat_cdn_image_url(url: str) -> bool:
    url = (url or "").strip()
    if not url.startswith("http"):
        return False
    host = urlparse(url).netloc.lower()
    return any(cdn in host for cdn in WECHAT_CDN_IMAGE_HOSTS)


def extract_wechat_image_urls_from_text(text: str) -> list[str]:
    """Extract WeChat CDN image URLs embedded in plain text (e.g. MP digest)."""
    if not text:
        return []
    urls: list[str] = []
    seen: set[str] = set()
    for match in _MMBIZ_URL_RE.findall(text):
        url = match.strip()
        if url and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def split_digest_and_cover(text: str) -> tuple[str, str]:
    """Split MP digest into human text vs cover image URL."""
    text = (text or "").strip()
    if not text:
        return "", ""
    if is_wechat_cdn_image_url(text):
        return "", text
    urls = extract_wechat_image_urls_from_text(text)
    cover = urls[0] if urls else ""
    return text, cover


def extract_image_urls_from_content_html(content_html: str) -> list[str]:
    """Extract unique image URLs from WeChat article body HTML."""
    if not content_html.strip():
        return []
    soup = BeautifulSoup(content_html, "html.parser")
    urls: list[str] = []
    seen: set[str] = set()
    for img in soup.find_all("img"):
        for attr in ("data-src", "src", "data-original"):
            url = (img.get(attr) or "").strip()
            if not url or url.startswith("data:"):
                continue
            if url not in seen:
                seen.add(url)
                urls.append(url)
    return urls


def _extract_content_html(content_el) -> tuple[str, str, list[str]]:
    for tag in content_el.find_all(["script", "style", "noscript"]):
        tag.decompose()
    for sel in (".qr_code_pc", ".reward_area", ".rich_media_tool"):
        for tag in content_el.select(sel):
            tag.decompose()
    for img in content_el.find_all("img"):
        data_src = img.get("data-src")
        if data_src and not img.get("src"):
            img["src"] = data_src
    html = str(content_el)
    text = _clean_text(content_el.get_text("\n", strip=True))
    image_urls = extract_image_urls_from_content_html(html)
    return html, text, image_urls


def parse_wechat_article_html(html: str, url: str = "") -> WeChatArticleContent:
    """Parse a rendered WeChat article page into structured content."""
    if not html.strip():
        return WeChatArticleContent(url=url, status="parse_failed")

    lower = html.lower()
    if "sec captcha" in lower or "环境异常" in html or "verify" in lower and "captcha" in lower:
        return WeChatArticleContent(url=url, status="captcha")

    soup = BeautifulSoup(html, "html.parser")
    title = _extract_element_text(
        soup,
        ("#activity-name", "h1.rich_media_title", 'meta[property="og:title"]'),
    )
    if not title:
        og = soup.select_one('meta[property="og:title"]')
        if og and og.get("content"):
            title = _clean_text(og["content"])

    author = _extract_element_text(
        soup,
        ("#js_name", "#js_author_name", ".rich_media_meta_text"),
    )
    publish_time = _extract_element_text(soup, ("#publish_time",))

    content_el = soup.select_one("#js_content")
    if content_el is None:
        content_el = soup.select_one("#page-content")
    if content_el is None:
        return WeChatArticleContent(
            url=url,
            title=title,
            author=author,
            publish_time=publish_time,
            status="parse_failed",
        )

    content_html, content_text, image_urls = _extract_content_html(content_el)
    if len(content_text) < 20:
        return WeChatArticleContent(
            url=url,
            title=title,
            author=author,
            publish_time=publish_time,
            content_html=content_html,
            content_text=content_text,
            image_urls=image_urls,
            status="parse_failed",
        )

    return WeChatArticleContent(
        url=url,
        title=title or "",
        author=author,
        publish_time=publish_time,
        content_text=content_text,
        content_html=content_html,
        image_urls=image_urls,
        status="ok",
    )


DEFAULT_MP_SESSION_PATH = "data/wechat_mp_session.json"


def load_mp_session_cookies(session_path: str = DEFAULT_MP_SESSION_PATH) -> dict[str, str]:
    """Load MP backend cookies saved by ``scripts/wechat_mp_login.py``."""
    path = Path(session_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        cookies = data.get("cookies", {})
        return dict(cookies) if isinstance(cookies, dict) else {}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load MP session cookies: %s", exc)
        return {}


def build_wechat_http_session(
    *,
    use_mp_cookies: bool = True,
    session_path: str = DEFAULT_MP_SESSION_PATH,
) -> requests.Session:
    """Build a requests session with browser headers and optional MP cookies."""
    http = requests.Session()
    http.headers.update(BROWSER_HEADERS)
    http.trust_env = False
    if use_mp_cookies:
        for key, value in load_mp_session_cookies(session_path).items():
            if key and value is not None:
                http.cookies.set(key, str(value))
    return http


def _article_cache_path(cache_dir: str, url: str) -> Path:
    digest = hashlib.sha256(normalize_wechat_article_url(url).encode("utf-8")).hexdigest()[:16]
    return Path(cache_dir) / f"{digest}.html"


def _read_article_cache(cache_dir: str, url: str) -> str:
    if not cache_dir:
        return ""
    path = _article_cache_path(cache_dir, url)
    if path.exists():
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
    return ""


def _write_article_cache(cache_dir: str, url: str, html: str) -> None:
    if not cache_dir or not html.strip():
        return
    path = _article_cache_path(cache_dir, url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def fetch_wechat_article_html_playwright(url: str, *, timeout: int = 30) -> str:
    """Fetch article HTML via headless Chromium when plain HTTP is blocked."""
    from playwright.sync_api import sync_playwright

    normalized = normalize_wechat_article_url(url)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page(user_agent=BROWSER_HEADERS["User-Agent"])
            page.goto(normalized, wait_until="domcontentloaded", timeout=timeout * 1000)
            try:
                page.wait_for_selector("#js_content", timeout=min(timeout, 15) * 1000)
            except Exception:
                pass
            page.wait_for_timeout(1500)
            return page.content()
        finally:
            browser.close()


def download_wechat_image(
    url: str,
    dest_path: str,
    *,
    session: Optional[requests.Session] = None,
    timeout: int = 30,
) -> bool:
    """Download a WeChat CDN image to a local path."""
    url = (url or "").strip()
    if not url or url.startswith("data:"):
        return False
    if url.startswith("//"):
        url = "https:" + url

    http = session or build_wechat_http_session(use_mp_cookies=False)
    try:
        resp = http.get(
            url,
            timeout=timeout,
            headers={"Referer": "https://mp.weixin.qq.com/"},
        )
        resp.raise_for_status()
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        Path(dest_path).write_bytes(resp.content)
        return True
    except Exception as exc:
        logger.debug("Failed to download WeChat image %s: %s", url[:80], exc)
        return False


def extract_payload_metadata(payload: dict | None) -> dict[str, str]:
    """Extract digest, author, title, and link from adapter ``raw_payload``."""
    payload = payload or {}
    article_json = payload.get("article_json") if isinstance(payload.get("article_json"), dict) else {}

    author = str(payload.get("author") or article_json.get("author") or article_json.get("author_name") or "").strip()
    digest = str(article_json.get("digest") or payload.get("digest") or "").strip()
    title = str(article_json.get("title") or payload.get("title") or "").strip()
    link = normalize_wechat_article_url(
        str(article_json.get("link") or article_json.get("content_url") or payload.get("source_url") or "").strip()
    )
    cover = str(article_json.get("cover") or payload.get("cover") or "").strip()
    return {
        "author": author,
        "digest": digest,
        "cover": cover,
        "title": title,
        "link": link,
    }


def fetch_wechat_article_content(
    url: str,
    *,
    timeout: int = 30,
    max_retries: int = 2,
    retry_backoff: float = 2.0,
    session: Optional[requests.Session] = None,
    use_mp_cookies: bool = True,
    playwright_fallback: bool = False,
    cache_dir: Optional[str] = None,
) -> WeChatArticleContent:
    """Fetch and parse a single WeChat article page."""
    normalized = normalize_wechat_article_url(url)
    if not is_valid_wechat_article_url(normalized):
        return WeChatArticleContent(url=url, status="invalid_url")

    cached_html = _read_article_cache(cache_dir or "", normalized)
    if cached_html:
        parsed = parse_wechat_article_html(cached_html, normalized)
        if parsed.status == "ok":
            return parsed

    http = session or build_wechat_http_session(use_mp_cookies=use_mp_cookies)

    last_error = ""
    last_status = ""
    for attempt in range(max_retries + 1):
        try:
            resp = http.get(normalized, timeout=timeout, allow_redirects=True)
            resp.raise_for_status()
            if "text/html" not in (resp.headers.get("Content-Type") or ""):
                last_error = f"unexpected content-type: {resp.headers.get('Content-Type')}"
            else:
                _write_article_cache(cache_dir or "", normalized, resp.text)
                parsed = parse_wechat_article_html(resp.text, normalized)
                if parsed.status == "ok":
                    return parsed
                if parsed.status == "captcha":
                    last_status = parsed.status
                    break
                last_error = parsed.status
                last_status = parsed.status
        except Exception as exc:
            last_error = str(exc)

        if attempt < max_retries:
            sleep_s = retry_backoff * (2 ** attempt)
            logger.debug("Retry WeChat article fetch in %.1fs: %s", sleep_s, normalized)
            time.sleep(sleep_s)

    if playwright_fallback and last_status in {"captcha", "parse_failed", "fetch_failed"}:
        try:
            html = fetch_wechat_article_html_playwright(normalized, timeout=timeout)
            _write_article_cache(cache_dir or "", normalized, html)
            parsed = parse_wechat_article_html(html, normalized)
            if parsed.status == "ok":
                logger.info("Playwright fallback succeeded for %s", normalized)
                return parsed
            last_error = parsed.status
        except Exception as exc:
            last_error = f"playwright:{exc}"

    logger.warning("Failed to fetch WeChat article %s: %s", normalized, last_error)
    return WeChatArticleContent(url=normalized, status=last_status or "fetch_failed")
