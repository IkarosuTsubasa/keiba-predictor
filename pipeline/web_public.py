import html
import json
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from public_share_copy import (
    PUBLIC_SHARE_DETAIL_LABEL as SHARE_COPY_DETAIL_LABEL,
    PUBLIC_SHARE_HASHTAG as SHARE_COPY_HASHTAG,
    PUBLIC_SHARE_MAX_CHARS as SHARE_COPY_MAX_CHARS,
    PUBLIC_SHARE_URL as SHARE_COPY_URL,
)


BASE_DIR = Path(__file__).resolve().parent
PUBLIC_FRONTEND_DIST_DIR = BASE_DIR / "public_frontend_dist"

PUBLIC_BASE_PATH = "/keiba"
CONSOLE_BASE_PATH = f"{PUBLIC_BASE_PATH}/console"
PUBLIC_API_BASE_PATH = f"{PUBLIC_BASE_PATH}/api/public"
PUBLIC_SITE_ICON_PATH = f"{PUBLIC_BASE_PATH}/site-icon.png"
PUBLIC_FAVICON_PATH = f"{PUBLIC_BASE_PATH}/favicon.ico"
PUBLIC_OG_IMAGE_PATH = f"{PUBLIC_BASE_PATH}/og.png"
PUBLIC_SITE_URL = "https://www.ikaimo-ai.com"
PUBLIC_CANONICAL_URL = f"{PUBLIC_SITE_URL}{PUBLIC_BASE_PATH}"
PUBLIC_OG_IMAGE_URL = f"{PUBLIC_SITE_URL}{PUBLIC_OG_IMAGE_PATH}"
PUBLIC_META_TITLE = "いかいもAI競馬"
PUBLIC_META_DESCRIPTION = "4つのAIが同時に競馬予想"
PUBLIC_SHARE_URL = "https://www.ikaimo-ai.com/keiba"
PUBLIC_SHARE_DETAIL_LABEL = "全モデル・全買い目はこちら（無料公開中）"
PUBLIC_SHARE_HASHTAG = "#いかいもAI競馬 #競馬"
PUBLIC_SHARE_MAX_CHARS = 130


PUBLIC_META_TITLE = "いかいもAI競馬"
PUBLIC_META_DESCRIPTION = "4つのAIが同時に競馬予想を公開する競馬分析サイト"
PUBLIC_SHARE_DETAIL_LABEL = "全モデル・全買い目はこちら（無料公開中）"
PUBLIC_SHARE_HASHTAG = "#いかいもAI競馬 #競馬"
# Public share copy must come from public_share_copy as the single source of truth.
# Keep literal share strings out of this file to avoid accidental edits while changing meta or public-page logic.
PUBLIC_SHARE_URL = SHARE_COPY_URL
PUBLIC_SHARE_DETAIL_LABEL = SHARE_COPY_DETAIL_LABEL
PUBLIC_SHARE_HASHTAG = SHARE_COPY_HASHTAG
PUBLIC_SHARE_MAX_CHARS = SHARE_COPY_MAX_CHARS

def mount_public_assets(app: FastAPI) -> None:
    app.mount(
        "/assets",
        StaticFiles(directory=PUBLIC_FRONTEND_DIST_DIR / "assets", check_dir=False),
        name="public-assets",
    )
    app.mount(
        f"{PUBLIC_BASE_PATH}/assets",
        StaticFiles(directory=PUBLIC_FRONTEND_DIST_DIR / "assets", check_dir=False),
        name="keiba-public-assets",
    )


def prefix_public_html_routes(content=""):
    html_text = str(content or "")
    replacements = (
        ('href="/console', f'href="{CONSOLE_BASE_PATH}'),
        ('action="/console', f'action="{CONSOLE_BASE_PATH}'),
        ('href="/llm_today"', f'href="{PUBLIC_BASE_PATH}"'),
        ('action="/llm_today"', f'action="{PUBLIC_BASE_PATH}"'),
        ('href="/site-icon.png"', f'href="{PUBLIC_SITE_ICON_PATH}"'),
        ('href="/favicon.ico"', f'href="{PUBLIC_FAVICON_PATH}"'),
    )
    for source, target in replacements:
        html_text = html_text.replace(source, target)
    return html_text


def load_public_index_html():
    index_path = PUBLIC_FRONTEND_DIST_DIR / "index.html"
    for enc in ("utf-8", "utf-8-sig"):
        try:
            with open(index_path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(index_path, "r", encoding="cp932") as f:
        return f.read()


PUBLIC_PAGE_META = {
    PUBLIC_BASE_PATH: {
        "title": "いかいもAI競馬",
        "description": "複数のAI視点を重ねて競馬分析を公開する競馬分析サイト",
    },
    f"{PUBLIC_BASE_PATH}/history": {
        "title": "履歴分析 | いかいもAI競馬",
        "description": "LLMと量化モデルの過去成績をまとめて振り返る公開ヒストリーページです。",
    },
    f"{PUBLIC_BASE_PATH}/about": {
        "title": "このサイトについて | いかいもAI競馬",
        "description": "いかいもAI競馬の考え方と、複数の視点を重ねる競馬分析の方針を紹介します。",
    },
    f"{PUBLIC_BASE_PATH}/guide": {
        "title": "ガイド | いかいもAI競馬",
        "description": "印、ROI、買い目、見送り判断など、サイト内で表示される情報の見方を案内します。",
    },
    f"{PUBLIC_BASE_PATH}/methodology": {
        "title": "分析方法 | いかいもAI競馬",
        "description": "多面的な評価、比較、オッズとの整合、見送り判断など、本サイトの分析フレームを紹介します。",
    },
    f"{PUBLIC_BASE_PATH}/privacy": {
        "title": "プライバシーポリシー | いかいもAI競馬",
        "description": "Cookie、アクセス情報、広告配信に関する当サイトのプライバシーポリシーです。",
    },
    f"{PUBLIC_BASE_PATH}/terms": {
        "title": "利用規約 | いかいもAI競馬",
        "description": "いかいもAI競馬の利用条件、禁止事項、免責範囲を記載した利用規約です。",
    },
    f"{PUBLIC_BASE_PATH}/disclaimer": {
        "title": "免責事項 | いかいもAI競馬",
        "description": "予想情報、投票判断、情報の正確性、責任範囲に関する免責事項を案内します。",
    },
    f"{PUBLIC_BASE_PATH}/contact": {
        "title": "お問い合わせ | いかいもAI競馬",
        "description": "掲載内容や運営に関する連絡先と、お問い合わせ時の案内を掲載しています。",
    },
    CONSOLE_BASE_PATH: {
        "title": "管理コンソール | いかいもAI競馬",
        "description": "管理者向けコンソールです。",
        "noindex": True,
    },
    f"{CONSOLE_BASE_PATH}/workspace": {
        "title": "Workspace | いかいもAI競馬",
        "description": "管理者向けワークスペースです。",
        "noindex": True,
    },
}


def _public_page_meta(path=""):
    normalized_path = str(path or "").rstrip("/") or PUBLIC_BASE_PATH
    if normalized_path.startswith(f"{PUBLIC_BASE_PATH}/race/"):
        meta = PUBLIC_PAGE_META[PUBLIC_BASE_PATH].copy()
        meta["title"] = "レース詳細 | いかいもAI競馬"
        meta["description"] = "各レースの買い目、印、モデル別の推奨馬を見やすく整理した詳細ページです。"
        meta["canonical_url"] = f"{PUBLIC_SITE_URL}{normalized_path}"
        return meta
    meta = PUBLIC_PAGE_META.get(normalized_path, PUBLIC_PAGE_META[PUBLIC_BASE_PATH]).copy()
    meta["canonical_url"] = f"{PUBLIC_SITE_URL}{normalized_path}"
    return meta


def inject_public_meta_tags(content="", path=""):
    html_text = str(content or "")
    if not html_text:
        return html_text

    page_meta = _public_page_meta(path)
    meta_title = page_meta["title"]
    meta_description = page_meta["description"]
    canonical_url = page_meta["canonical_url"]
    robots_tag = '<meta name="robots" content="noindex, nofollow" />' if page_meta.get("noindex") else ""

    title_tag = f"<title>{html.escape(meta_title)}</title>"
    description_tag = f'<meta name="description" content="{html.escape(meta_description)}" />'
    html_text = re.sub(r"<title>.*?</title>", title_tag, html_text, count=1, flags=re.IGNORECASE | re.DOTALL)
    html_text = re.sub(
        r'<meta\s+name=["\']description["\'][^>]*>',
        description_tag,
        html_text,
        count=1,
        flags=re.IGNORECASE,
    )
    html_text = re.sub(r'\s*<link\s+rel=["\']canonical["\'][^>]*>\s*', "\n", html_text, flags=re.IGNORECASE)
    html_text = re.sub(r'\s*<meta\s+property=["\']og:[^>]*>\s*', "\n", html_text, flags=re.IGNORECASE)
    html_text = re.sub(r'\s*<meta\s+name=["\']twitter:[^>]*>\s*', "\n", html_text, flags=re.IGNORECASE)

    meta_block = f"""
    <link rel="canonical" href="{html.escape(canonical_url)}" />
    <meta property="og:type" content="website" />
    <meta property="og:site_name" content="いかいもAI競馬" />
    <meta property="og:title" content="{html.escape(meta_title)}" />
    <meta property="og:description" content="{html.escape(meta_description)}" />
    <meta property="og:url" content="{html.escape(canonical_url)}" />
    <meta property="og:image" content="{html.escape(PUBLIC_OG_IMAGE_URL)}" />
    <meta property="og:image:alt" content="{html.escape(meta_title)}" />
    <meta name="twitter:card" content="summary_large_image" />
    <meta name="twitter:title" content="{html.escape(meta_title)}" />
    <meta name="twitter:description" content="{html.escape(meta_description)}" />
    <meta name="twitter:image" content="{html.escape(PUBLIC_OG_IMAGE_URL)}" />
    {robots_tag}
    """.strip()
    return re.sub(r"</head>", f"{meta_block}\n  </head>", html_text, count=1, flags=re.IGNORECASE)


def _public_share_runtime_html():
    runtime = """
<style>
.share-title-row,
.share-title-inline {
  display: flex;
  align-items: center;
  gap: 12px;
}
.share-title-row {
  justify-content: space-between;
}
.share-inline-button {
  width: 36px;
  height: 36px;
  flex: 0 0 36px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  appearance: none;
  border: 1px solid rgba(15, 18, 24, 0.18);
  border-radius: 999px;
  background: #111827;
  color: #ffffff;
  cursor: pointer;
  transition: transform 0.18s ease, background 0.18s ease, border-color 0.18s ease;
}
.share-inline-button:hover {
  transform: translateY(-1px);
  background: #0f172a;
  border-color: rgba(15, 18, 24, 0.28);
}
.share-inline-button img {
  width: 15px;
  height: 15px;
  display: block;
  object-fit: contain;
}
@media (max-width: 760px) {
  .share-title-row {
    align-items: flex-start;
  }
  .share-title-inline {
    align-items: flex-start;
  }
}
</style>
<script>
(() => {
  const SHARE_DETAIL_LABEL = "__SHARE_DETAIL_LABEL__";
  const SHARE_URL = "__SHARE_URL__";
  const SHARE_HASHTAG = "__SHARE_HASHTAG__";
  const SHARE_MAX_CHARS = 130;

  const parseRaceHeader = (title) => {
    const text = String(title || "").trim();
    const matched = text.match(/^(.*?)(\\d+R)$/i);
    if (!matched) {
      return text ? `#${text}` : "#\\u7af6\\u99acAI";
    }
    let venue = String(matched[1] || "").replace(/\\s+/g, "");
    const raceNo = String(matched[2] || "").trim();
    if (venue && !venue.endsWith("\\u7af6\\u99ac")) {
      venue += "\\u7af6\\u99ac";
    }
    if (venue) {
      return `#${venue} ${raceNo}`;
    }
    return raceNo || "#\\u7af6\\u99acAI";
  };

  const splitLines = (text) =>
    String(text || "")
      .split(/\\n+/)
      .map((item) => item.trim())
      .filter(Boolean);

  const buildShareText = (raceTitle, card) => {
    const presetText = String(card?.dataset?.shareText || "").trim();
    if (presetText) {
      return presetText;
    }
    let ticketText = "";
    let marksText = "\\u5370\\u306a\\u3057";
    if (card.matches(".model-card")) {
      const blocks = Array.from(card.querySelectorAll(".model-block"));
      ticketText = blocks[0]?.querySelector("p")?.innerText || "";
      marksText = blocks[2]?.querySelector("p")?.innerText || "\\u5370\\u306a\\u3057";
    } else {
      const mainHorse = card.querySelector(".ai-pick-summary__main strong")?.textContent?.trim() || "";
      const subMarks = Array.from(card.querySelectorAll(".ai-pick-summary__submark")).map((item) => {
        const symbol = item.querySelector("em")?.textContent?.trim() || "";
        const horseNo = item.querySelector("strong")?.textContent?.trim() || "";
        return symbol && horseNo ? `${symbol}${horseNo}` : "";
      }).filter(Boolean);
      const markParts = [];
      if (mainHorse) {
        markParts.push(`\\u25ce${mainHorse}`);
      }
      markParts.push(...subMarks);
      if (markParts.length) {
        marksText = markParts.join(" ");
      }
      ticketText = Array.from(card.querySelectorAll(".bet-preview-list li")).map((item) => item.textContent?.trim() || "").filter(Boolean).join("\\n");
    }
    const header = parseRaceHeader(raceTitle);
    const ticketLines = splitLines(ticketText);
    const tailLines = [SHARE_DETAIL_LABEL, SHARE_URL, SHARE_HASHTAG];
    const lines = [header, String(marksText || "\\u5370\\u306a\\u3057").trim() || "\\u5370\\u306a\\u3057", "", "\\u8cb7\\u3044\\u76ee\\uff08\\u4e00\\u90e8\\uff09"];
    for (const ticketLine of ticketLines) {
      const candidate = [...lines, ticketLine, "", ...tailLines].join("\\n");
      if (candidate.length > SHARE_MAX_CHARS) {
        break;
      }
      lines.push(ticketLine);
    }
    if (lines.length === 4) {
      const fallback = [...lines, "\\u8cb7\\u3044\\u76ee\\u306a\\u3057", "", ...tailLines].join("\\n");
      if (fallback.length <= SHARE_MAX_CHARS) {
        lines.push("\\u8cb7\\u3044\\u76ee\\u306a\\u3057");
      }
    }
    let text = [...lines, "", ...tailLines].join("\\n");
    if (text.length <= SHARE_MAX_CHARS) {
      return text;
    }
    text = [header, String(marksText || "\\u5370\\u306a\\u3057").trim() || "\\u5370\\u306a\\u3057", "", ...tailLines].join("\\n");
    if (text.length <= SHARE_MAX_CHARS) {
      return text;
    }
    const tailLength = ["", ...tailLines].join("\\n").length;
    const remain = Math.max(1, SHARE_MAX_CHARS - header.length - tailLength - 3);
    return [header, String(marksText || "\\u5370\\u306a\\u3057").trim().slice(0, remain), "", ...tailLines].join("\\n");
  };

  const openShare = async (text) => {
    const shareUrl = `https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}`;
    const isMobileShare =
      /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent || "") ||
      (window.matchMedia && window.matchMedia("(max-width: 760px)").matches) ||
      ("ontouchstart" in window);
    if (isMobileShare && navigator.share) {
      try {
        await navigator.share({ text });
        return;
      } catch (error) {
        if (error && error.name === "AbortError") {
          return;
        }
      }
    }
    if (isMobileShare) {
      window.location.href = shareUrl;
      return;
    }
    const width = 720;
    const height = 640;
    const left = Math.max(0, Math.round((window.screen.width - width) / 2));
    const top = Math.max(0, Math.round((window.screen.height - height) / 2));
    const popup = window.open(
      shareUrl,
      "ikaimo-share",
      `popup=yes,width=${width},height=${height},left=${left},top=${top},resizable=yes,scrollbars=yes`
    );
    if (popup && !popup.closed) {
      try {
        popup.focus();
      } catch (_error) {
      }
      return;
    }
    window.location.href = shareUrl;
  };

  const createShareButton = () => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "share-inline-button";
    button.setAttribute("aria-label", "\\u30b7\\u30a7\\u30a2");
    button.setAttribute("title", "\\u30b7\\u30a7\\u30a2");
    button.innerHTML =
      '<svg viewBox="0 0 24 24" width="15" height="15" fill="currentColor" aria-hidden="true" focusable="false">' +
      '<path d="M18.901 1.153h3.68l-8.04 9.19L24 22.847h-7.406l-5.8-7.584-6.636 7.584H.478l8.6-9.83L0 1.153h7.594l5.243 6.932 6.064-6.932Zm-1.29 19.494h2.04L6.486 3.24H4.298l13.313 17.407Z"/>' +
      "</svg>";
    return button;
  };

  const parseRaceCardMinutes = (raceCard) => {
    if (!raceCard) {
      return null;
    }
    const badges = Array.from(raceCard.querySelectorAll(".race-card-header__badges span"));
    for (const badge of badges) {
      const text = String(badge.textContent || "").trim();
      const matched = text.match(/^(\\d{1,2}):(\\d{2})$/);
      if (!matched) {
        continue;
      }
      const hour = Number(matched[1]);
      const minute = Number(matched[2]);
      if (!Number.isFinite(hour) || !Number.isFinite(minute)) {
        continue;
      }
      return hour * 60 + minute;
    }
    return null;
  };

  const compareRaceCardsByTime = (left, right) => {
    const leftMinutes = parseRaceCardMinutes(left);
    const rightMinutes = parseRaceCardMinutes(right);
    if (leftMinutes !== null && rightMinutes !== null) {
      if (leftMinutes !== rightMinutes) {
        return rightMinutes - leftMinutes;
      }
    } else if (leftMinutes !== null || rightMinutes !== null) {
      return leftMinutes !== null ? -1 : 1;
    }
    const leftTitle = String(left?.querySelector(".race-card-header h3")?.textContent || "").trim();
    const rightTitle = String(right?.querySelector(".race-card-header h3")?.textContent || "").trim();
    return leftTitle.localeCompare(rightTitle, "ja");
  };

  const sortRaceGrids = () => {
    document.querySelectorAll(".race-grid").forEach((grid) => {
      const cards = Array.from(grid.querySelectorAll(":scope > .race-card"));
      if (cards.length < 2) {
        return;
      }
      const sorted = [...cards].sort(compareRaceCardsByTime);
      const changed = sorted.some((card, index) => card !== cards[index]);
      if (!changed) {
        return;
      }
      sorted.forEach((card) => grid.appendChild(card));
    });
  };

  const findCardsForShare = (root) => {
    const modernCards = Array.from(root.querySelectorAll(".ai-pick-summary"));
    if (modernCards.length) {
      return modernCards;
    }
    return Array.from(root.querySelectorAll(".model-card"));
  };

  const mountLegacyShareButton = (summary) => {
    if (!summary || summary.dataset.shareMounted === "1") {
      return;
    }
    const raceCopy = summary.querySelector(".race-copy");
    const title = raceCopy?.querySelector("h2");
    if (!raceCopy || !title) {
      return;
    }
    const row = document.createElement("div");
    row.className = "share-title-row";
    title.parentNode.insertBefore(row, title);
    row.appendChild(title);
    const button = createShareButton();
    row.appendChild(button);
    const handleShare = async (event) => {
      event.preventDefault();
      event.stopPropagation();
      const raceBoard = summary.closest(".race-board");
      const cards = findCardsForShare(raceBoard || document);
      if (!cards.length) {
        return;
      }
      const selected = cards[Math.floor(Math.random() * cards.length)];
      const text = buildShareText(title.textContent || "", selected);
      if (!text) {
        return;
      }
      await openShare(text);
    };
    button.addEventListener("click", handleShare);
    button.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      event.stopPropagation();
    });
    summary.dataset.shareMounted = "1";
  };

  const mountModernShareButton = (header) => {
    if (!header || header.dataset.shareMounted === "1") {
      return;
    }
    const main = header.querySelector(".race-card-header__main");
    const titleHost = main?.querySelector("div");
    const title = titleHost?.querySelector("h3");
    if (!main || !titleHost || !title) {
      return;
    }
    const row = document.createElement("div");
    row.className = "share-title-inline";
    titleHost.insertBefore(row, title);
    row.appendChild(title);
    const button = createShareButton();
    row.appendChild(button);
    const handleShare = async (event) => {
      event.preventDefault();
      event.stopPropagation();
      const raceCard = header.closest(".race-card");
      const cards = findCardsForShare(raceCard || document);
      if (!cards.length) {
        return;
      }
      const selected = cards[Math.floor(Math.random() * cards.length)];
      const text = buildShareText(title.textContent || "", selected);
      if (!text) {
        return;
      }
      await openShare(text);
    };
    button.addEventListener("click", handleShare);
    button.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      event.stopPropagation();
    });
    header.dataset.shareMounted = "1";
  };

  const refreshShareButtons = () => {
    document.querySelectorAll(".race-summary").forEach(mountLegacyShareButton);
    document.querySelectorAll(".race-card-header").forEach(mountModernShareButton);
    sortRaceGrids();
  };

  const observer = new MutationObserver(() => {
    refreshShareButtons();
  });

  const start = () => {
    refreshShareButtons();
    observer.observe(document.body, { childList: true, subtree: true });
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
})();
</script>
"""
    return (
        runtime
        .replace('"__SHARE_DETAIL_LABEL__"', json.dumps(PUBLIC_SHARE_DETAIL_LABEL, ensure_ascii=True))
        .replace('"__SHARE_URL__"', json.dumps(PUBLIC_SHARE_URL, ensure_ascii=True))
        .replace('"__SHARE_HASHTAG__"', json.dumps(PUBLIC_SHARE_HASHTAG, ensure_ascii=True))
    )


def inject_public_share_runtime(html_text):
    content = str(html_text or "")
    runtime = _public_share_runtime_html()
    if runtime in content:
        return content
    if "</body>" in content:
        return content.replace("</body>", runtime + "\n</body>", 1)
    return content + runtime


def build_public_index_response(path=""):
    html_text = load_public_index_html()
    html_text = prefix_public_html_routes(html_text)
    html_text = inject_public_meta_tags(html_text, path=path)
    html_text = inject_public_share_runtime(html_text)
    return HTMLResponse(html_text)


def register_public_static_routes(app: FastAPI) -> None:
    @app.get(PUBLIC_SITE_ICON_PATH)
    @app.get("/site-icon.png")
    def public_site_icon():
        icon_path = PUBLIC_FRONTEND_DIST_DIR / "site-icon.png"
        if icon_path.exists():
            return FileResponse(icon_path)
        raise HTTPException(status_code=404, detail="site icon not found")

    @app.get(PUBLIC_FAVICON_PATH)
    @app.get("/favicon.ico")
    def public_favicon():
        icon_path = PUBLIC_FRONTEND_DIST_DIR / "site-icon.png"
        if icon_path.exists():
            return FileResponse(icon_path, media_type="image/png")
        raise HTTPException(status_code=404, detail="favicon not found")

    @app.get(PUBLIC_OG_IMAGE_PATH)
    @app.get("/og.png")
    def public_og_image():
        og_path = PUBLIC_FRONTEND_DIST_DIR / "og.png"
        if og_path.exists():
            return FileResponse(og_path, media_type="image/png")
        fallback_path = PUBLIC_FRONTEND_DIST_DIR / "site-icon.png"
        if fallback_path.exists():
            return FileResponse(fallback_path, media_type="image/png")
        raise HTTPException(status_code=404, detail="og image not found")
