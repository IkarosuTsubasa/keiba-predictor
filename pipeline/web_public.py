import html
import json
import re
from datetime import datetime, timedelta
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
from site_copy import load_site_copy


BASE_DIR = Path(__file__).resolve().parent
PUBLIC_FRONTEND_DIST_DIR = BASE_DIR / "public_frontend_dist"
PUBLIC_APP_ADS_TXT_FILE = BASE_DIR / "app-ads.txt"

PUBLIC_BASE_PATH = "/keiba"
CONSOLE_BASE_PATH = f"{PUBLIC_BASE_PATH}/console"
PUBLIC_API_BASE_PATH = f"{PUBLIC_BASE_PATH}/api/public"
PUBLIC_SITE_ICON_PATH = f"{PUBLIC_BASE_PATH}/site-icon.png"
PUBLIC_FAVICON_PATH = f"{PUBLIC_BASE_PATH}/favicon.ico"
PUBLIC_APPLE_TOUCH_ICON_PATH = f"{PUBLIC_BASE_PATH}/apple-touch-icon.png"
PUBLIC_OG_IMAGE_PATH = f"{PUBLIC_BASE_PATH}/og.png"
PUBLIC_GOOGLE_PLAY_BADGE_PATH = f"{PUBLIC_BASE_PATH}/GetItOnGooglePlay_Badge_Web_color_Japanese.png"
PUBLIC_APP_ADS_TXT_PATH = "/app-ads.txt"
PUBLIC_ADS_TXT_PATH = "/ads.txt"
PUBLIC_SITE_URL = "https://www.ikaimo-ai.com"
PUBLIC_CANONICAL_URL = f"{PUBLIC_SITE_URL}{PUBLIC_BASE_PATH}"
PUBLIC_OG_IMAGE_URL = f"{PUBLIC_SITE_URL}{PUBLIC_OG_IMAGE_PATH}"
SITE_COPY = load_site_copy()
SITE_NAME = str(SITE_COPY.get("site", {}).get("name") or "いかいもAI競馬")
PUBLIC_META_TITLE = str(SITE_COPY.get("site", {}).get("home_page_title") or SITE_NAME)
PUBLIC_META_DESCRIPTION = str(
    SITE_COPY.get("site", {}).get("home_page_description") or "競馬分析サイト"
)
PUBLIC_PAGES_COPY = dict(SITE_COPY.get("public_pages", {}) or {})
ROUTE_META_COPY = dict(SITE_COPY.get("route_meta", {}) or {})
HOME_COPY = dict(SITE_COPY.get("home", {}) or {})
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
        ('href="/apple-touch-icon.png"', f'href="{PUBLIC_APPLE_TOUCH_ICON_PATH}"'),
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
        "title": PUBLIC_META_TITLE,
        "description": PUBLIC_META_DESCRIPTION,
    }
}

for page in PUBLIC_PAGES_COPY.values():
    page_path = str(page.get("path") or "").strip()
    page_title = str(page.get("title") or "").strip()
    page_description = str(page.get("meta_description") or "").strip()
    if not page_path or not page_title or not page_description:
        continue
    PUBLIC_PAGE_META[page_path] = {
        "title": f"{page_title} | {SITE_NAME}",
        "description": page_description,
    }

for route_meta in ROUTE_META_COPY.values():
    route_path = str(route_meta.get("path") or "").strip()
    route_title = str(route_meta.get("title") or "").strip()
    route_description = str(route_meta.get("description") or "").strip()
    if not route_path or not route_title or not route_description:
        continue
    PUBLIC_PAGE_META[route_path] = {
        "title": route_title,
        "description": route_description,
        "noindex": bool(route_meta.get("noindex")),
    }


def _public_page_meta(path=""):
    normalized_path = str(path or "").rstrip("/") or PUBLIC_BASE_PATH
    if normalized_path.startswith(f"{PUBLIC_BASE_PATH}/race/"):
        meta = PUBLIC_PAGE_META[PUBLIC_BASE_PATH].copy()
        meta["title"] = str(ROUTE_META_COPY.get("race_detail", {}).get("title") or f"レース詳細 | {SITE_NAME}")
        meta["description"] = str(
            ROUTE_META_COPY.get("race_detail", {}).get("description")
            or "各レースの買い目、印、モデル別の推奨馬を見やすく整理した詳細ページです。"
        )
        meta["canonical_url"] = f"{PUBLIC_SITE_URL}{normalized_path}"
        return meta
    if normalized_path.startswith(f"{PUBLIC_BASE_PATH}/reports/"):
        meta = PUBLIC_PAGE_META.get(f"{PUBLIC_BASE_PATH}/reports", PUBLIC_PAGE_META[PUBLIC_BASE_PATH]).copy()
        meta["title"] = str(ROUTE_META_COPY.get("report_detail", {}).get("title") or f"私の日報 | {SITE_NAME}")
        meta["description"] = str(
            ROUTE_META_COPY.get("report_detail", {}).get("description")
            or "保存された日報記事の詳細ページです。"
        )
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
    <meta property="og:site_name" content="{html.escape(SITE_NAME)}" />
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


def _safe_text(value=""):
    return str(value or "").strip()


def _public_home_intro_html(payload=None):
    payload = dict(payload or {})
    totals = dict(payload.get("totals", {}) or {})
    hero = dict(payload.get("hero", {}) or {})
    lead_race = dict(hero.get("lead_race", {}) or {})
    leader = dict(hero.get("leader", {}) or {})
    home_hero = dict(HOME_COPY.get("hero", {}) or {})
    featured_section = dict(HOME_COPY.get("featured_section", {}) or {})
    featured_cards = list(HOME_COPY.get("featured_items", []) or [])
    method_section = dict(HOME_COPY.get("method_section", {}) or {})
    method_cards = list(HOME_COPY.get("method_steps", []) or [])

    def _parse_main_horse(mark_text=""):
        matched = re.search(r"◎\s*([0-9]+)", _safe_text(mark_text))
        return matched.group(1) if matched else ""

    def _agreement_stats(race_row):
        cards = list((race_row or {}).get("cards", []) or [])
        main_horses = [_parse_main_horse(card.get("marks_text")) for card in cards]
        main_horses = [item for item in main_horses if item]
        counts = {}
        for horse_no in main_horses:
            counts[horse_no] = counts.get(horse_no, 0) + 1
        top_horse = ""
        top_count = 0
        for horse_no, count in counts.items():
            if count > top_count:
                top_horse = horse_no
                top_count = count
        no_bet_count = 0
        for card in cards:
            if _safe_text(card.get("decision_text")).lower() == "no_bet":
                no_bet_count += 1
        agreement_ratio = float(top_count) / float(len(main_horses)) if main_horses else 0.0
        return {
            "card_count": len(cards),
            "valid_main_count": len(main_horses),
            "unique_main_count": len(counts),
            "top_horse": top_horse,
            "top_count": top_count,
            "no_bet_count": no_bet_count,
            "agreement_ratio": agreement_ratio,
        }

    def _analysis_tags(stats):
        tags = []
        if float(stats.get("agreement_ratio", 0.0) or 0.0) >= 0.75:
            tags.append("高一致")
        elif int(stats.get("unique_main_count", 0) or 0) >= 3 or int(stats.get("no_bet_count", 0) or 0) >= 2:
            tags.append("見解差あり")
        if int(stats.get("top_count", 0) or 0) >= 2 and float(stats.get("agreement_ratio", 0.0) or 0.0) >= 0.5:
            tags.append("軸向き")
        if int(stats.get("unique_main_count", 0) or 0) >= 3 or int(stats.get("no_bet_count", 0) or 0) >= 1:
            tags.append("波乱注意")
        if not tags:
            tags.append("軸向き")
        return tags[:3]

    def _tokyo_today_key():
        return (datetime.utcnow() + timedelta(hours=9)).strftime("%Y-%m-%d")

    def _target_date_heading(date_text="", date_label=""):
        matched = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", _safe_text(date_text))
        if matched:
            return f"{int(matched.group(1))}年{int(matched.group(2))}月{int(matched.group(3))}日"
        return _safe_text(date_label) or "対象日"

    def _recommendation_reason(stats, lead_meta_text, leader_label_text):
        if float(stats.get("agreement_ratio", 0.0) or 0.0) >= 0.75:
            return f"{leader_label_text}を含む複数モデルの視点が近く、読み筋の起点として確認しやすいレースです。"
        if int(stats.get("unique_main_count", 0) or 0) >= 3 or int(stats.get("no_bet_count", 0) or 0) >= 2:
            return "本命候補や見送り判断が割れやすく、複数LLMの差を比較する価値が高いレースです。"
        if int(stats.get("top_count", 0) or 0) >= 2:
            return "軸候補の重なりが見えやすく、買い目構成の差だけを落ち着いて比較しやすいレースです。"
        return f"{lead_meta_text}の条件で判断材料が揃っており、定量モデルとLLMの差を読み始める基点に向いています。"

    def _cards_html(items):
        out = []
        for item in items:
            tags_html = "".join(
                f"<span>{html.escape(tag)}</span>" for tag in list(item.get("tags", []) or []) if _safe_text(tag)
            )
            out.append(
                f"""
                <article class="public-home-static-intro__content-card">
                  <span>{html.escape(_safe_text(item.get('category')))}</span>
                  <strong>{html.escape(_safe_text(item.get('title')))}</strong>
                  <p>{html.escape(_safe_text(item.get('excerpt') or item.get('description')))}</p>
                  <div class="public-home-static-intro__content-tags">{tags_html}</div>
                  <a href="{html.escape(_safe_text(item.get('href')))}">{html.escape(_safe_text(item.get('cta')) or '続きを読む')}</a>
                </article>
                """
            )
        return "".join(out)

    race_count = int(totals.get("race_count", 0) or 0)
    settled_count = int(totals.get("settled_count", 0) or 0)
    updated_at = _safe_text(payload.get("generated_at_label")) or "公開中"
    target_date = _safe_text(payload.get("target_date"))
    target_date_label = _target_date_heading(target_date, payload.get("target_date_label"))
    is_today = bool(target_date) and target_date == _tokyo_today_key()
    summary_heading = "本日のサマリー" if is_today else f"{target_date_label}のサマリー"
    guide_cta_label = _safe_text(HOME_COPY.get("list_cta_label")) or "レース一覧を見る"
    race_list_href = f"{PUBLIC_BASE_PATH}?date={html.escape(target_date)}" if target_date else PUBLIC_BASE_PATH
    lead_race_title = _safe_text(lead_race.get("race_title")) or f"{target_date_label}の注目レース"
    leader_label = _safe_text(leader.get("label")) or "各モデル"
    lead_meta_items = [
        _safe_text(lead_race.get("location")),
        _safe_text(lead_race.get("distance_label")),
        _safe_text(lead_race.get("track_condition")),
    ]
    lead_meta_items = [item for item in lead_meta_items if item]
    lead_meta = " / ".join(lead_meta_items) if lead_meta_items else target_date_label
    lead_stats = _agreement_stats(lead_race)
    analysis_tags_html = "".join(f"<span>{html.escape(tag)}</span>" for tag in _analysis_tags(lead_stats))
    lead_text = (
        f"{lead_race_title}を起点に、対象日の{race_count}レースを比較できます。"
        f"{leader_label}を含む各モデルの判断差と買い目構成の違いを、同じ導線で確認できます。"
    )
    reason_text = _recommendation_reason(lead_stats, lead_meta, leader_label)

    metrics_html = "".join(
        [
            f"""
            <article class="public-home-static-intro__metric">
              <span>対象日</span>
              <strong>{html.escape(target_date_label)}</strong>
            </article>
            """,
            f"""
            <article class="public-home-static-intro__metric">
              <span>データ更新</span>
              <strong>{html.escape(updated_at)}</strong>
            </article>
            """,
            f"""
            <article class="public-home-static-intro__metric">
              <span>対象レース数</span>
              <strong>{race_count}レース</strong>
            </article>
            """,
            f"""
            <article class="public-home-static-intro__metric">
              <span>結果確定</span>
              <strong>{settled_count}レース</strong>
            </article>
            """,
        ]
    )

    return f"""
<style>
.public-home-static-intro {{
  width: min(1440px, calc(100vw - 24px));
  margin: 14px auto 0;
  color: #1a1c1d;
}}
.home-app-ready .public-home-static-intro {{
  display: none;
}}
.public-home-static-intro__inner,
.public-home-static-intro__summary,
.public-home-static-intro__content-card {{
  border-radius: 28px;
  background: #ffffff;
  box-shadow: 0 18px 48px rgba(26, 28, 29, 0.08);
}}
.public-home-static-intro__inner {{
  padding: 24px;
  display: grid;
  gap: 20px;
  background:
    radial-gradient(circle at 0% 0%, rgba(203, 167, 47, 0.14), transparent 28%),
    radial-gradient(circle at 100% 0%, rgba(26, 35, 126, 0.12), transparent 24%),
    linear-gradient(180deg, #ffffff, #f7f7fa);
}}
.public-home-static-intro__hero {{
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 18px;
}}
.public-home-static-intro__copy,
.public-home-static-intro__summary,
.public-home-static-intro__section,
.public-home-static-intro__content-card {{
  display: grid;
  gap: 12px;
}}
.public-home-static-intro__eyebrow,
.public-home-static-intro__section-eyebrow,
.public-home-static-intro__metric span,
.public-home-static-intro__content-card span {{
  color: #735c00;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}}
.public-home-static-intro__copy h1,
.public-home-static-intro__summary strong,
.public-home-static-intro__section h2,
.public-home-static-intro__content-card strong {{
  margin: 0;
  color: #000666;
  font-family: "Manrope", "Noto Sans JP", sans-serif;
  letter-spacing: -0.05em;
}}
.public-home-static-intro__copy h1 {{
  max-width: 620px;
  font-size: clamp(28px, 4vw, 44px);
  line-height: 1.02;
}}
.public-home-static-intro__summary strong {{
  font-size: clamp(24px, 3vw, 34px);
  line-height: 0.98;
}}
.public-home-static-intro__copy p,
.public-home-static-intro__summary p,
.public-home-static-intro__section p,
.public-home-static-intro__content-card p {{
  margin: 0;
  color: #5f6270;
  line-height: 1.72;
}}
.public-home-static-intro__actions,
.public-home-static-intro__summary-badges,
.public-home-static-intro__content-tags {{
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}}
.public-home-static-intro__primary,
.public-home-static-intro__secondary,
.public-home-static-intro__summary a,
.public-home-static-intro__content-card a {{
  min-height: 44px;
  padding: 0 16px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  text-decoration: none;
  font-weight: 700;
}}
.public-home-static-intro__primary {{
  background: linear-gradient(135deg, #000666, #1a237e);
  color: #ffffff;
}}
.public-home-static-intro__secondary,
.public-home-static-intro__summary a,
.public-home-static-intro__content-card a {{
  background: rgba(255, 255, 255, 0.86);
  color: #000666;
  box-shadow: inset 0 0 0 1px rgba(198, 197, 212, 0.18);
}}
.public-home-static-intro__summary {{
  padding: 22px;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.92), rgba(245, 245, 248, 0.98));
}}
.public-home-static-intro__summary-reason {{
  display: grid;
  gap: 6px;
  padding: 12px 14px;
  border-radius: 18px;
  background: rgba(224, 224, 255, 0.42);
}}
.public-home-static-intro__summary-reason span {{
  color: #735c00;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}}
.public-home-static-intro__summary-badges span,
.public-home-static-intro__content-tags span {{
  min-height: 28px;
  padding: 0 10px;
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  background: rgba(224, 224, 255, 0.62);
  color: #000666;
  font-size: 11px;
  font-weight: 700;
}}
.public-home-static-intro__summary-meta {{
  margin: 0;
  color: #5f6270;
  font-size: 13px;
}}
.public-home-static-intro__metrics,
.public-home-static-intro__content-grid {{
  display: grid;
  gap: 10px;
}}
.public-home-static-intro__metrics {{
  grid-template-columns: repeat(2, minmax(0, 1fr));
}}
.public-home-static-intro__content-grid {{
  grid-template-columns: repeat(3, minmax(0, 1fr));
}}
.public-home-static-intro__metric,
.public-home-static-intro__content-card {{
  min-height: 100%;
  padding: 16px;
  background: linear-gradient(180deg, #ffffff, #f5f5f8);
}}
.public-home-static-intro__metric strong {{
  font-size: clamp(18px, 2vw, 28px);
  line-height: 1;
}}
@media (max-width: 1180px) {{
  .public-home-static-intro__hero,
  .public-home-static-intro__content-grid {{
    grid-template-columns: 1fr;
  }}
}}
@media (max-width: 680px) {{
  .public-home-static-intro__inner {{
    padding: 18px;
    border-radius: 24px;
  }}
  .public-home-static-intro__copy h1 {{
    font-size: 29px;
  }}
  .public-home-static-intro__actions {{
    display: grid;
  }}
}}
</style>
<section class="public-home-static-intro" aria-label="ホームイントロ">
  <div class="public-home-static-intro__inner">
    <div class="public-home-static-intro__hero">
      <div class="public-home-static-intro__copy">
        <span class="public-home-static-intro__eyebrow">{html.escape(_safe_text(home_hero.get("eyebrow")))}</span>
        <h1>{html.escape(_safe_text(home_hero.get("title")))}</h1>
        <p>{html.escape(_safe_text(home_hero.get("description")))}</p>
        <div class="public-home-static-intro__actions">
          <a class="public-home-static-intro__primary" href="{race_list_href}">{html.escape(guide_cta_label)}</a>
        </div>
      </div>
    </div>
    <section class="public-home-static-intro__section">
      <span class="public-home-static-intro__section-eyebrow">{html.escape(_safe_text(featured_section.get("eyebrow")))}</span>
      <h2>{html.escape(_safe_text(featured_section.get("title")))}</h2>
      <p>{html.escape(_safe_text(featured_section.get("description")))}</p>
      <div class="public-home-static-intro__content-grid">{_cards_html(featured_cards)}</div>
    </section>
    <section class="public-home-static-intro__section">
      <span class="public-home-static-intro__section-eyebrow">{html.escape(_safe_text(method_section.get("eyebrow")))}</span>
      <h2>{html.escape(_safe_text(method_section.get("title")))}</h2>
      <p>{html.escape(_safe_text(method_section.get("description")))}</p>
      <div class="public-home-static-intro__content-grid">{_cards_html(method_cards)}</div>
    </section>
  </div>
</section>
""".strip()

def inject_public_home_intro(content="", path="", payload=None):
    html_text = str(content or "")
    normalized_path = str(path or "").rstrip("/") or PUBLIC_BASE_PATH
    if normalized_path != PUBLIC_BASE_PATH or not html_text or payload is None:
        return html_text

    intro_html = _public_home_intro_html(payload=payload)
    if intro_html in html_text:
        return html_text
    slot_marker = "<!-- PUBLIC_HOME_INTRO_SLOT -->"
    if slot_marker in html_text:
        return html_text.replace(slot_marker, intro_html, 1)
    if '<div id="root"></div>' in html_text:
        return html_text.replace('<div id="root"></div>', f"{intro_html}\n    <div id=\"root\"></div>", 1)
    return intro_html + html_text


def inject_public_initial_board_data(content="", path="", payload=None):
    html_text = str(content or "")
    normalized_path = str(path or "").rstrip("/") or PUBLIC_BASE_PATH
    if normalized_path != PUBLIC_BASE_PATH or not html_text or payload is None:
        return html_text
    if 'id="keiba-public-board-data"' in html_text:
        return html_text
    try:
        json_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    except (TypeError, ValueError):
        return html_text
    script = (
        '<script id="keiba-public-board-data" type="application/json">'
        f"{html.escape(json_text)}"
        "</script>"
    )
    if '<div id="root"></div>' in html_text:
        return html_text.replace('<div id="root"></div>', f"{script}\n    <div id=\"root\"></div>", 1)
    if "</body>" in html_text:
        return html_text.replace("</body>", f"{script}\n</body>", 1)
    return html_text + script


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
.share-inline-button:disabled {
  cursor: wait;
  opacity: 0.72;
  transform: none;
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
  const SHARE_HASHTAG = "__SHARE_HASHTAG__";
  const IMAGE_WIDTH = 1200;
  const IMAGE_HEIGHT = 675;
  const FONT_FAMILY = '"Hiragino Sans", "Yu Gothic", "Yu Gothic UI", "Meiryo", "Noto Sans JP", sans-serif';
  const MARK_SYMBOLS = ["◎", "○", "▲", "△", "☆"];

  const cleanText = (value) => String(value || "").replace(/\\s+/g, " ").trim();

  const formatRaceTitle = (title) => {
    const text = String(title || "").trim();
    const matched = text.match(/^(.*?)(\\d+R)$/i);
    if (!matched) {
      return text || "\\u7af6\\u99acAI";
    }
    let venue = String(matched[1] || "").replace(/\\s+/g, "");
    const raceNo = String(matched[2] || "").trim();
    if (venue && !venue.endsWith("\\u7af6\\u99ac")) {
      venue += "\\u7af6\\u99ac";
    }
    if (venue) {
      return `${venue} ${raceNo}`;
    }
    return raceNo || "\\u7af6\\u99acAI";
  };

  const drawRoundRect = (ctx, x, y, width, height, radius) => {
    const r = Math.min(radius, width / 2, height / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + width - r, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + r);
    ctx.lineTo(x + width, y + height - r);
    ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
    ctx.lineTo(x + r, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  };

  const fillRoundRect = (ctx, x, y, width, height, radius, fillStyle) => {
    drawRoundRect(ctx, x, y, width, height, radius);
    ctx.fillStyle = fillStyle;
    ctx.fill();
  };

  const strokeRoundRect = (ctx, x, y, width, height, radius, strokeStyle, lineWidth = 1) => {
    drawRoundRect(ctx, x, y, width, height, radius);
    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = lineWidth;
    ctx.stroke();
  };

  const setFont = (ctx, weight, size) => {
    ctx.font = `${weight} ${size}px ${FONT_FAMILY}`;
  };

  const ellipsize = (ctx, text, maxWidth) => {
    const source = cleanText(text);
    if (!source || ctx.measureText(source).width <= maxWidth) {
      return source;
    }
    let out = source;
    while (out.length > 1 && ctx.measureText(`${out}...`).width > maxWidth) {
      out = out.slice(0, -1);
    }
    return `${out}...`;
  };

  const drawText = (ctx, text, x, y, maxWidth) => {
    ctx.fillText(maxWidth ? ellipsize(ctx, text, maxWidth) : cleanText(text), x, y);
  };

  const drawPill = (ctx, text, x, y, options = {}) => {
    const label = cleanText(text);
    if (!label) {
      return 0;
    }
    const size = options.size || 22;
    const weight = options.weight || 700;
    const paddingX = options.paddingX || 18;
    const height = options.height || 44;
    setFont(ctx, weight, size);
    const width = Math.ceil(ctx.measureText(label).width + paddingX * 2);
    fillRoundRect(ctx, x, y, width, height, Math.min(18, height / 2), options.fill || "rgba(199, 166, 109, 0.14)");
    strokeRoundRect(ctx, x, y, width, height, Math.min(18, height / 2), options.stroke || "rgba(199, 166, 109, 0.34)", 1);
    ctx.fillStyle = options.color || "#f5f1ea";
    ctx.textBaseline = "middle";
    drawText(ctx, label, x + paddingX, y + height / 2 + 1, width - paddingX * 2);
    ctx.textBaseline = "alphabetic";
    return width;
  };

  const parseMarks = (text) => [...String(text || "").matchAll(/([◎○▲△☆])\\s*([0-9]+)/g)]
    .map((item) => ({
      symbol: item[1],
      horseNo: item[2],
    }))
    .filter((item) => item.symbol && item.horseNo);

  const uniqueMarks = (marks) => {
    const seen = new Set();
    return Array.from(marks || []).filter((item) => {
      const key = `${item.symbol}-${item.horseNo}`;
      if (!item.symbol || !item.horseNo || seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
  };

  const collectMarks = (card) => {
    if (!card) {
      return [];
    }
    if (card.matches(".model-card")) {
      const blocks = Array.from(card.querySelectorAll(".model-block"));
      return uniqueMarks(parseMarks(blocks[2]?.querySelector("p")?.innerText || ""));
    }
    if (card.matches(".model-race-summary")) {
      return uniqueMarks(Array.from(card.querySelectorAll(".model-race-summary__mark")).map((item) => ({
        symbol: item.querySelector("em")?.textContent?.trim() || "",
        horseNo: item.querySelector("strong")?.textContent?.trim() || "",
      })));
    }
    const presetMarks = parseMarks(card.dataset?.shareText || "");
    const mainHorse = card.querySelector(".ai-pick-summary__main strong")?.textContent?.trim() || "";
    const mainSymbol = card.querySelector(".ai-pick-summary__main span")?.textContent?.trim() || "◎";
    const marks = [];
    if (mainHorse && mainHorse !== "-") {
      marks.push({ symbol: mainSymbol || "◎", horseNo: mainHorse });
    }
    marks.push(
      ...Array.from(card.querySelectorAll(".ai-pick-summary__submark")).map((item) => ({
        symbol: item.querySelector("em")?.textContent?.trim() || "",
        horseNo: item.querySelector("strong")?.textContent?.trim() || "",
      })),
    );
    return uniqueMarks(marks.length ? marks : presetMarks);
  };

  const formatMark = (mark) => {
    if (!mark || !mark.symbol || !mark.horseNo) {
      return "";
    }
    return `${mark.symbol} ${mark.horseNo}`;
  };

  const resolvePayload = (raceTitle, card) => {
    const raceCard = card?.closest(".race-card") || card?.closest(".race-board") || null;
    const header = raceCard?.querySelector(".race-card-header") || document;
    const marks = collectMarks(card);
    const mainMark =
      marks.find((item) => item.symbol === "◎") ||
      marks[0] ||
      { symbol: "◎", horseNo: "-" };
    const supportMarks = MARK_SYMBOLS
      .filter((symbol) => symbol !== mainMark.symbol)
      .map((symbol) => marks.find((item) => item.symbol === symbol))
      .filter(Boolean)
      .slice(0, 4);
    const badges = Array.from(header.querySelectorAll(".race-card-header__badges span"))
      .map((item) => cleanText(item.textContent))
      .filter(Boolean)
      .slice(0, 3);
    const modelName =
      cleanText(card?.querySelector(".ai-pick-summary__model")?.textContent) ||
      cleanText(card?.closest(".model-top-five-board")?.querySelector(".model-top-five-board__tabs .is-active")?.textContent) ||
      "\\u7dcf\\u5408\\u4e88\\u6e2c";
    const metaBadge = card?.querySelector(".model-meta-badge");
    const metaLabel = cleanText(metaBadge?.querySelector("span")?.textContent);
    const metaValue = cleanText(metaBadge?.querySelector("strong")?.textContent);
    const confidence = cleanText(raceCard?.querySelector(".race-card__metric-cell strong")?.textContent);
    const decision = cleanText(raceCard?.querySelector(".race-card__decision-cell strong")?.textContent);
    const status = cleanText(header.querySelector(".race-card-header__status")?.textContent);
    return {
      raceTitle: formatRaceTitle(raceTitle),
      subtitle: cleanText(header.querySelector(".race-card-header__subtitle")?.textContent),
      badges,
      modelName,
      metaLabel,
      metaValue,
      confidence,
      decision,
      status,
      mainMark,
      supportMarks,
    };
  };

  const buildShareText = (raceTitle) => {
    const header = formatRaceTitle(raceTitle);
    return [header, "AI\\u6700\\u7d42\\u8a55\\u4fa1\\u3092\\u753b\\u50cf\\u3067\\u5171\\u6709", "", SHARE_HASHTAG]
      .filter((line) => String(line || "").trim())
      .join("\\n");
  };

  const drawShareImage = (payload) => new Promise((resolve, reject) => {
    const canvas = document.createElement("canvas");
    canvas.width = IMAGE_WIDTH;
    canvas.height = IMAGE_HEIGHT;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      reject(new Error("canvas context unavailable"));
      return;
    }

    const bg = ctx.createLinearGradient(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT);
    bg.addColorStop(0, "#05080d");
    bg.addColorStop(0.52, "#091018");
    bg.addColorStop(1, "#101923");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT);

    ctx.save();
    ctx.globalAlpha = 0.18;
    ctx.strokeStyle = "#c7a66d";
    ctx.lineWidth = 1;
    for (let x = 72; x < IMAGE_WIDTH; x += 96) {
      ctx.beginPath();
      ctx.moveTo(x, 36);
      ctx.lineTo(x - 220, IMAGE_HEIGHT - 36);
      ctx.stroke();
    }
    ctx.restore();

    fillRoundRect(ctx, 46, 36, IMAGE_WIDTH - 92, IMAGE_HEIGHT - 72, 18, "rgba(16, 25, 37, 0.86)");
    strokeRoundRect(ctx, 46, 36, IMAGE_WIDTH - 92, IMAGE_HEIGHT - 72, 18, "rgba(202, 184, 145, 0.28)", 2);

    ctx.fillStyle = "#c7a66d";
    setFont(ctx, 800, 30);
    drawText(ctx, "\\u3044\\u304b\\u3044\\u3082AI\\u7af6\\u99ac", 74, 80, 320);
    ctx.fillStyle = "#9ba7b8";
    setFont(ctx, 700, 19);
    drawText(ctx, "AI\\u6700\\u7d42\\u8a55\\u4fa1", 74, 112, 260);

    let pillX = 820;
    const topPills = [payload.status, payload.decision].filter(Boolean).slice(0, 2);
    topPills.forEach((pill) => {
      const width = drawPill(ctx, pill, pillX, 58, {
        size: 22,
        height: 44,
        fill: "rgba(107, 211, 155, 0.12)",
        stroke: "rgba(107, 211, 155, 0.34)",
        color: "#d7ffe7",
      });
      pillX += width + 12;
    });

    ctx.fillStyle = "#f5f1ea";
    setFont(ctx, 900, 58);
    drawText(ctx, payload.raceTitle || "\\u7af6\\u99acAI", 74, 174, 790);

    ctx.fillStyle = "#9ba7b8";
    setFont(ctx, 700, 24);
    drawText(ctx, payload.subtitle || payload.badges.join(" / "), 76, 214, 700);

    fillRoundRect(ctx, 74, 248, 438, 236, 18, "rgba(5, 8, 13, 0.58)");
    strokeRoundRect(ctx, 74, 248, 438, 236, 18, "rgba(199, 166, 109, 0.36)", 2);
    ctx.fillStyle = "#c7a66d";
    setFont(ctx, 900, 118);
    drawText(ctx, payload.mainMark.symbol || "◎", 112, 392, 140);
    ctx.fillStyle = "#f5f1ea";
    setFont(ctx, 900, 158);
    drawText(ctx, payload.mainMark.horseNo || "-", 270, 405, 180);
    ctx.fillStyle = "#9ba7b8";
    setFont(ctx, 700, 24);
    drawText(ctx, "\\u672c\\u547d\\u30b7\\u30b0\\u30ca\\u30eb", 116, 448, 250);

    fillRoundRect(ctx, 544, 248, 582, 236, 18, "rgba(21, 32, 46, 0.78)");
    strokeRoundRect(ctx, 544, 248, 582, 236, 18, "rgba(202, 184, 145, 0.18)", 1);
    ctx.fillStyle = "#c7a66d";
    setFont(ctx, 800, 22);
    drawText(ctx, "\\u63a1\\u7528\\u30e2\\u30c7\\u30eb", 580, 300, 210);
    ctx.fillStyle = "#f5f1ea";
    setFont(ctx, 900, 44);
    drawText(ctx, payload.modelName || "\\u7dcf\\u5408\\u4e88\\u6e2c", 580, 354, 500);

    const metricText =
      payload.metaLabel && payload.metaValue && payload.metaLabel !== "\\u6307\\u6a19"
        ? `${payload.metaLabel} ${payload.metaValue}`
        : (payload.metaValue || payload.metaLabel);
    const infoItems = [
      ["\\u6307\\u6a19", metricText],
      ["\\u4fe1\\u983c\\u5ea6", payload.confidence],
      ["\\u72b6\\u614b", payload.status],
    ].filter((item) => cleanText(item[1]));
    let infoX = 580;
    let infoY = 392;
    infoItems.slice(0, 3).forEach((item) => {
      fillRoundRect(ctx, infoX, infoY, 164, 58, 14, "rgba(255, 255, 255, 0.045)");
      ctx.fillStyle = "#9ba7b8";
      setFont(ctx, 700, 17);
      drawText(ctx, item[0], infoX + 18, infoY + 23, 126);
      ctx.fillStyle = "#f5f1ea";
      setFont(ctx, 800, 21);
      drawText(ctx, item[1], infoX + 18, infoY + 48, 126);
      infoX += 178;
    });

    fillRoundRect(ctx, 74, 520, 1052, 78, 18, "rgba(199, 166, 109, 0.10)");
    strokeRoundRect(ctx, 74, 520, 1052, 78, 18, "rgba(199, 166, 109, 0.24)", 1);
    ctx.fillStyle = "#c7a66d";
    setFont(ctx, 800, 22);
    drawText(ctx, "\\u4e0a\\u4f4d\\u5370", 106, 568, 120);
    let markX = 226;
    const support = payload.supportMarks.length ? payload.supportMarks : [];
    support.forEach((mark) => {
      const width = drawPill(ctx, formatMark(mark), markX, 538, {
        size: 32,
        height: 46,
        paddingX: 20,
        fill: "rgba(245, 241, 234, 0.055)",
        stroke: "rgba(245, 241, 234, 0.12)",
        color: "#f5f1ea",
      });
      markX += width + 14;
    });
    if (!support.length) {
      ctx.fillStyle = "#f5f1ea";
      setFont(ctx, 800, 32);
      drawText(ctx, "\\u5370\\u306a\\u3057", markX, 570, 240);
    }

    ctx.fillStyle = "#9ba7b8";
    setFont(ctx, 700, 22);
    drawText(ctx, "\\u4e88\\u6e2c\\u5370\\u30fb\\u8cb7\\u3044\\u76ee\\u3092\\u6bce\\u65e5\\u66f4\\u65b0", 74, 626, 460);
    ctx.fillStyle = "#c7a66d";
    setFont(ctx, 800, 22);
    drawText(ctx, SHARE_HASHTAG, 760, 626, 360);

    canvas.toBlob((blob) => {
      if (blob) {
        resolve(blob);
      } else {
        reject(new Error("image blob unavailable"));
      }
    }, "image/png", 0.94);
  });

  const openTextShare = (text) => {
    const shareUrl = `https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}`;
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

  const downloadBlob = (blob) => {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "ikaimo-ai-keiba-share.png";
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  const openShare = async (text, payload) => {
    const blob = await drawShareImage(payload);
    const file = new File([blob], "ikaimo-ai-keiba-share.png", { type: "image/png" });
    if (navigator.canShare && navigator.canShare({ files: [file], text }) && navigator.share) {
      try {
        await navigator.share({ files: [file], text });
        return;
      } catch (error) {
        if (error && error.name === "AbortError") {
          return;
        }
      }
    }
    downloadBlob(blob);
    openTextShare(text);
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
    const modernCards = Array.from(root.querySelectorAll(".ai-pick-summary, .model-race-summary"));
    if (modernCards.length) {
      return modernCards;
    }
    return Array.from(root.querySelectorAll(".model-card"));
  };

  const cardHasMarksForShare = (card) => {
    const presetText = String(card?.dataset?.shareText || "").trim();
    if (presetText) {
      return !/[\\r\\n]印なし(?:[\\r\\n]|$)/.test(`\\n${presetText}\\n`);
    }
    if (card.matches(".model-card")) {
      const marksText = card.querySelector(".model-block:nth-child(3) p")?.innerText || "";
      return String(marksText).trim() && !String(marksText).includes("印なし");
    }
    if (card.matches(".model-race-summary")) {
      return Array.from(card.querySelectorAll(".model-race-summary__mark strong")).some((item) => {
        const text = item.textContent?.trim() || "";
        return text && text !== "-";
      });
    }
    const mainHorse = card.querySelector(".ai-pick-summary__main strong")?.textContent?.trim() || "";
    const subMarks = Array.from(card.querySelectorAll(".ai-pick-summary__submark")).length;
    return (mainHorse && mainHorse !== "-") || subMarks > 0;
  };

  const cardHasTicketsForShare = (card) => {
    const presetText = String(card?.dataset?.shareText || "").trim();
    if (presetText) {
      return /(単勝|複勝|ワイド|馬連|馬単|3連複)/.test(presetText) && !/買い目なし/.test(presetText);
    }
    if (card.matches(".model-card")) {
      const ticketText = card.querySelector(".model-block p")?.innerText || "";
      return String(ticketText).trim() && !String(ticketText).includes("買い目なし");
    }
    return Array.from(card.querySelectorAll(".bet-preview-list li")).some((item) => {
      const text = item.textContent?.trim() || "";
      return text && !text.includes("買い目なし");
    });
  };

  const pickCardForShare = (cards) => {
    const scored = Array.from(cards || []).map((card) => ({
      card,
      score: (cardHasMarksForShare(card) ? 2 : 0) + (cardHasTicketsForShare(card) ? 1 : 0),
    }));
    if (!scored.length) {
      return null;
    }
    const bestScore = Math.max(...scored.map((item) => item.score));
    const candidates = scored.filter((item) => item.score === bestScore).map((item) => item.card);
    if (!candidates.length) {
      return scored[0].card;
    }
    return candidates[Math.floor(Math.random() * candidates.length)];
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
      const selected = pickCardForShare(cards);
      const text = buildShareText(title.textContent || "");
      if (!text) {
        return;
      }
      const payload = resolvePayload(title.textContent || "", selected);
      button.disabled = true;
      try {
        await openShare(text, payload);
      } finally {
        button.disabled = false;
      }
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
      const selected = pickCardForShare(cards);
      const text = buildShareText(title.textContent || "");
      if (!text) {
        return;
      }
      const payload = resolvePayload(title.textContent || "", selected);
      button.disabled = true;
      try {
        await openShare(text, payload);
      } finally {
        button.disabled = false;
      }
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
    return runtime.replace('"__SHARE_HASHTAG__"', json.dumps(PUBLIC_SHARE_HASHTAG, ensure_ascii=True))


def inject_public_share_runtime(html_text):
    content = str(html_text or "")
    runtime = _public_share_runtime_html()
    if runtime in content:
        return content
    if "</body>" in content:
        return content.replace("</body>", runtime + "\n</body>", 1)
    return content + runtime


def build_public_index_response(path="", home_intro_payload=None, initial_board_payload=None):
    html_text = load_public_index_html()
    html_text = prefix_public_html_routes(html_text)
    html_text = inject_public_meta_tags(html_text, path=path)
    html_text = inject_public_home_intro(html_text, path=path, payload=home_intro_payload)
    html_text = inject_public_initial_board_data(html_text, path=path, payload=initial_board_payload)
    html_text = inject_public_share_runtime(html_text)
    return HTMLResponse(html_text)


def register_public_static_routes(app: FastAPI) -> None:
    @app.get(PUBLIC_APP_ADS_TXT_PATH)
    @app.get(f"{PUBLIC_BASE_PATH}/app-ads.txt")
    def public_app_ads_txt():
        if not PUBLIC_APP_ADS_TXT_FILE.exists():
            raise HTTPException(status_code=404, detail="app-ads.txt not found")
        return FileResponse(PUBLIC_APP_ADS_TXT_FILE, media_type="text/plain; charset=utf-8")

    @app.get(PUBLIC_ADS_TXT_PATH)
    @app.get(f"{PUBLIC_BASE_PATH}/ads.txt")
    def public_ads_txt():
        ads_path = PUBLIC_FRONTEND_DIST_DIR / "ads.txt"
        if ads_path.exists():
            return FileResponse(ads_path, media_type="text/plain; charset=utf-8")
        raise HTTPException(status_code=404, detail="ads.txt not found")

    @app.get(f"{PUBLIC_BASE_PATH}/affiliate/{{asset_path:path}}")
    @app.get("/affiliate/{asset_path:path}")
    def public_affiliate_asset(asset_path: str):
        relative_path = Path(asset_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise HTTPException(status_code=404, detail="affiliate asset not found")
        asset_file = PUBLIC_FRONTEND_DIST_DIR / "affiliate" / relative_path
        if asset_file.exists() and asset_file.is_file():
            return FileResponse(asset_file)
        raise HTTPException(status_code=404, detail="affiliate asset not found")

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

    @app.get(PUBLIC_APPLE_TOUCH_ICON_PATH)
    @app.get("/apple-touch-icon.png")
    def public_apple_touch_icon():
        icon_path = PUBLIC_FRONTEND_DIST_DIR / "site-icon.png"
        if icon_path.exists():
            return FileResponse(icon_path, media_type="image/png")
        raise HTTPException(status_code=404, detail="apple touch icon not found")

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

    @app.get(PUBLIC_GOOGLE_PLAY_BADGE_PATH)
    @app.get("/GetItOnGooglePlay_Badge_Web_color_Japanese.png")
    def public_google_play_badge():
        badge_path = PUBLIC_FRONTEND_DIST_DIR / "GetItOnGooglePlay_Badge_Web_color_Japanese.png"
        if badge_path.exists():
            return FileResponse(badge_path, media_type="image/png")
        raise HTTPException(status_code=404, detail="google play badge not found")




