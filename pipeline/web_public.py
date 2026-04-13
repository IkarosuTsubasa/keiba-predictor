import html
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
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

PUBLIC_BASE_PATH = "/keiba"
CONSOLE_BASE_PATH = f"{PUBLIC_BASE_PATH}/console"
PUBLIC_API_BASE_PATH = f"{PUBLIC_BASE_PATH}/api/public"
PUBLIC_SITE_ICON_PATH = f"{PUBLIC_BASE_PATH}/site-icon.png"
PUBLIC_FAVICON_PATH = f"{PUBLIC_BASE_PATH}/favicon.ico"
PUBLIC_APPLE_TOUCH_ICON_PATH = f"{PUBLIC_BASE_PATH}/apple-touch-icon.png"
PUBLIC_OG_IMAGE_PATH = f"{PUBLIC_BASE_PATH}/og.png"
PUBLIC_ADS_TXT_PATH = "/ads.txt"
PUBLIC_SITE_URL = "https://www.ikaimo-ai.com"
PUBLIC_ADS_TXT_REDIRECT_URL = "https://srv.adstxtmanager.com/19390/www.ikaimo-ai.com"
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

  const toAbsoluteUrl = (href) => {
    const text = String(href || "").trim();
    if (!text) {
      return SHARE_URL;
    }
    try {
      return new URL(text, window.location.origin).toString();
    } catch (_error) {
      return SHARE_URL;
    }
  };

  const resolveDetailUrl = (card) => {
    if (window.location.pathname.includes("/race/")) {
      return window.location.href;
    }
    const raceCard = card?.closest(".race-card");
    const detailHref = raceCard?.querySelector(".race-card__toggle")?.getAttribute("href") || "";
    return toAbsoluteUrl(detailHref);
  };

  const replaceShareUrl = (text, detailUrl) => {
    const source = String(text || "").trim();
    if (!source) {
      return "";
    }
    const escapedShareUrl = SHARE_URL.replace(/[.*+?^${}()|[\\]\\\\]/g, "\\$&");
    const placeholderLinePattern = new RegExp(`(^|\\n)${escapedShareUrl}(?=\\n|$)`, "g");
    return placeholderLinePattern.test(source)
      ? source.replace(placeholderLinePattern, (matched, prefix) => `${prefix}${detailUrl}`)
      : source;
  };

  const buildShareText = (raceTitle, card) => {
    const detailUrl = resolveDetailUrl(card);
    const presetText = String(card?.dataset?.shareText || "").trim();
    if (presetText) {
      return replaceShareUrl(presetText, detailUrl);
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
    const tailLines = [SHARE_DETAIL_LABEL, detailUrl, SHARE_HASHTAG];
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

  const cardHasMarksForShare = (card) => {
    const presetText = String(card?.dataset?.shareText || "").trim();
    if (presetText) {
      return !/[\\r\\n]印なし(?:[\\r\\n]|$)/.test(`\\n${presetText}\\n`);
    }
    if (card.matches(".model-card")) {
      const marksText = card.querySelector(".model-block:nth-child(3) p")?.innerText || "";
      return String(marksText).trim() && !String(marksText).includes("印なし");
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
      const selected = pickCardForShare(cards);
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


def build_public_index_response(path="", home_intro_payload=None):
    html_text = load_public_index_html()
    html_text = prefix_public_html_routes(html_text)
    html_text = inject_public_meta_tags(html_text, path=path)
    html_text = inject_public_home_intro(html_text, path=path, payload=home_intro_payload)
    html_text = inject_public_share_runtime(html_text)
    return HTMLResponse(html_text)


def register_public_static_routes(app: FastAPI) -> None:
    @app.get(PUBLIC_ADS_TXT_PATH)
    @app.get(f"{PUBLIC_BASE_PATH}/ads.txt")
    def public_ads_txt():
        return RedirectResponse(url=PUBLIC_ADS_TXT_REDIRECT_URL, status_code=301)

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




