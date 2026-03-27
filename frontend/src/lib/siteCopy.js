import siteCopy from "../content/siteCopy.json";

export const SITE_COPY = siteCopy;

export const SITE_NAME = SITE_COPY.site.name;
export const HOME_PAGE_TITLE = SITE_COPY.site.home_page_title;
export const HOME_PAGE_DESCRIPTION = SITE_COPY.site.home_page_description;

export const HOME_HERO_COPY = SITE_COPY.home.hero;
export const HOME_LIST_CTA_LABEL = SITE_COPY.home.list_cta_label;

export const FEATURED_CONTENT_SECTION = SITE_COPY.home.featured_section;
export const FEATURED_CONTENT_ITEMS = SITE_COPY.home.featured_items;

export const METHOD_SUMMARY_SECTION = SITE_COPY.home.method_section;
export const METHOD_SUMMARY_STEPS = SITE_COPY.home.method_steps;

export const BEGINNER_GUIDE_SECTION = SITE_COPY.home.beginner_guide_section;
export const BEGINNER_GUIDE_LINKS = SITE_COPY.home.beginner_guide_links;

export const PUBLIC_PAGE_CONTENT = Object.fromEntries(
  Object.values(SITE_COPY.public_pages).map((page) => [page.path, page]),
);
