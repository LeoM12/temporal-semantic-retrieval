"""
Builds a balanced, timestamped article corpus based on a keyword list for various topics.

Source: stanford-oval/ccnews
  - CC-News, cleaned and deduplicated, 2016 to June 2024
  - partitioned by crawl year: configs "2016".."2024" (+ "default")
  - fields include: plain_text, published_date (YYYY-MM-DD), article_title, tags,
    categories, language, language_score, requested_url, responded_url, ...

Strategy:
one streaming pass per year config. Each article is routed to exactly one
topic (if it matches) until the cell (topic, year) reaches its quota.
Streaming instead of downloading: the corpus is far too large for local filtering.

Usage:
  python build_ccnews_dataset.py             # all years in YEARS
  python build_ccnews_dataset.py 2020        # only 2020  (split runs per year)
  python build_ccnews_dataset.py 2019 2020   # multiple years
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import datasets
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("ccnews")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# --------------------------------------------------------------------------
# 1) CONFIG -- topics and keywords for matching algorithm
# --------------------------------------------------------------------------
TOPICS: dict[str, list[str]] = {
    "financial_markets": [
        "Federal Reserve", "the Fed", "FOMC", "central bank",
        "interest rate", "interest rates", "rate hike", "rate cut",
        "rate rise", "rate decision", "monetary policy", "ECB",
        "European Central Bank", "Bank of England", "Bank of Japan",
        "benchmark rate", "borrowing costs", "stock market", "stock markets",
        "stock exchange", "Wall Street", "Dow Jones", "Dow",
        "S&P 500", "Nasdaq", "NYSE", "FTSE", "FTSE 100",
        "equities", "equity market", "stock index", "blue chip",
        "share price", "market rally", "market selloff", "trading day",
        "bond market", "bond yield", "treasury yield", "government bonds",
        "Treasury", "sovereign debt", "gilts", "financial markets",
        "market volatility", "investor sentiment",
    ],

    "internet_platforms": [
        "Facebook", "Twitter", "Instagram", "YouTube", "Snapchat",
        "LinkedIn", "Reddit", "WhatsApp", "Pinterest", "Netflix",
        "Spotify", "streaming", "streaming service", "livestream",
        "social media", "social network", "online platform", "internet",
        "broadband", "search engine", "web browser", "smartphone app",
        "mobile app", "app store", "online video", "podcast",
        "data privacy", "online privacy", "data breach",
        "online safety", "content moderation",
    ],

    "uk_football": [
        "Premier League", "FA Cup", "EFL Championship", "League Cup",
        "Carabao Cup", "Community Shield", "English Football League",
        "Manchester United", "Man United", "Man Utd", "Manchester City",
        "Man City", "Liverpool FC", "Chelsea FC", "Arsenal FC",
        "Tottenham", "Spurs", "Everton", "Leicester City", "West Ham",
        "Newcastle United", "Aston Villa", "Wolverhampton", "Wolves",
        "Leeds United", "Brighton", "Crystal Palace", "Southampton FC",
        "Burnley", "Watford FC", "Norwich City", "Brentford", "Fulham FC",
        "Bournemouth", "Nottingham Forest", "Sheffield United", "West Brom",
        "Stoke City", "Swansea City", "Hull City", "Middlesbrough",
        "Cardiff City", "transfer window", "Premier League table",
        "relegation", "Wembley", "English football",
    ],
}

DATASET = "stanford-oval/ccnews"

#2016 subset is available but contains about a tenth of the amount of articles than the other subsets,
#so it was left out at the moment
YEARS = list(range(2017, 2025))

TARGET_PER_CELL = 3700
LANG = "en"
MIN_LANG_SCORE = 0.80      # optional confidence filter (dataset already has >= 0.70)
OUT_DIR = Path("testing_data")
LOG_EVERY = 50_000         # log streaming progress every N rows
MAX_RETRIES = 5



# --------------------------------------------------------------------------
# 2) SCHEMA -- only the fields needed for the experiments later
# --------------------------------------------------------------------------
@dataclass
class Article:
    id: str
    topic: str
    year: int
    query: str            # the keyword that matched this article
    article_title: str
    plain_text: str
    published_date: str   # YYYY-MM-DD (original from CC-News)
    tags: str
    categories: str


def stable_id(url: str, article_title: str, text: str) -> str:
    # Returns a SHA-1 hash of the URL (preferred) or article_title+text snippet as a stable article ID.
    basis = url.strip().lower() if url else (article_title + text[:200])
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# 3) KEYWORD MATCHING -- First match of a topic to the title is used
# --------------------------------------------------------------------------
def compile_patterns(topics: dict[str, list[str]]) -> dict[str, re.Pattern]:
    # Compiles each topic's keyword list into a single OR word-boundary regex.
    return {
        t: re.compile(r"\b(" + "|".join(re.escape(k.lower()) for k in kws) + r")\b")
        for t, kws in topics.items()
    }


def match_topic(patterns: dict[str, re.Pattern], article_title: str,
                needed: dict[str, int]) -> tuple[str | None, str | None]:
    # Returns the first topic whose keyword appears in 'article_title',
    # along with the matched keyword. Returns (None, None) if no topic matches.
    for topic, pat in patterns.items():
        if needed.get(topic, 0) <= 0:
            continue
        m = pat.search(article_title)
        if m:
            return topic, m.group(1)
    return None, None


# --------------------------------------------------------------------------
# 4) PERSISTENCE + RESUMABILITY (in case script crashes during runtime)
# --------------------------------------------------------------------------
class Store:
    """One JSONL file per topic. On startup, existing IDs and counts are loaded
    so that a re-run (or a separate per-year run) continues seamlessly."""

    def __init__(self, out_dir: Path):
        # Creates the output directory and loads any already-collected articles.
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.seen_ids: set[str] = set()
        self.cell_counts: dict[tuple[str, int], int] = {}
        self._load_existing()

    def _load_existing(self):
        # Scans existing JSONL files to rebuild seen_ids and cell_counts for resumability.
        for fp in self.out_dir.glob("*.jsonl"):
            with fp.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.seen_ids.add(rec["id"])
                    key = (rec["topic"], rec["year"])
                    self.cell_counts[key] = self.cell_counts.get(key, 0) + 1
        if self.seen_ids:
            log.info("Resume: %d articles already collected.", len(self.seen_ids))

    def count(self, topic: str, year: int) -> int:
        # Returns the number of articles already collected for a given (topic, year) cell.
        return self.cell_counts.get((topic, year), 0)

    def is_dup(self, art: Article) -> bool:
        # Returns True if this article's ID has already been stored.
        return art.id in self.seen_ids

    def add(self, art: Article):
        # Appends an article as a JSON line to its topic file and updates in-memory state.
        with (self.out_dir / f"{art.topic}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(art), ensure_ascii=False) + "\n")
        self.seen_ids.add(art.id)
        self.cell_counts[(art.topic, art.year)] = self.count(art.topic, art.year) + 1

    def write_manifest(self):
        # Writes a manifest.json with the actual article count per cell for auditing corpus balance.
        m = {
            "target_per_cell": TARGET_PER_CELL,
            "lang": LANG,
            "cells": {f"{t}|{y}": n for (t, y), n in sorted(self.cell_counts.items())},
            "total": sum(self.cell_counts.values()),
        }
        (self.out_dir / "manifest.json").write_text(
            json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8"
        )


# --------------------------------------------------------------------------
# 5) PROCESS ONE YEAR  (one streaming pass, multi-topic routing)
# --------------------------------------------------------------------------
def process_year(year: int, patterns: dict[str, re.Pattern], store: Store):
    # Streams the CC-News config for 'year' and routes matching articles into
    # their (topic, year) cells until all quotas for this year are filled.
    needed = {t: TARGET_PER_CELL - store.count(t, year) for t in TOPICS}
    if all(n <= 0 for n in needed.values()):
        log.info("[%d] all cells already full.", year)
        return

    log.info("[%d] starting stream. Still needed: %s", year, needed)

    ds_state = None

    for attempt in range(MAX_RETRIES):
        try:
            ds = load_dataset(DATASET, name=str(year), split="train", streaming=True)
            if(ds_state != None):
                ds.load_state_dict(ds_state)

            scanned = 0
            added = 0
            for row in ds:
                scanned += 1
                if scanned % 950_000 == 0:
                    ds_state = ds.state_dict()
                    log.info("Reached limit of 950_000. Checkpoint saved. Rebuilding connection...")
                    time.sleep(15)
                    ds = load_dataset(DATASET, name=str(year), split="train", streaming=True)
                    ds.load_state_dict(ds_state)
                    log.info("Starting stream at last checkpoint...")
                    continue
                if scanned % LOG_EVERY == 0:
                    open_cells = {t: n for t, n in needed.items() if n > 0}
                    log.info("[%d] scanned=%d  +%d  open=%s", year, scanned, added, open_cells)

                # Language filter
                if row.get("language") != LANG:
                    continue
                if (row.get("language_score") or 0.0) < MIN_LANG_SCORE:
                    continue

                # Publish date filter
                pub = row.get("published_date") or ""
                if len(pub) < 4 or not pub[:4].isdigit() or int(pub[:4]) != year:
                    continue

                # Keyword match: first topic with an open quota wins
                article_title = row.get("title") or ""
                topic, kw = match_topic(patterns, article_title.lower(), needed)
                if topic is None:
                    continue

                url = row.get("responded_url") or row.get("requested_url") or ""
                text = row.get("plain_text") or ""
                art = Article(
                    id=stable_id(url, article_title, text),
                    topic=topic, year=year, query=kw,
                    article_title=article_title, plain_text=text, published_date=pub,
                    tags=row.get("tags") or "", categories=row.get("categories") or "",
                )
                if store.is_dup(art):
                    continue

                store.add(art)
                added += 1
                needed[topic] -= 1
                if all(n <= 0 for n in needed.values()):
                    log.info("[%d] all cells full after %d rows.", year, scanned)
                    break
            break
        except (RuntimeError, ConnectionError, OSError) as e:
            if attempt < MAX_RETRIES - 1:
                wait = 10 * (attempt + 1)
                print(f"  Connection error on {year} (attempt {attempt + 1}/{MAX_RETRIES}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Failed to sample {year} after {MAX_RETRIES} attempts, skipping.")
    
    store.write_manifest()
    short = {t: store.count(t, year) for t in TOPICS if store.count(t, year) < TARGET_PER_CELL}
    if short:
        log.warning("[%d] shortfall (subset exhausted, quota not reached): %s", year, short)
    log.info("[%d] done. +%d articles (scanned %d).", year, added, scanned)


def main():
    # Entry point: parses optional year arguments, then runs process_year for each.
    years = [int(a) for a in sys.argv[1:]] or YEARS
    patterns = compile_patterns(TOPICS)
    store = Store(OUT_DIR)
    for y in years:
        process_year(y, patterns, store)
    store.write_manifest()
    log.info("Total: %d articles across all cells.", sum(store.cell_counts.values()))


if __name__ == "__main__":
    main()