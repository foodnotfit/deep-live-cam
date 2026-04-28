#!/usr/bin/env python3
"""
Download public-domain face photos for the Family Day demo.

Uses Wikipedia's REST API (https://en.wikipedia.org/api/rest_v1/page/summary/)
to fetch the canonical lead image for each historical/cultural figure.
Most of these are CC-BY-SA or public domain — fine for a contained
educational demo. The script:

  1. Skips entries that already exist in faces/ (so re-runs are safe)
  2. Fetches the lead image URL from Wikipedia's API
  3. Downloads the image to faces/<slug>.jpg
  4. Reports success / failure per entry

Re-run anytime to pick up missing entries; existing files are left alone.
"""

import os
import sys
import time
import json
import urllib.request
import urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve().parent
FACES_DIR = HERE / "faces"
FACES_DIR.mkdir(exist_ok=True)

# Curated list — heavily weighted toward current pop culture (music, actors,
# athletes) for maximum kid-recognition, with a small set of high-impact
# historical icons. All sourced from Wikipedia's CC-licensed lead images.
# Format: (filename_slug, wikipedia_page_title, why_kids_recognize)
ENTRIES = [
    # === Music — English-language ===
    ("bad_bunny",         "Bad_Bunny",                "Latin music superstar, kid-iconic"),
    ("beyonce",           "Beyoncé",                  "global icon"),
    ("billie_eilish",     "Billie_Eilish",            "huge with kids/teens"),
    ("drake",             "Drake_(musician)",         "rapper kids know"),
    ("justin_bieber",     "Justin_Bieber",            "still recognizable to kids"),
    ("selena_gomez",      "Selena_Gomez",             "Disney + music + acting"),
    ("olivia_rodrigo",    "Olivia_Rodrigo",           "huge with younger kids"),
    ("bruno_mars",        "Bruno_Mars",               "broad appeal across ages"),
    ("adele",             "Adele",                    "globally iconic voice"),
    ("harry_styles",      "Harry_Styles",             "huge teen following"),
    ("the_weeknd",        "The_Weeknd",               "current charts staple"),
    ("rihanna",           "Rihanna",                  "music + Fenty cultural icon"),
    ("ariana_grande",     "Ariana_Grande",            "music + voice acting"),
    ("doja_cat",          "Doja_Cat",                 "TikTok-era kid favorite"),
    ("dua_lipa",          "Dua_Lipa",                 "global pop, Barbie soundtrack"),
    ("lizzo",             "Lizzo",                    "personality kids love"),
    ("post_malone",       "Post_Malone",              "huge with older kids"),
    ("kendrick_lamar",    "Kendrick_Lamar",           "Grammy-winning rapper"),

    # === Music — Latin / Spanish-language ===
    ("shakira",           "Shakira",                  "Latin music icon, global"),
    ("karol_g",           "Karol_G",                  "Colombian Latin pop star"),
    ("rosalia",           "Rosalía",                  "Spanish flamenco/pop"),
    ("j_balvin",          "J_Balvin",                 "Colombian reggaeton star"),

    # === Music — K-pop (huge with kids globally) ===
    ("lisa_blackpink",    "Lisa_(rapper)",            "Blackpink, kid-iconic in K-pop"),
    ("jungkook",          "Jungkook",                 "BTS lead vocalist"),
    ("rm_bts",            "RM_(rapper)",              "BTS leader"),

    # === Actors kids know from current movies ===
    ("zendaya",           "Zendaya",                  "Spider-Man, Euphoria, Dune"),
    ("tom_holland",       "Tom_Holland",              "Spider-Man for current kids"),
    ("dwayne_johnson",    "Dwayne_Johnson",           "The Rock — kids love him"),
    ("will_smith",        "Will_Smith",               "Aladdin, kids' movies"),
    ("pedro_pascal",      "Pedro_Pascal",             "Mandalorian, Last of Us"),
    ("margot_robbie",     "Margot_Robbie",            "Barbie, huge kid-recognition"),
    ("ryan_reynolds",     "Ryan_Reynolds",            "Deadpool, kid-friendly humor"),
    ("tom_hanks",         "Tom_Hanks",                "Toy Story voice — family appeal"),

    # === Athletes — universally kid-engaging across regions ===
    ("lebron_james",      "LeBron_James",             "basketball icon"),
    ("cristiano_ronaldo", "Cristiano_Ronaldo",        "global soccer star"),
    ("lionel_messi",      "Lionel_Messi",             "global soccer star"),
    ("serena_williams",   "Serena_Williams",          "tennis legend"),
    ("simone_biles",      "Simone_Biles",             "gymnast, Olympic icon"),

    # === Historical icons (kept short — only the most visually recognizable) ===
    ("einstein",          "Albert_Einstein",          "iconic hair / 'I'm a genius!'"),
    ("lincoln",           "Abraham_Lincoln",          "beard + top hat, very recognizable"),
    ("mlk",               "Martin_Luther_King_Jr.",   "kids learn about him"),
    ("marie_curie",       "Marie_Curie",              "first female Nobel winner"),
    ("frida_kahlo",       "Frida_Kahlo",              "distinctive, strong female artist"),
    ("bruce_lee",         "Bruce_Lee",                "martial artist, very cool factor"),
    ("mozart",            "Wolfgang_Amadeus_Mozart",  "child prodigy composer"),
    ("gandhi",            "Mahatma_Gandhi",           "kids learn about him"),
]

# Wikimedia's policy requires a real, identifying User-Agent.
# https://meta.wikimedia.org/wiki/User-Agent_policy
USER_AGENT = ("DeepLiveCamFamilyDay/1.0 (educational classroom demo; "
              "https://github.com/foodnotfit/deep-live-cam) Python-urllib")


def _request_with_retry(url: str, timeout: int = 30, max_retries: int = 4) -> bytes:
    """Fetch a URL, respecting 429 Retry-After headers and backing off."""
    delay = 2.0  # base wait before retrying after a 429
    for attempt in range(max_retries):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                # Honor server-suggested Retry-After if present, else exponential backoff
                retry_after = e.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = float(retry_after)
                else:
                    wait = delay
                print(f"        rate-limited; waiting {wait:.0f}s before retry "
                      f"(attempt {attempt+2}/{max_retries})...")
                time.sleep(wait)
                delay = min(delay * 2, 60)  # cap exponential at 60s
                continue
            raise


def fetch_image_url(page_title: str) -> str:
    """Query Wikipedia's REST API for the canonical lead image URL of a page."""
    title = urllib.parse.quote(page_title)
    api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    data = json.loads(_request_with_retry(api_url, timeout=15))
    # Prefer originalimage (full resolution) over thumbnail
    if "originalimage" in data and "source" in data["originalimage"]:
        return data["originalimage"]["source"]
    if "thumbnail" in data and "source" in data["thumbnail"]:
        return data["thumbnail"]["source"]
    raise RuntimeError(f"No lead image found for '{page_title}'")


def download(url: str, dest: Path) -> int:
    """Download a URL to disk. Returns bytes written."""
    data = _request_with_retry(url, timeout=30)
    dest.write_bytes(data)
    return len(data)


def main() -> int:
    print("=" * 60)
    print(f" Downloading public-domain faces to {FACES_DIR}")
    print("=" * 60)

    successes, skipped, failures = [], [], []

    for slug, page, note in ENTRIES:
        dest_jpg = FACES_DIR / f"{slug}.jpg"
        dest_png = FACES_DIR / f"{slug}.png"
        if dest_jpg.exists() or dest_png.exists():
            print(f"  [skip ] {slug:22s}  already exists")
            skipped.append(slug)
            continue
        try:
            url = fetch_image_url(page)
            # Use the URL's actual extension for the saved file
            ext = url.rsplit(".", 1)[-1].lower()
            if ext not in ("jpg", "jpeg", "png", "webp"):
                ext = "jpg"
            dest = FACES_DIR / f"{slug}.{ext}"
            n = download(url, dest)
            print(f"  [  OK ] {slug:22s}  {note} ({n//1024} KB)")
            successes.append(slug)
            time.sleep(2.0)  # polite rate limiting — Wikimedia 429s aggressively
        except Exception as e:
            print(f"  [FAIL ] {slug:22s}  {type(e).__name__}: {str(e)[:60]}")
            failures.append((slug, str(e)))

    print()
    print("=" * 60)
    print(f" Summary:  {len(successes)} downloaded, "
          f"{len(skipped)} skipped, {len(failures)} failed")
    print("=" * 60)

    if failures:
        print("\n  Failed entries (Wikipedia page may have moved or no lead image):")
        for slug, err in failures:
            print(f"    {slug}: {err[:80]}")
        print("\n  You can manually find images for these and save them to faces/ as")
        print("  <slug>.jpg. Wikipedia's image is usually at the top right of the page.")

    if successes:
        print(f"\n  Next: run validate_faces.py to confirm all sources actually swap well.")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
