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

# Curated list — diverse historical/cultural figures kids learn about in school,
# all with reliable Creative Commons or public-domain Wikipedia portraits.
# Format: (filename_slug, wikipedia_page_title, why_kids_recognize)
ENTRIES = [
    # Scientists
    ("einstein",          "Albert_Einstein",          "iconic hair / 'I'm a genius!'"),
    ("marie_curie",       "Marie_Curie",              "first female Nobel winner"),
    ("nikola_tesla",      "Nikola_Tesla",             "kids love the Tesla connection"),
    ("charles_darwin",    "Charles_Darwin",           "evolution / iconic beard"),
    ("stephen_hawking",   "Stephen_Hawking",          "well-known modern scientist"),

    # Civil rights & social leaders
    ("mlk",               "Martin_Luther_King_Jr.",   "kids learn about him"),
    ("rosa_parks",        "Rosa_Parks",               "civil rights icon"),
    ("gandhi",            "Mahatma_Gandhi",           "kids learn about him"),
    ("malala",            "Malala_Yousafzai",         "young + relatable to kids"),

    # Historical leaders
    ("lincoln",           "Abraham_Lincoln",          "beard + top hat, very recognizable"),
    ("george_washington", "George_Washington",        "first US president"),
    ("queen_elizabeth_1", "Elizabeth_I",              "regal Tudor portrait"),

    # Artists
    ("frida_kahlo",       "Frida_Kahlo",              "distinctive, strong female artist"),
    ("picasso",           "Pablo_Picasso",            "famous painter"),

    # Adventurers / firsts
    ("amelia_earhart",    "Amelia_Earhart",           "pilot, brave, female"),
    ("neil_armstrong",    "Neil_Armstrong",           "first man on the moon"),

    # Cultural icons
    ("bruce_lee",         "Bruce_Lee",                "martial artist, very cool factor"),
    ("mozart",            "Wolfgang_Amadeus_Mozart",  "child prodigy composer"),

    # Authors
    ("shakespeare",       "William_Shakespeare",      "kids learn about him"),
    ("mark_twain",        "Mark_Twain",               "white hair + mustache"),

    # --- Current pop music / culture (CC-licensed Wikipedia event photos) ---
    # Kids respond hardest to music artists they hear on TikTok / streaming.
    # Mix of English-language and Latin/Spanish-language artists for diversity.
    ("bad_bunny",         "Bad_Bunny",                "Latin music superstar, kid-iconic"),
    ("beyonce",           "Beyoncé",                  "global icon"),
    ("billie_eilish",     "Billie_Eilish",            "huge with kids/teens"),
    ("drake",             "Drake_(musician)",         "rapper kids know"),
    ("justin_bieber",     "Justin_Bieber",            "still recognizable to kids"),
    ("selena_gomez",      "Selena_Gomez",             "Disney + music + acting"),
    ("olivia_rodrigo",    "Olivia_Rodrigo",           "huge with younger kids"),
    ("bruno_mars",        "Bruno_Mars",               "broad appeal across ages"),
    ("lizzo",             "Lizzo",                    "personality kids love"),
    ("adele",             "Adele",                    "globally iconic voice"),
    ("harry_styles",      "Harry_Styles",             "huge teen following"),
    ("shakira",           "Shakira",                  "Latin music icon, global"),
    ("the_weeknd",        "The_Weeknd",               "current charts staple"),
    ("rihanna",           "Rihanna",                  "music + Fenty cultural icon"),
    ("ariana_grande",     "Ariana_Grande",            "music + voice acting"),

    # Actors kids know from current movies (CC-licensed event/portrait photos)
    ("zendaya",           "Zendaya",                  "Spider-Man, Euphoria, Dune"),
    ("tom_holland",       "Tom_Holland",              "Spider-Man for current kids"),
    ("tom_hanks",         "Tom_Hanks",                "Toy Story voice — broad family appeal"),

    # Athletes — universally kid-engaging across regions
    ("lebron_james",      "LeBron_James",             "basketball icon"),
    ("cristiano_ronaldo", "Cristiano_Ronaldo",        "global soccer star"),
    ("lionel_messi",      "Lionel_Messi",             "global soccer star"),
    ("serena_williams",   "Serena_Williams",          "tennis legend"),
    ("simone_biles",      "Simone_Biles",             "gymnast, Olympic icon"),
]

USER_AGENT = ("DeepLiveCamFamilyDay/1.0 "
              "(educational demo; contact: local installation)")


def fetch_image_url(page_title: str) -> str:
    """Query Wikipedia's REST API for the canonical lead image URL of a page."""
    title = urllib.parse.quote(page_title)
    api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    req = urllib.request.Request(api_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    # Prefer originalimage (full resolution) over thumbnail
    if "originalimage" in data and "source" in data["originalimage"]:
        return data["originalimage"]["source"]
    if "thumbnail" in data and "source" in data["thumbnail"]:
        return data["thumbnail"]["source"]
    raise RuntimeError(f"No lead image found for '{page_title}'")


def download(url: str, dest: Path) -> int:
    """Download a URL to disk. Returns bytes written."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
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
            time.sleep(0.5)  # polite rate limiting on Wikipedia API
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
