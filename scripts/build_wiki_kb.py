# NOTE: this file may not run as intended as the contents it references are no longer necessary. 
# The file still has useful functionality though. 
"""
Utility script to build or extend a Wikipedia-based knowledge base (KB) of species.

Functionality:
- Reads a list of species names from `species_list.json`
- Fetches summaries via the Wikipedia REST API
- Appends new entries to `wiki_kb.json` while avoiding duplicates
- Can resume from an existing KB file

Notes:
- If `species_list.json` is missing, the script will fail unless modified
- The script is still useful as a standalone Wikipedia fetch utility (`get_wikipedia_summary`)
- Paths assume execution from the project root; adjust if running from another directory
"""

import json
import requests
import os

# load species list
with open("species_list.json") as f:
    species_list = json.load(f)

# ✅ load existing KB if it exists
if os.path.exists("wiki_kb.json"):
    with open("wiki_kb.json") as f:
        kb = json.load(f)
else:
    kb = []

# build set of already fetched species
existing_species = set(item["species"] for item in kb)


def get_wikipedia_summary(species):
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{species.replace(' ', '_')}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)

        if r.status_code != 200:
            return None

        data = r.json()

        return {
            "title": data.get("title", species),
            "summary": data.get("extract", ""),
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", "")
        }

    except:
        return None


for i, sp in enumerate(species_list, 1):
    if sp in existing_species:
        print(f"[{i}/{len(species_list)}] Skipping {sp} (already exists)")
        continue

    print(f"[{i}/{len(species_list)}] Fetching: {sp}")

    info = get_wikipedia_summary(sp)

    if info and info["summary"]:
        kb.append({
            "species": sp,
            "text": f"{info['title']}: {info['summary']}",
            "url": info["url"]
        })

print("\nTotal entries:", len(kb))

# save
with open("wiki_kb.json", "w") as f:
    json.dump(kb, f, indent=2)