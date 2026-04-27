import requests
from bs4 import BeautifulSoup
import os
from PIL import Image
from io import BytesIO
import time

# =========================
# SCRAPE DUKE SPECIES
# =========================
url = "https://gardens.duke.edu/gardens/plants/"
html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text
soup = BeautifulSoup(html, "html.parser")

plants = []

for item in soup.select(".list-container"):
    common = item.select_one(".list-item-title")
    latin = item.select_one(".item-excerpt i")

    common_name = common.get_text(strip=True) if common else None
    latin_name = latin.get_text(strip=True) if latin else None

    if common_name:
        plants.append((common_name, latin_name))

print(f"Total plants scraped: {len(plants)}")


# =========================
# EXACT MATCH CHECK
# =========================
def is_exact_species(species_name):
    try:
        r = requests.get(
            "https://api.inaturalist.org/v1/taxa",
            params={"q": species_name, "rank": "species"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
        )
        results = r.json().get("results", [])
    except:
        return False

    for t in results:
        # strict: must match canonical name exactly
        if t.get("name", "").lower() == species_name:
            return True

    return False


# =========================
# BUILD CLEAN SPECIES LIST
# =========================
raw_species = [
    " ".join(latin.split()[:2]).lower()
    for _, latin in plants
    if latin and len(latin.split()) >= 2
]

raw_species = list(set(raw_species))
print(f"Raw species: {len(raw_species)}")

species_list = [s for s in raw_species if is_exact_species(s)]

print(f"Valid species (exact match): {len(species_list)}")


# =========================
# DOWNLOAD FUNCTION
# =========================
def download_species_images(species_name, max_images=400, save_dir="../../data"):
    species_dir = os.path.join(save_dir, species_name.replace(" ", "_"))
    os.makedirs(species_dir, exist_ok=True)

    existing = len(os.listdir(species_dir))
    if existing >= max_images:
        print(f"Skipping {species_name} ({existing} already)")
        return existing

    count = existing
    page = 1

    while count < max_images:
        try:
            r = requests.get(
                "https://api.inaturalist.org/v1/observations",
                params={
                    "taxon_name": species_name,
                    "photos": True,
                    "per_page": 200,
                    "page": page,
                    "quality_grade": "research"
                },
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10
            )
            data = r.json()
        except:
            print(f"Failed API: {species_name}")
            break

        results = data.get("results", [])
        if not results:
            break  # no more pages

        for obs in results:
            if count >= max_images:
                break

            if obs.get("photos"):
                try:
                    url = obs["photos"][0]["url"].replace("square", "medium")
                    img_data = requests.get(url, timeout=10).content
                    img = Image.open(BytesIO(img_data)).convert("RGB")
                    img.save(os.path.join(species_dir, f"{count}.jpg"))
                    count += 1
                except:
                    pass

        page += 1

    return count


# =========================
# DOWNLOAD LOOP WITH PROGRESS
# =========================
start_time = time.time()

total = len(species_list)
completed = 0

for i, sp in enumerate(species_list, 1):
    t0 = time.time()

    print(f"\n[{i}/{total}] Downloading: {sp}")

    count = download_species_images(sp, max_images=250)

    elapsed = time.time() - t0
    completed += 1

    avg_time = (time.time() - start_time) / completed
    remaining = total - completed
    eta = remaining * avg_time / 60

    print(f"Saved: {count} images")
    print(f"Time for this: {elapsed:.1f}s | ETA: {eta:.1f} min")


print("\nDone.")
