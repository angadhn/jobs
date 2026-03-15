"""
Fetch job listings from frontier AI labs via their public job board APIs.

Uses Greenhouse API (Anthropic, DeepMind) and Ashby API (OpenAI).
Saves raw JSON to lab_jobs/<lab>_raw.json (cached, one file per lab).

Usage:
    uv run python scrape_labs.py                  # fetch all labs
    uv run python scrape_labs.py --lab anthropic   # fetch one lab
    uv run python scrape_labs.py --force           # re-fetch ignoring cache
"""

import argparse
import json
import os
import httpx


OUTPUT_DIR = "lab_jobs"

# ---------------------------------------------------------------------------
# Per-lab fetchers (using public job board APIs)
# ---------------------------------------------------------------------------

def fetch_greenhouse(board_name, lab_name):
    """Fetch jobs from Greenhouse job board API."""
    url = f"https://boards-api.greenhouse.io/v1/boards/{board_name}/jobs?content=false"
    resp = httpx.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    jobs = []
    for j in data.get("jobs", []):
        # Extract department from departments array
        depts = j.get("departments", [])
        team = depts[0]["name"] if depts else ""

        # Extract location
        location = j.get("location", {}).get("name", "")

        jobs.append({
            "title": j["title"],
            "team": team,
            "location": location,
            "url": j.get("absolute_url", ""),
            "lab": lab_name,
        })
    return jobs


def fetch_ashby(board_name, lab_name):
    """Fetch jobs from Ashby job board API."""
    url = f"https://api.ashbyhq.com/posting-api/job-board/{board_name}"
    resp = httpx.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    jobs = []
    for j in data.get("jobs", []):
        jobs.append({
            "title": j["title"],
            "team": j.get("departmentName", "") or j.get("department", ""),
            "location": j.get("locationName", "") or j.get("location", ""),
            "url": j.get("jobUrl", ""),
            "lab": lab_name,
        })
    return jobs


def fetch_anthropic():
    return fetch_greenhouse("anthropic", "anthropic")

def fetch_deepmind():
    return fetch_greenhouse("deepmind", "deepmind")

def fetch_openai():
    return fetch_ashby("openai", "openai")


LABS = {
    "anthropic": fetch_anthropic,
    "deepmind": fetch_deepmind,
    "openai": fetch_openai,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch AI lab job listings")
    parser.add_argument("--lab", choices=list(LABS.keys()), help="Fetch a single lab")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cached")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    labs_to_fetch = [args.lab] if args.lab else list(LABS.keys())

    # Check cache
    to_fetch = []
    for lab in labs_to_fetch:
        out_path = f"{OUTPUT_DIR}/{lab}_raw.json"
        if not args.force and os.path.exists(out_path):
            with open(out_path) as f:
                cached = json.load(f)
            print(f"  CACHED {lab} ({len(cached)} jobs)")
            continue
        to_fetch.append(lab)

    if not to_fetch:
        print("Nothing to fetch — all cached.")
        return

    print(f"\nFetching {len(to_fetch)} labs via job board APIs...\n")

    for lab in to_fetch:
        out_path = f"{OUTPUT_DIR}/{lab}_raw.json"
        print(f"  [{lab}] Fetching...", end=" ", flush=True)

        fetcher = LABS[lab]
        try:
            jobs = fetcher()

            # Deduplicate by title+location
            seen = set()
            unique_jobs = []
            for job in jobs:
                key = (job["title"], job.get("location", ""))
                if key not in seen:
                    seen.add(key)
                    unique_jobs.append(job)

            with open(out_path, "w") as f:
                json.dump(unique_jobs, f, indent=2)

            print(f"OK — {len(unique_jobs)} jobs")

        except Exception as e:
            print(f"ERROR: {e}")

    # Summary
    total = 0
    for lab in labs_to_fetch:
        out_path = f"{OUTPUT_DIR}/{lab}_raw.json"
        if os.path.exists(out_path):
            with open(out_path) as f:
                n = len(json.load(f))
            total += n
            print(f"  {lab}: {n} jobs")
    print(f"\nTotal: {total} jobs across {len(labs_to_fetch)} labs")


if __name__ == "__main__":
    main()
