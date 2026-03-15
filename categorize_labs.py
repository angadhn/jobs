"""
Categorize AI lab job postings into skill domains.

Uses OpenRouter LLM API when OPENROUTER_API_KEY is set, otherwise falls back
to keyword-based categorization. Reads lab_jobs/*_raw.json, outputs lab_jobs.json.

Usage:
    uv run python categorize_labs.py                              # auto (LLM or keywords)
    uv run python categorize_labs.py --model google/gemini-3-flash-preview
    uv run python categorize_labs.py --force                      # re-categorize all
"""

import argparse
import json
import os
import re
import time
import httpx
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "google/gemini-3-flash-preview"
OUTPUT_FILE = "lab_jobs.json"
INPUT_DIR = "lab_jobs"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
BATCH_SIZE = 25

DOMAINS = {
    "ai_research": "Research Scientist, Interpretability, Pretraining, ML Engineer (Research)",
    "ai_safety": "Alignment, Red Team, Safeguards, Responsible AI",
    "infrastructure": "GPU, Inference, Networking, Systems, Platform, SRE, Cloud",
    "product_engineering": "Product Engineer, API, Consumer Product, Frontend, Backend, Full-stack",
    "enterprise_deployment": "Forward Deployed Engineer, Solutions Architect, Technical Success",
    "sales_business": "Account Executive, Partnerships, Business Development, Revenue",
    "security": "Offensive Security, Detection & Response, AppSec, Security Engineer",
    "policy_trust": "Public Policy, Trust & Safety, Government Relations, Ethics",
    "data_centers": "Data Center Engineering, Supply Chain, Hardware, Facilities",
    "operations": "Finance, Legal, HR, Recruiting, People Operations, Workplace",
    "marketing_comms": "Marketing, Communications, Brand, Content, Events",
    "design": "Product Design, UX, Visual Design, Design Systems",
    "program_management": "TPM, Program Manager, Chief of Staff, Strategy & Ops",
}

SYSTEM_PROMPT = """\
You are categorizing AI company job postings into skill domains.

Here are the available domains and what they include:

""" + "\n".join(f"- **{k}**: {v}" for k, v in DOMAINS.items()) + """

You will receive a JSON array of job objects, each with an "id", "title", "team", and "lab" field.

For each job, assign:
1. **domain**: exactly one of the domain keys listed above
2. **skills**: 1-3 short skill tags (e.g. "python", "LLM fine-tuning", "enterprise sales")

Respond with ONLY a JSON array of objects in this exact format, no other text:
[
  {"id": 0, "domain": "domain_key", "skills": ["skill1", "skill2"]},
  ...
]

Match every job in the input. Use your best judgment for ambiguous titles.
"""


# ---------------------------------------------------------------------------
# Keyword-based categorizer (fallback when no API key)
# ---------------------------------------------------------------------------

KEYWORD_RULES = [
    # (domain, title_patterns, team_patterns)
    ("ai_safety", [
        r"safety", r"alignment", r"red.team", r"safeguard", r"responsible\s*ai",
        r"trust\s*&?\s*safety", r"frontier.*(risk|threat)", r"societal.impact",
        r"interpretab", r"honesty",
    ], [
        r"safety", r"alignment", r"safeguard", r"responsible",
    ]),
    ("ai_research", [
        r"research\s*(scientist|engineer|lead|manager)", r"pretraining", r"pre-training",
        r"post-training", r"reward.model", r"scaling", r"(?:^|\b)ml\s*engineer",
        r"machine.learning.*(scientist|engineer|research)",
        r"(?:^|\b)(?:sr\.?|senior|staff|principal)\s*research",
    ], [
        r"research", r"pretraining", r"scaling",
    ]),
    ("security", [
        r"security", r"secops", r"appsec", r"offensive\s*sec", r"detection\s*&?\s*response",
        r"threat\s*(intel|detect|hunt)", r"incident\s*response", r"penetration",
        r"vulnerability",
    ], [
        r"security",
    ]),
    ("infrastructure", [
        r"\bgpu\b", r"inference\b", r"kernel\s*eng", r"networking\b", r"\bsre\b",
        r"site.reliability", r"platform\s*eng", r"systems?\s*eng", r"devops",
        r"cloud\s*eng", r"accelerator", r"compiler", r"distributed\s*sys",
        r"ml\s*infra", r"training\s*infra", r"compute\s*infra",
        r"performance\s*eng",
    ], [
        r"infrastructure", r"platform", r"systems",
    ]),
    ("data_centers", [
        r"data\s*center", r"facilities", r"supply\s*chain", r"hardware\s*eng",
        r"electrical\s*eng", r"mechanical\s*eng", r"construction",
        r"site\s*selection", r"capacity\s*plan",
    ], [
        r"data.center", r"facilities", r"supply.chain",
    ]),
    ("enterprise_deployment", [
        r"forward\s*deploy", r"solutions?\s*(?:architect|eng)", r"technical\s*success",
        r"customer\s*eng", r"implementation", r"field\s*eng",
        r"technical\s*account", r"customer\s*success\s*eng",
    ], [
        r"deployment", r"solutions", r"customer\s*success",
    ]),
    ("sales_business", [
        r"account\s*(?:exec|manage|coord|direct)", r"\bsales\b", r"business\s*dev",
        r"partnership", r"revenue", r"go.to.market", r"commercial",
        r"deal\s*desk", r"sales\s*(?:eng|ops|strategy)",
    ], [
        r"sales", r"business\s*dev", r"go.to.market", r"revenue",
    ]),
    ("product_engineering", [
        r"software\s*eng", r"full.stack", r"frontend", r"backend", r"web\s*eng",
        r"product\s*eng", r"mobile\s*eng", r"\bapi\b.*eng", r"(?:ios|android)\s*eng",
        r"developer\s*(?:experience|tools|productivity|education)",
    ], [
        r"product\s*eng", r"applied\s*ai", r"consumer", r"api\s*(?:team|platform)",
    ]),
    ("policy_trust", [
        r"polic(?:y|ies)", r"government\s*(?:relations|affairs)", r"regulatory",
        r"public\s*(?:policy|affairs)", r"ethic", r"compliance",
    ], [
        r"policy", r"government", r"regulatory", r"legal.*policy",
    ]),
    ("marketing_comms", [
        r"marketing", r"communicat", r"\bbrand\b", r"content\s*(?:writer|strat|manag|market)",
        r"social\s*media", r"events?\s*(?:manage|coord|plan)", r"\bpr\b.*(?:manager|special)",
        r"creative\s*(?:dir|manag|prod)",
    ], [
        r"marketing", r"communications", r"brand",
    ]),
    ("design", [
        r"product\s*design", r"\bux\b", r"\bui\b", r"visual\s*design",
        r"design\s*(?:system|eng|lead|manage)", r"graphic\s*design",
        r"interaction\s*design", r"user\s*(?:experience|research)",
    ], [
        r"design",
    ]),
    ("program_management", [
        r"\btpm\b", r"technical\s*program", r"program\s*manag",
        r"chief\s*of\s*staff", r"strategy\s*&?\s*ops", r"project\s*manag",
        r"business\s*ops",
    ], [
        r"program\s*manag", r"strategy.*ops",
    ]),
    ("operations", [
        r"\bhr\b", r"human\s*resource", r"recruit", r"talent\s*(?:acqui|part|manage)",
        r"people\s*(?:ops|partner|manage)", r"financ", r"accounting",
        r"\blegal\b", r"counsel", r"paralegal", r"workplace", r"office\s*manag",
        r"executive\s*assist", r"payroll", r"benefits", r"compensation",
        r"procurement", r"vendor\s*manag", r"real\s*estate",
        r"(?:it|tech)\s*(?:support|ops|admin)", r"enablement",
        r"analytics.*(?:data|business)", r"data\s*(?:analyst|ops|eng|scien)",
        r"business\s*intel",
    ], [
        r"people", r"finance", r"legal", r"workplace", r"talent", r"recruiting",
        r"operations",
    ]),
]


def categorize_by_keywords(title, team):
    """Categorize a job by matching title/team against keyword patterns."""
    title_lower = title.lower()
    team_lower = team.lower() if team else ""

    for domain, title_patterns, team_patterns in KEYWORD_RULES:
        # Check title patterns
        for pattern in title_patterns:
            if re.search(pattern, title_lower):
                return domain
        # Check team patterns (weaker signal, only if title didn't match)
        for pattern in team_patterns:
            if re.search(pattern, team_lower):
                return domain

    return "operations"  # Default fallback


def extract_skills_from_title(title, domain):
    """Extract rough skill tags from job title."""
    skills = []
    title_lower = title.lower()

    skill_patterns = {
        "python": r"python",
        "machine learning": r"machine.learn|(?:^|\b)ml\b",
        "deep learning": r"deep.learn",
        "LLM": r"\bllm\b|language.model",
        "distributed systems": r"distributed",
        "GPU": r"\bgpu\b",
        "infrastructure": r"infra",
        "kubernetes": r"kubernetes|k8s",
        "cloud": r"cloud|aws|gcp|azure",
        "networking": r"network",
        "security": r"secur",
        "data engineering": r"data\s*eng",
        "data analysis": r"data\s*analy|analytics",
        "frontend": r"frontend|front.end|react|typescript",
        "backend": r"backend|back.end",
        "full-stack": r"full.stack",
        "mobile": r"mobile|ios|android",
        "enterprise sales": r"account\s*exec|enterprise.*sales",
        "partnerships": r"partner",
        "recruiting": r"recruit|talent",
        "design": r"design",
        "UX": r"\bux\b",
        "program management": r"program\s*manag|\btpm\b",
        "research": r"research",
        "alignment": r"alignment|safety",
        "policy": r"policy|regulatory",
        "legal": r"legal|counsel",
        "finance": r"financ|accounting",
        "marketing": r"marketing|brand",
        "communications": r"communicat|\bpr\b",
    }

    for skill, pattern in skill_patterns.items():
        if re.search(pattern, title_lower):
            skills.append(skill)
            if len(skills) >= 3:
                break

    if not skills:
        # Fallback based on domain
        domain_default_skills = {
            "ai_research": ["research", "machine learning"],
            "ai_safety": ["alignment", "safety"],
            "infrastructure": ["systems", "infrastructure"],
            "product_engineering": ["software engineering"],
            "enterprise_deployment": ["customer engineering"],
            "sales_business": ["sales"],
            "security": ["security"],
            "policy_trust": ["policy"],
            "data_centers": ["hardware"],
            "operations": ["operations"],
            "marketing_comms": ["marketing"],
            "design": ["design"],
            "program_management": ["program management"],
        }
        skills = domain_default_skills.get(domain, ["general"])

    return skills[:3]


# ---------------------------------------------------------------------------
# LLM categorizer
# ---------------------------------------------------------------------------

def categorize_batch_llm(client, batch, model):
    """Send a batch of jobs to the LLM for categorization."""
    input_data = [
        {"id": i, "title": j["title"], "team": j.get("team", ""), "lab": j["lab"]}
        for i, j in enumerate(batch)
    ]

    response = client.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(input_data)},
            ],
            "temperature": 0.1,
        },
        timeout=120,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]

    # Strip markdown code fences if present
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

    return json.loads(content)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_raw_jobs():
    """Load all raw job listings from lab_jobs/*_raw.json."""
    all_jobs = []
    for fname in sorted(os.listdir(INPUT_DIR)):
        if not fname.endswith("_raw.json"):
            continue
        path = os.path.join(INPUT_DIR, fname)
        with open(path) as f:
            jobs = json.load(f)
        print(f"  Loaded {len(jobs)} jobs from {fname}")
        all_jobs.extend(jobs)
    return all_jobs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Categorize lab jobs")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--force", action="store_true", help="Re-categorize all")
    args = parser.parse_args()

    all_jobs = load_raw_jobs()
    if not all_jobs:
        print("No raw job files found in lab_jobs/. Run scrape_labs.py first.")
        return

    use_llm = os.environ.get("OPENROUTER_API_KEY")

    if use_llm:
        print(f"\nUsing LLM ({args.model}) to categorize {len(all_jobs)} jobs")
    else:
        print(f"\nNo OPENROUTER_API_KEY — using keyword categorizer for {len(all_jobs)} jobs")

    # Load existing (for incremental LLM mode)
    existing = {}
    if os.path.exists(OUTPUT_FILE) and not args.force:
        with open(OUTPUT_FILE) as f:
            for job in json.load(f):
                key = (job["title"], job.get("lab", ""), job.get("location", ""))
                existing[key] = job
        print(f"Already categorized: {len(existing)}")

    to_categorize = []
    done = []
    for job in all_jobs:
        key = (job["title"], job.get("lab", ""), job.get("location", ""))
        if key in existing and not args.force:
            done.append(existing[key])
        else:
            to_categorize.append(job)

    if not to_categorize:
        print("All jobs already categorized.")
        return

    print(f"Need to categorize: {len(to_categorize)}")

    if use_llm:
        # LLM path
        client = httpx.Client()
        errors = 0

        for batch_start in range(0, len(to_categorize), BATCH_SIZE):
            batch = to_categorize[batch_start:batch_start + BATCH_SIZE]
            batch_num = batch_start // BATCH_SIZE + 1
            total_batches = (len(to_categorize) + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"  Batch {batch_num}/{total_batches} ({len(batch)} jobs)...", end=" ", flush=True)

            try:
                results = categorize_batch_llm(client, batch, args.model)
                results_by_id = {r["id"]: r for r in results}
                for i, job in enumerate(batch):
                    result = results_by_id.get(i, {})
                    job["domain"] = result.get("domain", "operations")
                    job["skills"] = result.get("skills", [])
                    done.append(job)
                print("OK")
            except Exception as e:
                print(f"ERROR: {e}")
                errors += 1
                for job in batch:
                    job["domain"] = categorize_by_keywords(job["title"], job.get("team", ""))
                    job["skills"] = extract_skills_from_title(job["title"], job["domain"])
                    done.append(job)

            with open(OUTPUT_FILE, "w") as f:
                json.dump(done, f, indent=2)

            if batch_start + BATCH_SIZE < len(to_categorize):
                time.sleep(args.delay)

        client.close()
        print(f"\nDone. Categorized {len(done)} jobs, {errors} batch errors.")
    else:
        # Keyword path
        for job in to_categorize:
            job["domain"] = categorize_by_keywords(job["title"], job.get("team", ""))
            job["skills"] = extract_skills_from_title(job["title"], job["domain"])
            done.append(job)

        with open(OUTPUT_FILE, "w") as f:
            json.dump(done, f, indent=2)

        print(f"\nDone. Categorized {len(done)} jobs via keywords.")

    # Summary by domain
    domain_counts = {}
    for job in done:
        d = job.get("domain", "unknown")
        domain_counts[d] = domain_counts.get(d, 0) + 1

    print("\nDomain breakdown:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        pct = count / len(done) * 100
        print(f"  {domain:25s} {count:4d} ({pct:.1f}%)")

    # Summary by lab
    lab_counts = {}
    for job in done:
        lab = job.get("lab", "unknown")
        lab_counts[lab] = lab_counts.get(lab, 0) + 1

    print("\nBy lab:")
    for lab, count in sorted(lab_counts.items(), key=lambda x: -x[1]):
        print(f"  {lab}: {count}")


if __name__ == "__main__":
    main()
