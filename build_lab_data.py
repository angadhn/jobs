"""
Cross-reference AI lab hiring data with Karpathy's BLS occupation scores.

Reads lab_jobs.json (categorized lab postings), scores.json (BLS exposure),
and occupations.csv (employment stats). Produces:
  - site/lab_data.json  (compact JSON for frontend visualization)
  - lab_analysis.md     (markdown report with tables and insights)

Usage:
    uv run python build_lab_data.py
"""

import csv
import json
import os

# ---------------------------------------------------------------------------
# Domain → BLS occupation slug mapping (hand-curated)
# ---------------------------------------------------------------------------

DOMAIN_BLS_MAP = {
    "ai_research": [
        "computer-and-information-research-scientists",
        "mathematicians-and-statisticians",
        "data-scientists",
    ],
    "ai_safety": [
        "computer-and-information-research-scientists",
        "information-security-analysts",
    ],
    "infrastructure": [
        "computer-systems-administrators",
        "network-and-computer-systems-administrators",
        "software-developers",
        "database-administrators",
        "computer-network-architects",
    ],
    "product_engineering": [
        "software-developers",
        "web-developers-and-digital-designers",
        "software-quality-assurance-analysts-and-testers",
    ],
    "enterprise_deployment": [
        "sales-engineers",
        "computer-systems-analysts",
        "management-analysts",
    ],
    "sales_business": [
        "sales-managers",
        "advertising-sales-agents",
        "market-research-analysts",
    ],
    "security": [
        "information-security-analysts",
    ],
    "policy_trust": [
        "political-scientists",
        "lawyers",
        "public-relations-specialists",
    ],
    "data_centers": [
        "electrical-and-electronics-engineers",
        "industrial-engineers",
        "construction-managers",
    ],
    "operations": [
        "human-resources-specialists",
        "financial-analysts",
        "accountants-and-auditors",
        "lawyers",
    ],
    "marketing_comms": [
        "advertising-promotions-and-marketing-managers",
        "public-relations-specialists",
        "technical-writers",
    ],
    "design": [
        "web-developers-and-digital-designers",
        "graphic-designers",
        "industrial-designers",
    ],
    "program_management": [
        "project-management-specialists",
        "management-analysts",
        "computer-and-information-systems-managers",
    ],
}

DOMAIN_LABELS = {
    "ai_research": "AI Research",
    "ai_safety": "AI Safety",
    "infrastructure": "Infrastructure",
    "product_engineering": "Product Engineering",
    "enterprise_deployment": "Enterprise Deployment",
    "sales_business": "Sales & Business",
    "security": "Security",
    "policy_trust": "Policy & Trust",
    "data_centers": "Data Centers",
    "operations": "Operations",
    "marketing_comms": "Marketing & Comms",
    "design": "Design",
    "program_management": "Program Management",
}


def load_scores():
    """Load BLS AI exposure scores keyed by slug."""
    with open("scores.json") as f:
        return {s["slug"]: s for s in json.load(f)}


def load_occupations_csv():
    """Load BLS occupation stats keyed by slug."""
    with open("occupations.csv") as f:
        reader = csv.DictReader(f)
        return {row["slug"]: row for row in reader}


def load_lab_jobs():
    """Load categorized lab job listings."""
    with open("lab_jobs.json") as f:
        return json.load(f)


def safe_int(val):
    """Convert to int or return None."""
    if not val:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def main():
    lab_jobs = load_lab_jobs()
    scores = load_scores()
    bls_data = load_occupations_csv()

    total_jobs = len(lab_jobs)
    print(f"Loaded {total_jobs} categorized lab jobs")

    # Count jobs per domain and per lab
    domain_jobs = {}
    lab_domain_jobs = {}
    domain_example_titles = {}

    for job in lab_jobs:
        domain = job.get("domain", "unknown")
        lab = job.get("lab", "unknown")

        domain_jobs[domain] = domain_jobs.get(domain, 0) + 1

        if lab not in lab_domain_jobs:
            lab_domain_jobs[lab] = {}
        lab_domain_jobs[lab][domain] = lab_domain_jobs[lab].get(domain, 0) + 1

        if domain not in domain_example_titles:
            domain_example_titles[domain] = []
        if len(domain_example_titles[domain]) < 5:
            domain_example_titles[domain].append(job["title"])

    # Build domain-level analysis
    domains = []
    for domain, count in sorted(domain_jobs.items(), key=lambda x: -x[1]):
        if domain == "unknown":
            continue

        pct = count / total_jobs * 100
        bls_slugs = DOMAIN_BLS_MAP.get(domain, [])

        # Compute average BLS exposure for matched occupations
        matched_exposures = []
        matched_occupations = []
        total_bls_jobs = 0
        for slug in bls_slugs:
            score_entry = scores.get(slug)
            bls_entry = bls_data.get(slug)
            if score_entry:
                matched_exposures.append(score_entry["exposure"])
                bls_jobs = safe_int(bls_entry.get("num_jobs_2024")) if bls_entry else None
                total_bls_jobs += bls_jobs or 0
                matched_occupations.append({
                    "slug": slug,
                    "title": score_entry["title"],
                    "exposure": score_entry["exposure"],
                    "bls_jobs": bls_jobs,
                    "outlook_pct": safe_int(bls_entry.get("outlook_pct")) if bls_entry else None,
                    "median_pay": safe_int(bls_entry.get("median_pay_annual")) if bls_entry else None,
                })

        avg_exposure = sum(matched_exposures) / len(matched_exposures) if matched_exposures else None

        # Paradox score: high lab demand + high BLS exposure
        # Normalized: (domain_pct / max_pct) * (avg_exposure / 10)
        paradox = (pct / 100) * (avg_exposure / 10) * 100 if avg_exposure else 0

        # Per-lab breakdown
        lab_breakdown = {}
        for lab, ld in lab_domain_jobs.items():
            if domain in ld:
                lab_breakdown[lab] = ld[domain]

        domains.append({
            "domain": domain,
            "label": DOMAIN_LABELS.get(domain, domain),
            "lab_jobs": count,
            "lab_pct": round(pct, 1),
            "avg_bls_exposure": round(avg_exposure, 1) if avg_exposure else None,
            "bls_employment": total_bls_jobs,
            "matched_occupations": matched_occupations,
            "example_titles": domain_example_titles.get(domain, []),
            "paradox_score": round(paradox, 1),
            "by_lab": lab_breakdown,
        })

    # Sort by job count
    domains.sort(key=lambda d: -d["lab_jobs"])

    # Aggregate stats
    all_exposures = [d["avg_bls_exposure"] for d in domains if d["avg_bls_exposure"] is not None]
    weighted_exposure = sum(
        d["avg_bls_exposure"] * d["lab_jobs"]
        for d in domains if d["avg_bls_exposure"] is not None
    ) / sum(d["lab_jobs"] for d in domains if d["avg_bls_exposure"] is not None) if all_exposures else 0

    labs_list = sorted(lab_domain_jobs.keys())

    site_data = {
        "total_jobs": total_jobs,
        "labs": labs_list,
        "lab_counts": {lab: sum(ld.values()) for lab, ld in lab_domain_jobs.items()},
        "weighted_avg_exposure": round(weighted_exposure, 1),
        "domains": domains,
    }

    # Write site data
    os.makedirs("site", exist_ok=True)
    with open("site/lab_data.json", "w") as f:
        json.dump(site_data, f, indent=2)
    print(f"Wrote site/lab_data.json ({len(domains)} domains)")

    # ---------------------------------------------------------------------------
    # Generate markdown report
    # ---------------------------------------------------------------------------

    lines = [
        "# Frontier AI Lab Hiring Analysis",
        "",
        f"**{total_jobs} open roles** across {len(labs_list)} frontier AI labs",
        "(" + ", ".join(f"{lab}: {site_data['lab_counts'][lab]}" for lab in labs_list) + ")",
        "",
        f"**Weighted average BLS exposure of hired roles: {weighted_exposure:.1f}/10**",
        "",
        "## Domain Breakdown",
        "",
        "| Domain | Jobs | % | Avg BLS Exposure | BLS Employment | Paradox |",
        "|--------|------|---|-----------------|----------------|---------|",
    ]

    for d in domains:
        exp_str = f"{d['avg_bls_exposure']}" if d['avg_bls_exposure'] is not None else "—"
        bls_str = f"{d['bls_employment']:,}" if d['bls_employment'] else "—"
        lines.append(
            f"| {d['label']} | {d['lab_jobs']} | {d['lab_pct']}% | "
            f"{exp_str} | {bls_str} | {d['paradox_score']} |"
        )

    lines.extend([
        "",
        "## Key Insights",
        "",
    ])

    # Find high-paradox domains
    high_paradox = [d for d in domains if d["paradox_score"] > 2]
    if high_paradox:
        lines.append("### High-Paradox Domains")
        lines.append("These domains have high lab demand AND high BLS AI exposure scores — "
                     "the labs are hiring heavily for roles their own tech supposedly threatens:")
        lines.append("")
        for d in sorted(high_paradox, key=lambda x: -x["paradox_score"]):
            lines.append(
                f"- **{d['label']}** ({d['lab_jobs']} roles, {d['lab_pct']}%) — "
                f"BLS exposure {d['avg_bls_exposure']}/10"
            )
            if d["example_titles"]:
                lines.append(f"  - Examples: {', '.join(d['example_titles'][:3])}")
        lines.append("")

    # Top domains by absolute count
    lines.append("### Biggest Hiring Areas")
    lines.append("")
    for d in domains[:5]:
        lines.append(f"1. **{d['label']}** — {d['lab_jobs']} roles ({d['lab_pct']}%)")
    lines.append("")

    # Domain-to-BLS mapping details
    lines.append("## BLS Occupation Cross-Reference")
    lines.append("")
    for d in domains:
        if not d["matched_occupations"]:
            continue
        lines.append(f"### {d['label']}")
        lines.append(f"Lab jobs: {d['lab_jobs']} | Avg BLS exposure: {d['avg_bls_exposure']}")
        lines.append("")
        lines.append("| BLS Occupation | Exposure | Jobs (2024) | Outlook |")
        lines.append("|---------------|----------|-------------|---------|")
        for occ in d["matched_occupations"]:
            jobs_str = f"{occ['bls_jobs']:,}" if occ['bls_jobs'] else "—"
            outlook_str = f"{occ['outlook_pct']}%" if occ['outlook_pct'] is not None else "—"
            lines.append(f"| {occ['title']} | {occ['exposure']}/10 | {jobs_str} | {outlook_str} |")
        lines.append("")

    # Per-lab breakdown
    lines.append("## Per-Lab Breakdown")
    lines.append("")
    for lab in labs_list:
        lab_total = site_data["lab_counts"][lab]
        lines.append(f"### {lab.title()} ({lab_total} roles)")
        lines.append("")
        lab_domains = [(d["domain"], d["by_lab"].get(lab, 0)) for d in domains if d["by_lab"].get(lab, 0) > 0]
        lab_domains.sort(key=lambda x: -x[1])
        for domain_key, count in lab_domains:
            pct = count / lab_total * 100
            label = DOMAIN_LABELS.get(domain_key, domain_key)
            lines.append(f"- {label}: {count} ({pct:.0f}%)")
        lines.append("")

    report = "\n".join(lines)
    with open("lab_analysis.md", "w") as f:
        f.write(report)
    print(f"Wrote lab_analysis.md ({len(lines)} lines)")


if __name__ == "__main__":
    main()
