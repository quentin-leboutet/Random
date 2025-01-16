import os
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from typing import List, Dict
import re

###############################################################################
# Configuration
###############################################################################
NIPS_FOLDER = "nips"
START_YEAR = 2006
END_YEAR = 2024

# Ensure our output folder exists
OUTPUT_DIR = "statistics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Keyword synonyms (example: group LLM variations)
KEYWORD_SYNONYMS = {
    'LLM': [
        'llm', 'llms', 'large language model', 'large language models',
        'larger language model', 'larger language models'
    ],
    'Diffusion': [
        'diffusion', 'diffusion model', 'diffusion models', 'ddpm',
        'ddim'
    ],
    'RL': [
        'rl', 'reinforcement learning'
    ],
    'Transformer': [
        'transformer', 'transformers', 'transformer model', 'transformer models'
    ],
    'GNN': [
        'gnn', 'graph neural network', 'graph neural networks'
    ],
    # Add more as needed
}

# Big Tech synonyms (maps multiple affiliation names to one canonical Big Tech label)
BIGTECH_SYNONYMS = {
    "Google": [
        "google"
    ],
    "Intel": [
        "intel"
    ],
    "Meta": [
        "meta", "facebook"
    ],
    "Microsoft": [
        "microsoft"
    ],
    "Nvidia": [
        "nvidia"
    ],
    "Amazon": [
        "amazon"
    ],
    "Apple": [
        "apple"
    ],
    "Hugging Face": [
        "hugging"
    ],
    "Alibaba": [
        "alibaba"
    ],
    "Baidu": [
        "baidu"
    ],
    "ByteDance": [
        "byte"
    ],
    "Tencent": [
        "tencent"
    ]
}

###############################################################################
# Helper functions
###############################################################################
def load_all_papers(folder: str, start_year: int, end_year: int) -> List[Dict]:
    """
    Loads and concatenates all papers from JSON files named nips{YYYY}.json
    within [start_year, end_year].
    Adds 'year' to each paper dict for later analysis.
    """
    all_papers = []
    for year in range(start_year, end_year + 1):
        json_path = os.path.join(folder, f"nips{year}.json")
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                papers_year = json.load(f)
                for p in papers_year:
                    p['year'] = year
                all_papers.extend(papers_year)
    return all_papers

def plot_pie_with_legend(area_counts, title, out_file):
    fig, ax = plt.subplots(figsize=(10,10))
    wedges, _, autotexts = ax.pie(
        area_counts["count"],
        labels=[None]*len(area_counts),  # no labels on wedges
        autopct='%1.1f%%',
        startangle=140
    )

    # Increase size of the autopct text
    plt.setp(autotexts, size=12, weight="bold")

    # Create a legend using the wedge handles
    ax.legend(
        wedges,
        area_counts["primary_area"],
        title="Primary Area",
        loc="center left",
        bbox_to_anchor=(1, 0.5)
    )

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.show()

def normalize_keywords(keywords_str: str, synonyms_map: Dict[str, List[str]]) -> List[str]:
    """
    Splits the 'keywords' field by semicolons, lowercases them,
    and maps any synonyms to a canonical form (e.g. "large language model" -> "LLM").
    Returns a list of standardized keywords.
    """
    raw_keywords = [kw.strip().lower() for kw in keywords_str.split(';') if kw.strip()]

    # Reverse lookup dict for synonyms
    reverse_lookup = {}
    for canonical_kw, variants in synonyms_map.items():
        for variant in variants:
            reverse_lookup[variant.lower()] = canonical_kw

    standardized = []
    for kw in raw_keywords:
        if kw in reverse_lookup:
            standardized.append(reverse_lookup[kw])
        else:
            standardized.append(kw)
    return standardized

def find_bigtech_set(aff_list):
    """
    Given a list of author affiliations for a paper,
    returns a set of canonical Big Tech labels found,
    ensuring we do *not* match substrings inside other words
    (e.g., 'intel' inside 'intelligence').
    """
    bigtech_found = set()
    for aff in aff_list:
        aff_lower = aff.lower()
        for bigtech, synonyms in BIGTECH_SYNONYMS.items():
            for s in synonyms:
                # Build a regex pattern with word boundaries around the synonym
                # e.g. \bintel\b or \bintel labs\b
                pattern = r"\b" + re.escape(s.lower()) + r"\b"
                # If we find any match in the affiliation string, add that bigtech
                if re.search(pattern, aff_lower):
                    bigtech_found.add(bigtech)
                    # We can break if we only want to record one match per bigtech;
                    # but if synonyms differ, continuing might be okay. Usually, break is fine.
                    break
    return bigtech_found

def ignore_paper(paper: Dict) -> bool:
    """
    Returns True if we want to exclude this paper from all analyses:
     - if status is "Highlighted" or "Journal".
    """
    status_val = paper.get("status", "").lower().strip()
    if status_val in ("highlighted", "journal"):
        return True
    return False

def is_accepted(paper: Dict) -> bool:
    """
    Returns True if the paper is accepted (Poster, Oral, Spotlight, etc.)
    i.e. status != "reject", and not empty, and not "highlighted" or "journal".
    """
    status_val = paper.get("status", "").lower().strip()
    if status_val in ["", "reject", "highlighted", "journal"]:
        return False
    return True

###############################################################################
# Main
###############################################################################
def main():
    # 1) Load papers
    papers = load_all_papers(NIPS_FOLDER, START_YEAR, END_YEAR)
    print(f"Loaded {len(papers)} total items from {START_YEAR}-{END_YEAR} (including all statuses).")

    # 2) Build a DataFrame, ignoring highlight/journal
    records = []
    for p in papers:
        if ignore_paper(p):
            # skip highlighted / journal
            continue

        pid = p.get("id", "")
        year = p.get("year", None)
        status = p.get("status", "")
        track = p.get("track", "")
        primary_area = p.get("primary_area", "")
        keywords_str = p.get("keywords", "")
        aff_str = p.get("aff", "")

        # Normalize keywords
        std_keywords = normalize_keywords(keywords_str, KEYWORD_SYNONYMS)

        # Parse affiliations
        aff_list = [a.strip() for a in aff_str.split(';') if a.strip()]
        bigtech_found = find_bigtech_set(aff_list)
        bigtech_str = ",".join(sorted(list(bigtech_found)))

        records.append({
            "paper_id": pid,
            "year": year,
            "status": status.strip().title(),  # e.g. 'Reject', 'Poster', 'Oral', 'Spotlight', ...
            "track": track,
            "primary_area": primary_area,
            "keywords": std_keywords,
            "affiliations": aff_list,
            "bigtech_labels": bigtech_str
        })

    df = pd.DataFrame(records)
    df["accepted"] = df.apply(is_accepted, axis=1)

    print(f"DataFrame after ignoring (Highlighted/Journal): {df.shape[0]} papers total.")
    print(f"Accepted papers: {df[df['accepted']==True].shape[0]}")

    ###########################################################################
    # FIGURE 1: 
    # Only accepted papers. Plot the # of accepted, # of poster, # of oral, # of spotlight,
    # plus the ratio of each to total accepted (poster/accepted, oral/accepted, spotlight/accepted).
    ###########################################################################
    accepted_df = df[df["accepted"] == True].copy()
    # Group by year => accepted papers
    accepted_by_year = accepted_df.groupby("year")["paper_id"].count().rename("accepted")

    # # Poster vs # Oral vs # Spotlight
    status_counts = accepted_df.groupby(["year", "status"])["paper_id"].count().unstack(fill_value=0)

    # Ensure the relevant statuses exist, or fill them if absent
    for st in ["Poster", "Oral", "Spotlight"]:
        if st not in status_counts.columns:
            status_counts[st] = 0

    # Merge them
    yearly_data = pd.DataFrame({"accepted": accepted_by_year})
    yearly_data = yearly_data.join(status_counts[["Poster", "Oral", "Spotlight"]], how="left")
    yearly_data.fillna(0, inplace=True)

    # Ratios
    # Protect against division by zero in case some year is empty
    yearly_data["poster_ratio"] = yearly_data["Poster"] / yearly_data["accepted"].replace(0, np.nan)
    yearly_data["oral_ratio"] = yearly_data["Oral"] / yearly_data["accepted"].replace(0, np.nan)
    yearly_data["spotlight_ratio"] = yearly_data["Spotlight"] / yearly_data["accepted"].replace(0, np.nan)
    yearly_data.reset_index(inplace=True)  # so year is a column

    # Create the figure
    plt.figure(figsize=(10, 8))

    # Subplot 1: # accepted, # poster, # oral, # spotlight
    ax1 = plt.subplot(2,1,1)
    plt.plot(yearly_data["year"], yearly_data["accepted"], marker='o', label="Accepted")
    plt.plot(yearly_data["year"], yearly_data["Poster"], marker='o', label="Poster")
    plt.plot(yearly_data["year"], yearly_data["Oral"], marker='o', label="Oral")
    plt.plot(yearly_data["year"], yearly_data["Spotlight"], marker='o', label="Spotlight")
    plt.title("Accepted vs. Poster / Oral / Spotlight (count)")
    plt.ylabel("# Papers")
    plt.xticks(range(START_YEAR, END_YEAR + 1), rotation=45)
    plt.legend()
    plt.tight_layout()

    # Subplot 2: ratio
    plt.subplot(2,1,2, sharex=ax1)
    plt.plot(yearly_data["year"], yearly_data["poster_ratio"], marker='s', label="Poster Ratio")
    plt.plot(yearly_data["year"], yearly_data["oral_ratio"], marker='s', label="Oral Ratio")
    plt.plot(yearly_data["year"], yearly_data["spotlight_ratio"], marker='s', label="Spotlight Ratio")
    plt.xlabel("Year")
    plt.ylabel("Ratio (of accepted)")
    plt.xticks(range(START_YEAR, END_YEAR+1), rotation=45)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    out1 = os.path.join(OUTPUT_DIR, "01_accepted_poster_oral_spotlight.png")
    plt.savefig(out1)
    plt.show()
    print(f"Figure #1 saved to {out1}")

    ###########################################################################
    # FIGURE 2:
    # Big Tech: number of accepted papers per year, plus line for Non-BigTech
    #
    # We'll define a paper as BigTech if it has bigtech_labels != "".
    # Then we'll also produce lines for each bigtech individually.
    ###########################################################################
    # First, create a new column has_bigtech (True/False)
    accepted_df["has_bigtech"] = accepted_df["bigtech_labels"].apply(lambda x: len(x.strip()) > 0)

    # Summaries:
    # A) Single line for "Big Tech" vs "Non Big Tech"
    # B) lines for each big tech

    # A) Big Tech vs Non Big Tech
    summary_bt = accepted_df.groupby(["year", "has_bigtech"])["paper_id"].count().reset_index()
    # Pivot
    pivot_bt = summary_bt.pivot(index="year", columns="has_bigtech", values="paper_id").fillna(0)
    # Reindex from START_YEAR to END_YEAR so we see missing years
    pivot_bt = pivot_bt.reindex(range(START_YEAR, END_YEAR+1), fill_value=0)
    pivot_bt.columns = ["NonBigTech", "BigTech"]  # rename for clarity

    plt.figure(figsize=(10,6))
    plt.plot(pivot_bt.index, pivot_bt["BigTech"], marker='o', label="Big Tech")
    plt.plot(pivot_bt.index, pivot_bt["NonBigTech"], marker='o', label="Non-BigTech")
    plt.title("Number of Accepted Papers: Big Tech vs. Non Big Tech")
    plt.xlabel("Year")
    plt.ylabel("# Papers")
    plt.xticks(range(START_YEAR, END_YEAR+1), rotation=45)
    plt.legend()
    plt.tight_layout()
    out2a = os.path.join(OUTPUT_DIR, "02a_bigtech_vs_nonbigtech.png")
    plt.savefig(out2a)
    plt.show()
    print(f"Figure #2a saved to {out2a}")

    # B) Lines for each big tech
    # 1) Remove duplicates on "paper_id" if your data might have repeated rows
    accepted_unique = accepted_df.drop_duplicates(subset=["paper_id"]).copy()

    # 2) Expand each row into (year, paper_id, bigtech) but ensure bigtech is unique
    rows_tech = []
    for idx, row in accepted_unique.iterrows():
        # "bigtech_labels" might look like "Intel,Intel" or "Google,Meta" or ""
        techs = [t.strip() for t in row["bigtech_labels"].split(',') if t.strip()]
        techs_set = set(techs)  # ensure uniqueness
        for t in techs_set:
            rows_tech.append({
                "year": row["year"],
                "paper_id": row["paper_id"],
                "bigtech": t
            })

    bigtech_df = pd.DataFrame(rows_tech)

    # 3) Now group by (year, bigtech) counting unique paper_id
    bt_counts = bigtech_df.groupby(["year", "bigtech"])["paper_id"].nunique().reset_index(name="num_papers")

    # 4) Pivot and reindex to ensure all years are shown
    pivot_co = bt_counts.pivot(index="year", columns="bigtech", values="num_papers").fillna(0)
    pivot_co = pivot_co.reindex(range(START_YEAR, END_YEAR+1), fill_value=0)

    # 5) Plot
    plt.figure(figsize=(10,6))
    for col in pivot_co.columns:
        plt.plot(pivot_co.index, pivot_co[col], marker='o', label=col)
    plt.title("Number of Accepted Papers per Year by Big Tech (Counted Once/Paper)")
    plt.xlabel("Year")
    plt.ylabel("# Papers")
    plt.xticks(range(START_YEAR, END_YEAR+1), rotation=45)
    plt.legend()
    plt.tight_layout()

    out2b = os.path.join(OUTPUT_DIR, "bigtech_by_company_fixed.png")
    plt.savefig(out2b)
    plt.show()
    print(f"Figure #2b saved to {out2b}")

    ###########################################################################
    # FIGURE 3:
    # We had an error with "DataFrame index must be unique" => we fix it by dropping duplicates
    # before setting index. We'll only do this step if we want to join primary_area or other info.
    #
    # For example, if we want to do a pivot with Big Tech by year & primary_area => stacked bar,
    # we must ensure each paper ID is unique.
    ###########################################################################
    # Example: Big Tech evolution of primary_area. We'll do a stacked bar per year, for each Big Tech.
    # But user might have duplicates in 'paper_id'.
    # So let's first remove duplicates on "paper_id" from the accepted_df.
    accepted_unique = accepted_df.drop_duplicates(subset=["paper_id"]).copy()
    # Now we can do set_index safely
    accepted_map = accepted_unique.set_index("paper_id")[["primary_area"]].to_dict("index")

    # We'll create a new DataFrame (year, paper_id, bigtech, primary_area),
    # then do pivot for each bigtech's primary_area distribution by year.
    if not bigtech_df.empty:
        # Remove duplicates in accepted_df so we don't double-count same paper ID
        accepted_unique = accepted_df.drop_duplicates(subset=["paper_id"]).copy()
        accepted_map = accepted_unique.set_index("paper_id")[["primary_area"]].to_dict("index")

        # Attach primary_area if not already attached
        if "primary_area" not in bigtech_df.columns:
            bigtech_df["primary_area"] = bigtech_df["paper_id"].apply(
                lambda pid: accepted_map[pid]["primary_area"] if pid in accepted_map else "Unknown"
            )

        # Filter bigtech_df to year 2024 only
        bigtech_2024 = bigtech_df[bigtech_df["year"] == 2024].copy()
        if bigtech_2024.empty:
            print("No Big Tech data for year 2024 => skipping Figure 3 pie plots.")
        else:
            unique_techs = sorted(bigtech_2024["bigtech"].unique())
            for bt in unique_techs:
                subdf = bigtech_2024[bigtech_2024["bigtech"] == bt]
                if subdf.empty:
                    continue
                
                # Group by primary_area => count # papers
                area_counts = subdf.groupby("primary_area")["paper_id"].nunique().reset_index(name="count")
                if area_counts["count"].sum() == 0:
                    print(f"{bt}: no papers in 2024.")
                    continue
                
                # Build the pie plot
                out_file = os.path.join(OUTPUT_DIR, f"03_primary_areas_{bt}_2024_pie.png")
                plot_pie_with_legend(area_counts, f"{bt} - Primary Areas (2024)", out_file)
                print(f"[Figure 3] Big Tech {bt} (2024) pie chart saved to {out_file}")

    else:
        print("No bigtech_df data => skipping Figure 3.")

    ###########################################################################
    # FIGURE 4:
    # Top keywords among Spotlight/Oral papers by year
    ###########################################################################
    # Filter: year=2024, Spotlight/Oral
    spotoral_2024_df = accepted_df[
        (accepted_df["status"].str.lower().isin(["spotlight", "oral"])) &
        (accepted_df["year"] == 2024)
    ].copy()

    if spotoral_2024_df.empty:
        print("No Spotlight/Oral papers for year 2024 => skipping Figure 5.")
    else:
        # Flatten the keywords
        all_keywords = []
        for _, row in spotoral_2024_df.iterrows():
            all_keywords.extend(row["keywords"])

        if not all_keywords:
            print("No keywords found for year 2024 Spotlight/Oral => skipping Figure 5.")
        else:
            
            # Group by primary_area => count # papers
            area_counts = subdf.groupby("primary_area")["paper_id"].nunique().reset_index(name="count")
            
            # Count
            keyword_counter = Counter(all_keywords)
            # top-10
            top10 = keyword_counter.most_common(10)

            # If there are fewer than 10 distinct keywords, that's okay; we'll just plot however many exist
            labels, counts = zip(*top10)  # e.g. ["llm", "transformers", ...], [12, 9, ...]

            # Build the pie chart
            out5 = os.path.join(OUTPUT_DIR, "05_spotoral_top_keywords_2024_pie.png")
            plot_pie_with_legend(area_counts, "Top 10 Keywords (Spotlight/Oral) - Year 2024", out5)
            print(f"[Figure 5] Pie chart for top 10 keywords in 2024 (Spotlight/Oral) saved to {out5}")

    print("\nAll requested analyses complete! Check the 'statistics' folder for the PNG files.")

if __name__ == "__main__":
    main()