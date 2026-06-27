
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import hashlib
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import patches
from matplotlib.lines import Line2D

PROJECTS = {
    "PRJNA1037681": {
        "label": "Australia",
        "short": "AU",
        "color": "#4C78A8",
        "light": "#DDEBFA",
        "treecluster_threshold": 0.010,
        "treecluster_label": "0p010",
    },
    "PRJNA1130896": {
        "label": "United States",
        "short": "US",
        "color": "#F58518",
        "light": "#FCE5CA",
        "treecluster_threshold": 0.015,
        "treecluster_label": "0p015",
    },
}
PROJECT_ORDER = list(PROJECTS)
PROJECT_LABEL_TO_ID = {v["label"]: k for k, v in PROJECTS.items()}
COHORT_COLORS = {v["label"]: v["color"] for v in PROJECTS.values()}
BRANCH_COLORS = {
    "C4": "#6A51A3", "C6": "#1B9E77", "C18": "#D55E00", "C21": "#0072B2",
    "C1": "#E69F00", "C5": "#009E73", "C7": "#CC79A7", "Unclustered": "#BDBDBD",
}
G_START = 4652
G_END = 5617
HVR2_AA_START = 200
HVR2_AA_END = 322
HVR2_NT_START = G_START + (HVR2_AA_START - 1) * 3
HVR2_NT_END = G_START + HVR2_AA_END * 3 - 1
REFERENCE_TIP = "EPI_ISL_412866"
ISNV_AF_MIN = 0.03
ISNV_AF_MAX = 0.97
CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*", "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M", "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K", "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V", "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E", "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def repo_root() -> Path:
    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "config" / "analysis_config.yaml").exists():
            return candidate
    return Path(__file__).resolve().parents[1]


def step_dirs(step_slug: str, root: Path | None = None) -> tuple[Path, Path]:
    root = root or repo_root()
    data_dir = root / "data" / "processed_data" / step_slug
    fig_dir = root / "results" / "figures" / step_slug
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, fig_dir


def setup_style(base_font: float = 10.5) -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": base_font,
        "axes.labelsize": base_font + 1.0,
        "axes.titlesize": base_font + 1.7,
        "xtick.labelsize": base_font - 0.6,
        "ytick.labelsize": base_font - 0.6,
        "legend.fontsize": base_font - 0.8,
        "legend.title_fontsize": base_font - 0.2,
        "axes.linewidth": 0.95,
        "axes.edgecolor": "#333333",
        "xtick.major.width": 0.85,
        "ytick.major.width": 0.85,
        "xtick.major.size": 3.6,
        "ytick.major.size": 3.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })


def clean_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_label(ax, label: str, x: float = -0.10, y: float = 1.04, size: float = 14.0) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom", fontweight="bold", fontsize=size)


def save_pub_figure(fig, fig_dir: Path, stem: str, dpi: int = 450) -> dict[str, str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for ext in ["png", "pdf", "svg"]:
        path = fig_dir / f"{stem}.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")
        paths[ext] = str(path.relative_to(repo_root()))
    return paths


def write_table(df: pd.DataFrame, out_dir: Path, filename: str) -> Path:
    out = out_dir / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def read_csv(root: Path, rel: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(root / rel, **kwargs)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fasta_sequence_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(">"):
                count += 1
    return count


def fasta_lengths(path: Path) -> pd.DataFrame:
    records = []
    name = None
    chunks = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append({"record_id": name, "length_nt": len("".join(chunks))})
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if name is not None:
        records.append({"record_id": name, "length_nt": len("".join(chunks))})
    return pd.DataFrame(records)


def clade_sort_key(clade) -> tuple:
    text = str(clade)
    nums = [int(x) for x in re.findall(r"\d+", text)]
    return (text.split(".")[0], nums, text)


def branch_sort_key(branch) -> tuple:
    text = str(branch)
    if text == "Unclustered":
        return (10**8, text)
    match = re.search(r"(\d+)", text)
    return (int(match.group(1)) if match else 10**9, text)


def sorted_branches(values) -> list[str]:
    return sorted([str(v) for v in values if pd.notna(v)], key=branch_sort_key)


def lighten_color(color: str, amount: float = 0.65) -> str:
    rgb = np.array(mcolors.to_rgb(color))
    return mcolors.to_hex(1 - amount * (1 - rgb))


def parse_geo_stratum(row) -> str:
    loc = row.get("geographic_location")
    if pd.notna(loc) and str(loc).strip():
        text = str(loc).strip()
        if ":" in text:
            text = text.split(":", 1)[1].strip()
        if "," in text:
            text = text.split(",", 1)[0].strip()
        if text:
            return text
    for col in ["state", "city", "country"]:
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            return str(val).strip()
    return "unknown_geo"


def parse_sequence_name(sequence_name: str) -> dict[str, object]:
    match = re.match(r"^(PRJNA\d+)_(.+?)_(H\d+)_freq([0-9.eE+-]+)$", str(sequence_name))
    if not match:
        return {"project": np.nan, "SampleID": np.nan, "haplotype_id": np.nan, "haplotype_frequency": np.nan}
    project, sample, hap_id, freq = match.groups()
    return {"project": project, "SampleID": sample, "haplotype_id": hap_id, "haplotype_frequency": float(freq)}




def read_fasta_records(path: Path) -> pd.DataFrame:
    records = []
    name = None
    chunks = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    seq = "".join(chunks).upper()
                    records.append({"record_id": name, "sequence": seq, "length_nt": len(seq)})
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if name is not None:
        seq = "".join(chunks).upper()
        records.append({"record_id": name, "sequence": seq, "length_nt": len(seq)})
    return pd.DataFrame(records)


def input_haplotype_fasta_path(root: Path, project: str) -> Path:
    path = root / "data" / "input" / "haplotypes" / f"{project}_extracted_4652-5617.fasta"
    if path.exists():
        return path
    raise FileNotFoundError(f"No input haplotype FASTA found for {project}: {path}")


def input_consensus_fasta_path(root: Path, project: str) -> Path:
    candidates = [
        root / "data" / "input" / "consensus" / f"{project}_all_consensus_extracted_4652-5617.fasta",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No consensus FASTA found for {project}")


def input_global_fasta_path(root: Path) -> Path:
    candidates = [
        root / "data" / "input" / "global" / "nextstrain_2026_global_G_4652-5617_filtered.fasta",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No global G FASTA found")


def genome_base(seq: str, position: int) -> object:
    idx = int(position) - G_START
    seq = str(seq).upper()
    if idx < 0 or idx >= len(seq):
        return np.nan
    base = seq[idx]
    return base if base in {"A", "C", "G", "T"} else np.nan


def site_combo_from_sequence(seq: str, project: str, sites: list[dict[str, object]] | None = None) -> str:
    if sites is None:
        raise ValueError("site_combo_from_sequence requires sites from the LoFreq-linked site screen")
    bases = [genome_base(seq, site["position"]) for site in sites]
    return "-".join("N" if pd.isna(base) else str(base) for base in bases)


def aa_from_sequence(seq: str, aa_position: int) -> object:
    start = (int(aa_position) - 1) * 3
    codon = str(seq).upper()[start:start + 3]
    if len(codon) != 3 or any(base not in "ACGT" for base in codon):
        return np.nan
    return CODON_TABLE.get(codon, np.nan)


def combo_size(combo: str) -> int:
    combo = str(combo)
    return combo.count("+") + 1 if combo else 0


def combo_from_branches(labels) -> str:
    return "+".join(sorted(set(map(str, labels)), key=branch_sort_key))


def is_cross_branch(combo: str) -> bool:
    return "+" in str(combo)


def load_haplotype_records(root: Path) -> pd.DataFrame:
    rows = []
    for project, cfg in PROJECTS.items():
        fasta = input_haplotype_fasta_path(root, project)
        for rec in read_fasta_records(fasta).itertuples(index=False):
            if rec.record_id == REFERENCE_TIP:
                continue
            parsed = parse_sequence_name(rec.record_id)
            if pd.isna(parsed["SampleID"]):
                continue
            rows.append({
                "project": project,
                "project_label": cfg["label"],
                "project_short": cfg["short"],
                "SampleID": parsed["SampleID"],
                "SequenceName": rec.record_id,
                "haplotype_id": parsed["haplotype_id"],
                "hap_id": parsed["haplotype_id"],
                "haplotype_frequency": float(parsed["haplotype_frequency"]),
                "Frequency": float(parsed["haplotype_frequency"]),
                "G_seq": rec.sequence,
                "G_length": rec.length_nt,
                "input_fasta": str(fasta.relative_to(root)),
            })
    hap = pd.DataFrame(rows)
    if hap.empty:
        return hap
    return hap.sort_values(["project", "SampleID", "haplotype_id"]).reset_index(drop=True)


def build_haplotype_site_combinations(root: Path, samples: pd.DataFrame | None = None, site_sets: dict[str, list[dict[str, object]]] | None = None) -> pd.DataFrame:
    hap = load_haplotype_records(root)
    if samples is not None and not samples.empty:
        keep = samples[["project", "SampleID"]].drop_duplicates()
        hap = hap.merge(keep, on=["project", "SampleID"], how="inner")
    site_sets = site_sets or derive_linked_site_sets(root)
    rows = []
    for row in hap.itertuples(index=False):
        sites = site_sets[row.project]
        positions = [str(site["position"]) for site in sites]
        bases = [genome_base(row.G_seq, site["position"]) for site in sites]
        rows.append({
            "project": row.project,
            "SampleID": row.SampleID,
            "SequenceName": row.SequenceName,
            "haplotype_id": row.haplotype_id,
            "haplotype_frequency": row.haplotype_frequency,
            "site_positions": ";".join(positions),
            "site_combo": "-".join("N" if pd.isna(base) else str(base) for base in bases),
        })
    return pd.DataFrame(rows)


def haplotype_count_distribution_from_input(root: Path) -> pd.DataFrame:
    hap = load_haplotype_records(root)
    counts = hap.groupby(["project", "SampleID"], as_index=False).agg(n_haplotypes=("SequenceName", "count"))
    counts["haplotype_count_class"] = counts["n_haplotypes"].map(lambda x: "4+" if int(x) >= 4 else str(int(x)))
    dist = counts.groupby(["project", "haplotype_count_class"], as_index=False).agg(n_samples=("SampleID", "nunique"))
    dist["project_label"] = dist["project"].map(lambda p: PROJECTS[p]["label"])
    return dist


def treecluster_cache_dir(root: Path) -> Path:
    out = root / "data" / "processed_data" / "treecluster_cache"
    out.mkdir(parents=True, exist_ok=True)
    return out


def treecluster_generation_commands(root: Path) -> pd.DataFrame:
    rows = []
    cache = treecluster_cache_dir(root)
    for project, cfg in PROJECTS.items():
        fasta = input_haplotype_fasta_path(root, project)
        prefix = cache / f"{project}_extracted_4652-5617"
        treefile = Path(str(prefix) + ".treefile")
        tc_out = cache / f"{project}_extracted_4652-5617_treecluster_t{cfg['treecluster_label']}.txt"
        rows.append({
            "project": project,
            "project_label": cfg["label"],
            "step": "IQ-TREE",
            "command": f"iqtree -s {fasta.relative_to(root)} -o {REFERENCE_TIP} -bb 1000 -m MFP -T AUTO -pre {prefix.relative_to(root)}",
            "output": str(treefile.relative_to(root)),
        })
        rows.append({
            "project": project,
            "project_label": cfg["label"],
            "step": "TreeCluster",
            "command": f"TreeCluster.py -i {treefile.relative_to(root)} -t {cfg['treecluster_threshold']:.3f} -o {tc_out.relative_to(root)}",
            "output": str(tc_out.relative_to(root)),
        })
    return pd.DataFrame(rows)




def treecluster_manifest_path(root: Path, project: str) -> Path:
    return treecluster_cache_dir(root) / f"{project}_treecluster_cache_manifest.csv"


def expected_treecluster_cache_record(root: Path, project: str) -> dict[str, object]:
    cfg = PROJECTS[project]
    fasta = input_haplotype_fasta_path(root, project)
    cache = treecluster_cache_dir(root)
    prefix = cache / f"{project}_extracted_4652-5617"
    treefile = Path(str(prefix) + ".treefile")
    tc_out = cache / f"{project}_extracted_4652-5617_treecluster_t{cfg['treecluster_label']}.txt"
    iqtree_cmd = f"iqtree -s {fasta.relative_to(root)} -o {REFERENCE_TIP} -bb 1000 -m MFP -T AUTO -pre {prefix.relative_to(root)}"
    treecluster_cmd = f"TreeCluster.py -i {treefile.relative_to(root)} -t {cfg['treecluster_threshold']:.3f} -o {tc_out.relative_to(root)}"
    return {
        "project": project,
        "input_haplotype_fasta": str(fasta.relative_to(root)),
        "input_haplotype_fasta_sha256": sha256_file(fasta),
        "treecluster_threshold": cfg["treecluster_threshold"],
        "iqtree_command": iqtree_cmd,
        "treecluster_command": treecluster_cmd,
        "treefile": str(treefile.relative_to(root)),
        "treecluster_output": str(tc_out.relative_to(root)),
    }


def write_treecluster_cache_manifest(root: Path, project: str) -> None:
    record = expected_treecluster_cache_record(root, project)
    pd.DataFrame([record]).to_csv(treecluster_manifest_path(root, project), index=False)


def treecluster_cache_is_valid(root: Path, project: str, treefile: Path, tc_out: Path) -> bool:
    manifest_path = treecluster_manifest_path(root, project)
    if not (treefile.exists() and tc_out.exists() and manifest_path.exists()):
        return False
    expected = expected_treecluster_cache_record(root, project)
    observed = pd.read_csv(manifest_path)
    if observed.empty:
        return False
    row = observed.iloc[0].to_dict()
    for key in ["input_haplotype_fasta", "input_haplotype_fasta_sha256", "treecluster_threshold", "iqtree_command", "treecluster_command"]:
        if str(row.get(key)) != str(expected[key]):
            return False
    return True

def ensure_treecluster_output(root: Path, project: str, rebuild: bool | None = None) -> tuple[Path, Path, str]:
    cfg = PROJECTS[project]
    cache = treecluster_cache_dir(root)
    fasta = input_haplotype_fasta_path(root, project)
    prefix = cache / f"{project}_extracted_4652-5617"
    treefile = Path(str(prefix) + ".treefile")
    tc_out = cache / f"{project}_extracted_4652-5617_treecluster_t{cfg['treecluster_label']}.txt"
    if rebuild is None:
        rebuild = os.environ.get("RSVA_REBUILD_TREE", "0").lower() in {"1", "true", "yes"}
    cache_valid = treecluster_cache_is_valid(root, project, treefile, tc_out)
    if not rebuild and cache_valid:
        return treefile, tc_out, "cache_verified"

    tools = {
        "iqtree": shutil.which("iqtree") or shutil.which("iqtree2"),
        "TreeCluster.py": shutil.which("TreeCluster.py"),
    }
    if (rebuild or not cache_valid) and tools["iqtree"] and tools["TreeCluster.py"]:
        iqtree_cmd = [tools["iqtree"], "-s", str(fasta), "-o", REFERENCE_TIP, "-bb", "1000", "-m", "MFP", "-T", "AUTO", "-pre", str(prefix), "-redo"]
        with (cache / f"{project}_iqtree.log").open("w") as log:
            subprocess.run(iqtree_cmd, stdout=log, stderr=subprocess.STDOUT, text=True, check=True)
        tc_cmd = [tools["TreeCluster.py"], "-i", str(treefile), "-t", f"{cfg['treecluster_threshold']:.3f}", "-o", str(tc_out)]
        with (cache / f"{project}_treecluster.log").open("w") as log:
            subprocess.run(tc_cmd, stdout=log, stderr=subprocess.STDOUT, text=True, check=True)
        write_treecluster_cache_manifest(root, project)
        status = "cache_rebuilt" if rebuild else "cache_rebuilt_missing_or_invalid"
        return treefile, tc_out, status

    raise FileNotFoundError(
        f"TreeCluster output is missing or invalid for {project}, and IQ-TREE/TreeCluster were not both found. "
        "Install iqtree and TreeCluster.py, or run notebook 07 with RSVA_REBUILD_TREE=1 in an environment that has them, "
        "to rebuild branch assignments from the haplotype FASTA file."
    )


def treecluster_haplotype_assignments(root: Path, rebuild: bool | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    manifest_rows = []
    for project, cfg in PROJECTS.items():
        treefile, tc_path, status = ensure_treecluster_output(root, project, rebuild=rebuild)
        tc = pd.read_csv(tc_path, sep="\t")
        tc["project"] = project
        tc["project_label"] = cfg["label"]
        tc["TreeCluster_branch"] = tc["ClusterNumber"].map(lambda x: "Unclustered" if int(x) < 0 else f"C{int(x)}")
        parsed = pd.DataFrame([parse_sequence_name(x) for x in tc["SequenceName"]])
        tc["SampleID"] = parsed["SampleID"].to_numpy()
        tc["haplotype_id"] = parsed["haplotype_id"].to_numpy()
        tc["haplotype_frequency"] = parsed["haplotype_frequency"].to_numpy()
        frames.append(tc)
        manifest_rows.append({
            "project": project,
            "project_label": cfg["label"],
            "input_haplotype_fasta": str(input_haplotype_fasta_path(root, project).relative_to(root)),
            "treefile": str(treefile.relative_to(root)),
            "treecluster_output": str(tc_path.relative_to(root)),
            "treecluster_threshold": cfg["treecluster_threshold"],
            "status": status,
        })
    assignments = pd.concat(frames, ignore_index=True)
    manifest = pd.DataFrame(manifest_rows)
    return assignments, manifest


def sample_branch_combinations_from_assignments(root: Path, assignments: pd.DataFrame) -> pd.DataFrame:
    hap = load_haplotype_records(root)
    meta = load_metadata(root)
    valid = assignments.dropna(subset=["SampleID"]).copy()
    valid = valid[valid["TreeCluster_branch"].ne("Unclustered")].copy()
    valid = valid.merge(hap[["SequenceName", "G_seq", "G_length"]], on="SequenceName", how="left")
    rows = []
    for (project, sample), sub in valid.groupby(["project", "SampleID"], sort=False):
        branches = sorted_branches(sub["TreeCluster_branch"].unique())
        rows.append({
            "project": project,
            "project_label": PROJECTS[project]["label"],
            "SampleID": sample,
            "n_haplotypes": int(sub["SequenceName"].nunique()),
            "n_TreeCluster_branches": int(len(branches)),
            "TreeCluster_combo": "+".join(branches),
        })
    combos = pd.DataFrame(rows)
    meta_cols = ["project", "SampleID", "year_month", "geo_stratum", "clade"]
    combos = combos.merge(meta[meta_cols], on=["project", "SampleID"], how="left")
    combos["stratum"] = combos["year_month"].fillna("unknown_month") + "|" + combos["geo_stratum"].fillna("unknown_geo")
    combos["virus_clade"] = combos["clade"]
    return combos.sort_values(["project", "SampleID"]).reset_index(drop=True)


def get_sample_branch_combinations(root: Path, rebuild_tree: bool | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignments, manifest = treecluster_haplotype_assignments(root, rebuild=rebuild_tree)
    combos = sample_branch_combinations_from_assignments(root, assignments)
    return combos, assignments, manifest


def select_target_branch_combos(root: Path) -> pd.DataFrame:
    combos, _, _ = get_sample_branch_combinations(root)
    rows = []
    for project, cfg in PROJECTS.items():
        sub = combos[(combos["project"].eq(project)) & (combos["TreeCluster_combo"].astype(str).str.contains("+", regex=False, na=False))]
        counts = sub["TreeCluster_combo"].value_counts()
        if counts.empty:
            raise ValueError(f"No cross-branch TreeCluster combo found for {project}")
        target_combo = str(counts.index[0])
        rows.append({
            "project": project,
            "project_label": cfg["label"],
            "target_combo": target_combo,
            "target_branches": target_combo.split("+"),
            "n_target_samples": int(counts.iloc[0]),
            "selection_rule": "most frequent exact cross-branch TreeCluster combo",
        })
    return pd.DataFrame(rows)


def target_branch_combo(root: Path, project: str) -> str:
    targets = select_target_branch_combos(root)
    return str(targets.loc[targets["project"].eq(project), "target_combo"].iloc[0])


def target_branches_for_project(root: Path, project: str) -> list[str]:
    return target_branch_combo(root, project).split("+")


def target_combo_from_table(table: pd.DataFrame, project: str) -> str:
    sub = table[(table["project"].eq(project)) & (table["TreeCluster_combo"].astype(str).str.contains("+", regex=False, na=False))]
    counts = sub["TreeCluster_combo"].value_counts()
    if counts.empty:
        raise ValueError(f"No cross-branch combo in table for {project}")
    return str(counts.index[0])


def selected_linked_validation_samples(root: Path) -> pd.DataFrame:
    combos, _, _ = get_sample_branch_combinations(root)
    meta = load_metadata(root)
    targets = select_target_branch_combos(root)
    selected = []
    for row in targets.itertuples(index=False):
        project = row.project
        cfg = PROJECTS[project]
        sub = combos[(combos["project"].eq(project)) & (combos["TreeCluster_combo"].eq(row.target_combo))].copy()
        sub = sub.merge(meta[["project", "SampleID", "collection_date", "collection_dt", "year_month", "clade", "country", "state", "city", "geographic_location", "geo_stratum"]], on=["project", "SampleID"], how="left", suffixes=("", "_meta"))
        for col in ["year_month", "geo_stratum", "clade"]:
            meta_col = f"{col}_meta"
            if meta_col in sub.columns:
                sub[col] = sub[col].fillna(sub[meta_col])
        sub["selected_TreeCluster_combo"] = row.target_combo
        sub["validation_sample_set"] = f"{cfg['short']} linked iSNV validation set"
        sub["selection_rule"] = row.selection_rule
        selected.append(sub)
    out = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    out = out.sort_values(["project", "collection_dt", "SampleID"]).reset_index(drop=True)
    out["global_sample_index"] = out.groupby("project").cumcount() + 1
    out["global_short_label"] = out.apply(lambda r: f"{PROJECTS[r['project']]['short']}-S{int(r['global_sample_index']):02d}", axis=1)
    return out


def normalized_lofreq_raw(root: Path) -> pd.DataFrame:
    raw = read_csv(root, "data/input/isnv/lofreq_raw_calls.csv")
    raw = raw.rename(columns={"sample": "SampleID", "ref": "Reference", "alt": "Allele", "dp": "DP", "af": "AF"})
    raw["Reference"] = raw["Reference"].astype(str).str.upper()
    raw["Allele"] = raw["Allele"].astype(str).str.upper()
    raw["position"] = pd.to_numeric(raw["position"], errors="coerce").astype("Int64")
    raw["AF"] = pd.to_numeric(raw["AF"], errors="coerce")
    raw["DP"] = pd.to_numeric(raw.get("DP", np.nan), errors="coerce")
    raw["is_indel"] = raw.get("is_indel", False).astype(str).str.lower().isin(["true", "1", "yes"])
    return raw


def derive_linked_site_sets(root: Path, min_sample_fraction: float = 0.85) -> dict[str, list[dict[str, object]]]:
    selected = selected_linked_validation_samples(root)
    raw = normalized_lofreq_raw(root)
    rows = []
    site_sets: dict[str, list[dict[str, object]]] = {}
    for project in PROJECT_ORDER:
        samples = selected.loc[selected["project"].eq(project), "SampleID"].astype(str).unique()
        min_samples = int(math.ceil(len(samples) * min_sample_fraction))
        sub = raw[
            raw["project"].eq(project)
            & raw["SampleID"].astype(str).isin(samples)
            & raw["position"].between(G_START, G_END)
            & (~raw["is_indel"])
            & raw["AF"].between(ISNV_AF_MIN, ISNV_AF_MAX, inclusive="left")
        ].copy()
        rec = (
            sub.groupby(["position", "Reference", "Allele"], as_index=False)
            .agg(
                samples=("SampleID", "nunique"),
                median_AF=("AF", "median"),
                min_AF=("AF", "min"),
                max_AF=("AF", "max"),
                median_DP=("DP", "median"),
            )
            .sort_values(["samples", "position"], ascending=[False, True])
        )
        keep = rec[rec["samples"] >= min_samples].sort_values("position").copy()
        if keep.empty:
            raise ValueError(f"No linked-site candidates passed recurrence threshold for {project}")
        site_sets[project] = []
        for r in keep.itertuples(index=False):
            site = {"position": int(r.position), "ref": str(r.Reference), "alt": str(r.Allele)}
            site_sets[project].append(site)
            rows.append({
                "project": project,
                "project_label": PROJECTS[project]["label"],
                "position": int(r.position),
                "Reference": str(r.Reference),
                "Allele": str(r.Allele),
                "samples_with_candidate_iSNV": int(r.samples),
                "n_validation_samples": int(len(samples)),
                "min_samples_required": int(min_samples),
                "recurrence_fraction": float(r.samples / len(samples)) if len(samples) else np.nan,
                "median_AF": float(r.median_AF),
                "min_AF": float(r.min_AF),
                "max_AF": float(r.max_AF),
                "median_DP": float(r.median_DP),
                "screen_rule": f"G SNV; {ISNV_AF_MIN:.2f}<=AF<{ISNV_AF_MAX:.2f}; recurrent in >= {min_sample_fraction:.0%} of validation samples",
            })
    derive_linked_site_sets.last_screen_table = pd.DataFrame(rows)
    return site_sets


def g_protein_domain_layout() -> pd.DataFrame:
    return pd.DataFrame([
        {"domain": "CT", "start": 1, "end": 37, "color": "#BDBDBD"},
        {"domain": "TM", "start": 38, "end": 66, "color": "#6F6F6F"},
        {"domain": "HVR1", "start": 67, "end": 163, "color": "#86BCEB"},
        {"domain": "CCD", "start": 164, "end": 199, "color": "#A9D7A5"},
        {"domain": "HVR2", "start": 200, "end": 322, "color": "#F4A38D"},
    ])

def input_manifest(root: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = [
        ("metadata", "data/metadata/meta_v6_with_season_clade.csv"),
        ("depth_qc", "data/metadata/qualified_samples_depth200_pos90.csv"),
        ("reference", "data/reference/reference.fasta"),
        ("annotation", "data/reference/genome_annotation.gff"),
        ("G_reference_alignment", "data/reference/EPI_ISL_412866_G_4652-5617_reference_alignment.fasta"),
        ("LoFreq_raw_calls", "data/input/isnv/lofreq_raw_calls.csv"),
        ("global_G_consensus_FASTA", "data/input/global/nextstrain_2026_global_G_4652-5617_filtered.fasta"),
    ]
    for project, cfg in PROJECTS.items():
        required.extend([
            (f"{project}_haplotype_G_FASTA", f"data/input/haplotypes/{project}_extracted_4652-5617.fasta"),
            (f"{project}_study_consensus_G_FASTA", f"data/input/consensus/{project}_all_consensus_extracted_4652-5617.fasta"),
        ])
    rows = []
    for role, rel in required:
        path = root / rel
        record = {"role": role, "path": rel, "exists": path.exists(), "bytes": np.nan, "sha256": np.nan, "rows": np.nan, "columns": np.nan, "fasta_sequences": np.nan}
        if path.exists() and path.is_file():
            record["bytes"] = path.stat().st_size
            record["sha256"] = sha256_file(path)
            suffix = path.suffix.lower()
            if suffix == ".csv":
                try:
                    df = pd.read_csv(path, nrows=5)
                    record["columns"] = len(df.columns)
                    record["rows"] = sum(1 for _ in path.open("r", encoding="utf-8", errors="replace")) - 1
                except Exception:
                    pass
            if suffix in {".fa", ".fasta", ".fna"}:
                record["fasta_sequences"] = fasta_sequence_count(path)
        rows.append(record)
    manifest = pd.DataFrame(rows)
    summary = manifest.groupby("exists", as_index=False).agg(files=("path", "count"), total_bytes=("bytes", "sum"))
    write_table(manifest, out_dir, "input_file_manifest.csv")
    write_table(summary, out_dir, "input_file_manifest_summary.csv")
    return manifest, summary

def load_metadata(root: Path) -> pd.DataFrame:
    meta = read_csv(root, "data/metadata/meta_v6_with_season_clade.csv")
    meta = meta.rename(columns={"project_id": "project", "sra_id": "SampleID"})
    meta = meta[meta["project"].isin(PROJECT_ORDER)].copy()
    meta["collection_dt"] = pd.to_datetime(meta["collection_date"], errors="coerce")
    meta["year_month"] = meta["collection_dt"].dt.strftime("%Y-%m")
    meta["geo_stratum"] = meta.apply(parse_geo_stratum, axis=1)
    meta["project_label"] = meta["project"].map(lambda p: PROJECTS[p]["label"])
    meta["project_short"] = meta["project"].map(lambda p: PROJECTS[p]["short"])
    return meta


def load_depth_qc(root: Path) -> pd.DataFrame:
    qc = read_csv(root, "data/metadata/qualified_samples_depth200_pos90.csv")
    qc = qc.rename(columns={"sample": "SampleID"})
    qc = qc[qc["project"].isin(PROJECT_ORDER)].copy()
    return qc


def build_retained_sample_metadata(root: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta = load_metadata(root)
    qc = load_depth_qc(root)
    hap = load_haplotype_records(root)
    retained = hap.groupby(["project", "SampleID"], as_index=False).agg(n_haplotypes=("SequenceName", "count"))

    meta_cols = ["project", "SampleID", "project_label", "project_short", "clade", "collection_date", "collection_dt", "year_month", "country", "state", "city", "geographic_location", "geo_stratum"]
    sample_meta = retained.merge(meta[meta_cols], on=["project", "SampleID"], how="left")
    sample_meta = sample_meta.merge(qc[["project", "SampleID", "qualified_ratio", "mean_depth", "median_depth"]], on=["project", "SampleID"], how="left")
    sample_meta["sort_date"] = sample_meta["collection_dt"].fillna(pd.Timestamp("2099-01-01"))
    sample_meta = sample_meta.sort_values(["project", "sort_date", "SampleID"]).reset_index(drop=True)
    sample_meta["global_sample_index"] = sample_meta.groupby("project").cumcount() + 1
    sample_meta["global_short_label"] = sample_meta.apply(lambda r: f"{PROJECTS[r['project']]['short']}-S{int(r['global_sample_index']):02d}", axis=1)
    sample_meta["stratum"] = sample_meta["year_month"].fillna("unknown_month") + "|" + sample_meta["geo_stratum"].fillna("unknown_geo")
    sample_meta["project_label"] = sample_meta["project"].map(lambda p: PROJECTS[p]["label"])
    sample_meta["project_short"] = sample_meta["project"].map(lambda p: PROJECTS[p]["short"])

    summary_rows = []
    for project in PROJECT_ORDER:
        sub_meta = meta[meta["project"].eq(project)]
        sub_qc = qc[qc["project"].eq(project)]
        sub_ret = sample_meta[sample_meta["project"].eq(project)]
        summary_rows.append({
            "project": project,
            "project_label": PROJECTS[project]["label"],
            "metadata_samples": int(sub_meta["SampleID"].nunique()),
            "depth_qc_samples": int(sub_qc["SampleID"].nunique()),
            "retained_haplotype_samples": int(sub_ret["SampleID"].nunique()),
            "multi_haplotype_samples": int((sub_ret["n_haplotypes"] > 1).sum()),
            "first_collection_month": sub_ret["year_month"].dropna().min(),
            "last_collection_month": sub_ret["year_month"].dropna().max(),
            "n_clades_retained": int(sub_ret["clade"].dropna().nunique()),
        })
    summary = pd.DataFrame(summary_rows)
    month_clade = (
        sample_meta.dropna(subset=["year_month", "clade"])
        .groupby(["project", "project_label", "year_month", "clade"], as_index=False)["SampleID"]
        .nunique()
        .rename(columns={"SampleID": "n_samples"})
        .sort_values(["project", "year_month", "clade"])
    )
    write_table(sample_meta, out_dir, "current_T10_sample_metadata.csv")
    write_table(summary, out_dir, "sample_inclusion_summary.csv")
    write_table(month_clade, out_dir, "sample_month_clade_counts.csv")
    return sample_meta, summary, month_clade

def draw_sample_composition(month_clade: pd.DataFrame, fig_dir: Path) -> tuple[plt.Figure, dict[str, str]]:
    setup_style(10.4)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"wspace": 0.40})
    all_clades = sorted(month_clade["clade"].dropna().unique(), key=clade_sort_key)
    palette = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2", "#B07AA1", "#EDC948", "#9C755F", "#BAB0AC", "#86BCB6", "#8CD17D", "#FFBE7D", "#D37295"]
    clade_colors = {clade: palette[i % len(palette)] for i, clade in enumerate(all_clades)}
    for ax, project, panel in zip(axes, PROJECT_ORDER, ["A", "B"]):
        sub = month_clade[month_clade["project"].eq(project)].copy()
        months = sorted(sub["year_month"].dropna().unique())
        clades = sorted(sub["clade"].dropna().unique(), key=clade_sort_key)
        x = np.arange(len(months))
        bottom = np.zeros(len(months), dtype=float)
        for clade in clades:
            vals = sub[sub["clade"].eq(clade)].set_index("year_month")["n_samples"].reindex(months, fill_value=0).to_numpy(float)
            ax.bar(x, vals, bottom=bottom, width=0.72, color=clade_colors[clade], edgecolor="white", linewidth=0.55, label=clade)
            for xi, val, start in zip(x, vals, bottom):
                if val >= 2:
                    ax.text(xi, start + val / 2, str(int(val)), ha="center", va="center", fontsize=7.6, color="white", fontweight="bold")
            bottom += vals
        ax.set_title(PROJECTS[project]["label"], loc="left", pad=7)
        ax.set_xticks(x)
        ax.set_xticklabels(months, rotation=45, ha="right")
        ax.set_xlabel("Collection month")
        ax.set_ylabel("Samples")
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.65)
        ax.set_axisbelow(True)
        clean_axis(ax)
        panel_label(ax, panel, x=-0.14)
        ax.legend(title="Clade", frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1 if len(clades) <= 6 else 2, handlelength=1.0, columnspacing=0.9)
    paths = save_pub_figure(fig, fig_dir, "Fig1AB_sample_month_clade_composition")
    return fig, paths



def fasta_qc_by_project(root: Path) -> pd.DataFrame:
    rows = []
    for project, cfg in PROJECTS.items():
        for role, path in [
            ("study_consensus_G_FASTA", input_consensus_fasta_path(root, project)),
            ("haplotype_G_FASTA", input_haplotype_fasta_path(root, project)),
        ]:
            recs = read_fasta_records(path)
            if role == "study_consensus_G_FASTA":
                recs = recs[recs["record_id"].ne(REFERENCE_TIP)].copy()
                sample_count = recs["record_id"].nunique()
            else:
                parsed = pd.DataFrame([parse_sequence_name(x) for x in recs["record_id"]])
                recs = pd.concat([recs.reset_index(drop=True), parsed], axis=1)
                recs = recs.dropna(subset=["SampleID"]).copy()
                sample_count = recs["SampleID"].nunique()
            recs["ambiguous_fraction"] = recs["sequence"].map(lambda s: sum(base not in "ACGT" for base in str(s).upper()) / len(str(s)) if len(str(s)) else np.nan)
            rows.append({
                "project": project,
                "project_label": cfg["label"],
                "input_role": role,
                "input_path": str(path.relative_to(root)),
                "records": int(len(recs)),
                "samples": int(sample_count),
                "min_length_nt": int(recs["length_nt"].min()) if len(recs) else np.nan,
                "median_length_nt": float(recs["length_nt"].median()) if len(recs) else np.nan,
                "max_length_nt": int(recs["length_nt"].max()) if len(recs) else np.nan,
                "median_ambiguous_fraction": float(recs["ambiguous_fraction"].median()) if len(recs) else np.nan,
                "max_ambiguous_fraction": float(recs["ambiguous_fraction"].max()) if len(recs) else np.nan,
            })
    return pd.DataFrame(rows)


def reference_coordinate_summary(root: Path) -> pd.DataFrame:
    gff = pd.read_csv(root / "data/reference/genome_annotation.gff", sep="\t", comment="#", header=None, names=["seqid", "source", "type", "start", "end", "score", "strand", "phase", "attributes"])
    rows = []
    rows.append({"feature": "G_extracted_window", "start": G_START, "end": G_END, "length_nt": G_END - G_START + 1, "note": "G segment used for haplotype and consensus comparisons"})
    rows.append({"feature": "HVR2_window", "start": HVR2_NT_START, "end": HVR2_NT_END, "length_nt": HVR2_NT_END - HVR2_NT_START + 1, "note": "AA 200-322 mapped onto genome coordinates"})
    for row in gff.itertuples(index=False):
        if int(row.start) <= G_END and int(row.end) >= G_START:
            feature_type = "reference_record" if str(row.type) == "source" else str(row.type)
            rows.append({"feature": f"GFF_{feature_type}", "start": int(row.start), "end": int(row.end), "length_nt": int(row.end) - int(row.start) + 1, "note": str(row.attributes)[:120]})
    return pd.DataFrame(rows)

def read_preprocessing_alignment_summary(root: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    qc = load_depth_qc(root)
    fasta_summary = fasta_lengths(root / "data/reference/reference.fasta")
    g_fasta_summary = fasta_lengths(root / "data/reference/EPI_ISL_412866_G_4652-5617_reference_alignment.fasta")
    gff = pd.read_csv(root / "data/reference/genome_annotation.gff", sep="	", comment="#", header=None, names=["seqid", "source", "type", "start", "end", "score", "strand", "phase", "attributes"])
    gff_for_summary = gff.copy()
    gff_for_summary["type"] = gff_for_summary["type"].replace({"source": "reference_record"})
    feature_summary = gff_for_summary.groupby("type", as_index=False).agg(n_features=("type", "size"), min_start=("start", "min"), max_end=("end", "max"))
    coordinate_summary = reference_coordinate_summary(root)
    sequence_qc = fasta_qc_by_project(root)
    depth_summary = qc.groupby("project", as_index=False).agg(
        project_label=("project", lambda s: PROJECTS[s.iloc[0]]["label"]),
        qualified_samples=("SampleID", "nunique"),
        median_mean_depth=("mean_depth", "median"),
        median_median_depth=("median_depth", "median"),
        median_qualified_ratio=("qualified_ratio", "median"),
    )
    sample_meta, _, _ = build_retained_sample_metadata(root, out_dir)
    depth_hap = sample_meta[["project", "project_label", "SampleID", "n_haplotypes", "mean_depth", "median_depth", "qualified_ratio"]].copy()
    depth_hap["haplotype_count_class"] = depth_hap["n_haplotypes"].map(lambda x: "4+" if x >= 4 else str(int(x)))
    write_table(depth_summary, out_dir, "depth_qc_summary_by_project.csv")
    write_table(feature_summary, out_dir, "reference_annotation_feature_summary.csv")
    write_table(coordinate_summary, out_dir, "g_gene_coordinate_summary.csv")
    write_table(sequence_qc, out_dir, "g_sequence_qc_by_project.csv")
    write_table(fasta_summary.assign(reference_file="whole_genome_reference"), out_dir, "reference_fasta_summary.csv")
    write_table(g_fasta_summary.assign(reference_file="G_segment_reference_alignment"), out_dir, "g_reference_alignment_summary.csv")
    write_table(depth_hap, out_dir, "retained_depth_haplotype_table.csv")
    return depth_summary, feature_summary, coordinate_summary, sequence_qc, depth_hap

def draw_depth_haplotype(depth_hap: pd.DataFrame, fig_dir: Path) -> tuple[plt.Figure, dict[str, str]]:
    setup_style(10.5)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4), gridspec_kw={"width_ratios": [0.95, 1.30], "wspace": 0.34})
    ax = axes[0]
    for idx, project in enumerate(PROJECT_ORDER):
        label = PROJECTS[project]["label"]
        vals = depth_hap.loc[depth_hap["project"].eq(project), "median_depth"].dropna().to_numpy()
        parts = ax.violinplot(vals, positions=[idx], widths=0.56, showmeans=False, showmedians=False, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(lighten_color(PROJECTS[project]["color"], 0.48))
            body.set_edgecolor(PROJECTS[project]["color"])
            body.set_alpha(0.72)
        rng = np.random.default_rng(42 + idx)
        ax.scatter(np.full(len(vals), idx) + rng.normal(0, 0.045, len(vals)), vals, s=22, color=PROJECTS[project]["color"], alpha=0.75, edgecolor="white", linewidth=0.30)
        ax.boxplot(vals, positions=[idx], widths=0.22, patch_artist=True, showfliers=False, medianprops={"color": "#202020", "lw": 1.15}, boxprops={"facecolor": "white", "edgecolor": "#303030", "lw": 0.9}, whiskerprops={"color": "#303030", "lw": 0.9}, capprops={"color": "#303030", "lw": 0.9})
    ax.set_xticks([0, 1])
    ax.set_xticklabels([PROJECTS[p]["label"] for p in PROJECT_ORDER])
    ax.set_ylabel("Median depth across the G segment")
    ax.set_title("Sequencing depth", loc="left", pad=7)
    panel_label(ax, "A", x=-0.16)
    clean_axis(ax)
    ax = axes[1]
    handles = []
    for project in PROJECT_ORDER:
        label = PROJECTS[project]["label"]
        sub = depth_hap[depth_hap["project"].eq(project)].copy()
        rng = np.random.default_rng(100 if project == "PRJNA1037681" else 101)
        y = sub["n_haplotypes"].astype(float) + rng.normal(0, 0.035, len(sub))
        ax.scatter(sub["median_depth"], y, s=38, color=PROJECTS[project]["color"], alpha=0.82, edgecolor="white", linewidth=0.35, label=label)
        handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor=PROJECTS[project]["color"], markeredgecolor="white", markersize=7.2, label=label))
    ax.set_xscale("log")
    ax.set_yticks(sorted(depth_hap["n_haplotypes"].dropna().unique()))
    ax.set_xlabel("Median depth across the G segment")
    ax.set_ylabel("Reconstructed G-gene haplotypes per sample")
    ax.set_title("Depth and haplotype number", loc="left", pad=7)
    panel_label(ax, "B", x=-0.13)
    clean_axis(ax)
    ax.grid(axis="y", color="#EAEAEA", lw=0.55)
    ax.set_axisbelow(True)
    fig.legend(handles=handles, frameon=False, loc="center left", bbox_to_anchor=(0.86, 0.58), title="Cohort")
    fig.tight_layout(rect=[0, 0, 0.84, 1])
    paths = save_pub_figure(fig, fig_dir, "FigS_depth_and_haplotype_count")
    return fig, paths


def consensus_clade_annotation(root: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_meta, _, _ = build_retained_sample_metadata(root, out_dir)
    clade_counts = (
        sample_meta.groupby(["project", "project_label", "clade"], as_index=False)["SampleID"]
        .nunique()
        .rename(columns={"SampleID": "n_samples"})
        .sort_values(["project", "n_samples", "clade"], ascending=[True, False, True])
    )
    fasta_rows = []
    for project, cfg in PROJECTS.items():
        for path, role in [
            (input_consensus_fasta_path(root, project), "study_consensus_G_FASTA"),
            (input_haplotype_fasta_path(root, project), "haplotype_G_FASTA"),
        ]:
            lengths = fasta_lengths(path)
            fasta_rows.append({
                "project": project,
                "project_label": cfg["label"],
                "role": role,
                "path": str(path.relative_to(root)),
                "exists": path.exists(),
                "records": int(len(lengths)),
                "min_length_nt": int(lengths["length_nt"].min()) if not lengths.empty else np.nan,
                "max_length_nt": int(lengths["length_nt"].max()) if not lengths.empty else np.nan,
                "bytes": path.stat().st_size if path.exists() else np.nan,
            })
    tree_manifest = pd.DataFrame(fasta_rows)
    write_table(clade_counts, out_dir, "retained_sample_clade_counts.csv")
    write_table(tree_manifest, out_dir, "consensus_and_g_sequence_file_manifest.csv")
    return clade_counts, tree_manifest

def intrahost_variant_calling(root: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calls = read_csv(root, "data/input/isnv/lofreq_raw_calls.csv")
    sample_meta, _, _ = build_retained_sample_metadata(root, out_dir)
    retained = sample_meta[["project", "SampleID", "project_label", "global_short_label"]].drop_duplicates()
    calls = calls.rename(columns={"sample": "SampleID", "ref": "Reference", "alt": "Allele", "dp": "Coverage", "af": "AF"})
    calls = calls.merge(retained, on=["project", "SampleID"], how="inner")
    calls["position"] = pd.to_numeric(calls["position"], errors="coerce")
    calls["AF"] = pd.to_numeric(calls["AF"], errors="coerce")
    calls["Coverage"] = pd.to_numeric(calls["Coverage"], errors="coerce")
    calls["is_G"] = calls["position"].between(G_START, G_END)
    calls["is_HVR2"] = calls["position"].between(HVR2_NT_START, HVR2_NT_END)
    calls["aa_position"] = np.floor((calls["position"] - G_START) / 3 + 1).astype("Int64")
    filt = calls[
        calls["is_G"]
        & (~calls["is_indel"].fillna(False).astype(bool))
        & calls["AF"].between(ISNV_AF_MIN, ISNV_AF_MAX, inclusive="left")
    ].copy()
    filt["Frequency"] = filt["AF"] * 100.0
    filt["Count"] = np.rint(filt["AF"] * filt["Coverage"]).astype("Int64")
    filt["Type"] = "SNV"
    filt["mutation_right_name"] = filt["Reference"].astype(str) + "_" + filt["position"].astype(int).astype(str) + "_" + filt["Allele"].astype(str)
    filt["variant_class"] = "SNV"
    keep = ["project", "project_label", "SampleID", "global_short_label", "position", "Type", "Reference", "Allele", "Count", "Coverage", "AF", "Frequency", "mutation_right_name", "is_G", "is_HVR2", "aa_position", "variant_class", "filter", "qual", "sb"]
    filt = filt[keep].sort_values(["project", "SampleID", "position", "Allele"]).reset_index(drop=True)
    counts = (
        sample_meta[["project", "project_label", "SampleID", "global_short_label"]]
        .merge(filt.groupby(["project", "SampleID"]).size().rename("candidate_G_iSNVs").reset_index(), on=["project", "SampleID"], how="left")
    )
    counts["candidate_G_iSNVs"] = counts["candidate_G_iSNVs"].fillna(0).astype(int)
    recurrence = (
        filt.groupby(["project", "project_label", "position", "aa_position", "Reference", "Allele", "is_HVR2"], as_index=False)
        .agg(samples_with_candidate_iSNV=("SampleID", "nunique"), median_frequency_percent=("Frequency", "median"), min_frequency_percent=("Frequency", "min"), max_frequency_percent=("Frequency", "max"), median_depth=("Coverage", "median"))
        .sort_values(["project", "samples_with_candidate_iSNV", "position"], ascending=[True, False, True])
    )
    write_table(filt, out_dir, "lofreq_g_snv_calls_3to97.csv")
    write_table(counts, out_dir, "per_sample_g_isnv_counts.csv")
    write_table(recurrence, out_dir, "g_isnv_site_recurrence.csv")
    return filt, counts, recurrence


def draw_g_isnv_summary(calls: pd.DataFrame, counts: pd.DataFrame, fig_dir: Path) -> tuple[plt.Figure, dict[str, str]]:
    setup_style(10.5)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), gridspec_kw={"width_ratios": [0.92, 1.02, 1.20], "wspace": 0.38})
    ax = axes[0]
    for idx, project in enumerate(PROJECT_ORDER):
        vals = counts.loc[counts["project"].eq(project), "candidate_G_iSNVs"].to_numpy()
        rng = np.random.default_rng(20 + idx)
        ax.scatter(np.full(len(vals), idx) + rng.normal(0, 0.035, len(vals)), vals, s=28, color=PROJECTS[project]["color"], alpha=0.76, edgecolor="white", linewidth=0.35, zorder=3)
        ax.boxplot(vals, positions=[idx], widths=0.36, showfliers=False, patch_artist=True, boxprops={"facecolor": "white", "edgecolor": "#333333", "linewidth": 1.0}, medianprops={"color": "#333333", "linewidth": 1.2}, whiskerprops={"color": "#333333", "linewidth": 0.9}, capprops={"color": "#333333", "linewidth": 0.9})
    ax.set_xticks([0, 1])
    ax.set_xticklabels([PROJECTS[p]["label"] for p in PROJECT_ORDER])
    ax.set_ylabel("Candidate G-gene iSNVs per sample")
    ax.set_title("Per-sample iSNV burden", loc="left", pad=7)
    ax.grid(axis="y", color="#EAEAEA", lw=0.6)
    ax.set_axisbelow(True)
    clean_axis(ax)
    panel_label(ax, "C", x=-0.16)

    ax = axes[1]
    bins = np.linspace(3, 97, 20)
    for project in PROJECT_ORDER:
        sub = calls[calls["project"].eq(project)]
        ax.hist(sub["Frequency"].dropna(), bins=bins, histtype="stepfilled", alpha=0.38, color=PROJECTS[project]["color"], edgecolor=PROJECTS[project]["color"], linewidth=1.1, label=PROJECTS[project]["label"])
    ax.set_xlabel("iSNV allele frequency (%)")
    ax.set_ylabel("Candidate calls")
    ax.set_title("Allele-frequency distribution", loc="left", pad=7)
    ax.legend(frameon=False)
    clean_axis(ax)
    panel_label(ax, "D", x=-0.14)

    ax = axes[2]
    ax.axvspan(HVR2_NT_START, HVR2_NT_END, color="#F4A38D", alpha=0.12, lw=0, zorder=0)
    for project in PROJECT_ORDER:
        sub = calls[calls["project"].eq(project)]
        ax.scatter(sub["position"], sub["Frequency"], s=24, color=PROJECTS[project]["color"], edgecolor="white", linewidth=0.28, alpha=0.58, label=PROJECTS[project]["label"])
    ax.text((HVR2_NT_START + HVR2_NT_END) / 2, 99.3, "HVR2", ha="center", va="bottom", fontsize=9.5, color="#B84A31")
    ax.set_xlim(G_START - 20, G_END + 20)
    ax.set_ylim(0, 104)
    ax.set_xlabel("Genome coordinate within G")
    ax.set_ylabel("Allele frequency (%)")
    ax.set_title("G-region iSNV coordinates", loc="left", pad=7)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.98))
    ax.grid(axis="y", color="#ECECEC", lw=0.55)
    ax.set_axisbelow(True)
    clean_axis(ax)
    panel_label(ax, "S1", x=-0.13)
    fig.tight_layout()
    paths = save_pub_figure(fig, fig_dir, "Fig1CD_and_FigS1_G_iSNV_summary")
    return fig, paths



def _linked_site_plot_specs(table: pd.DataFrame | None = None) -> list[dict[str, object]]:
    palettes = {
        "PRJNA1037681": ["#3C5488", "#4DBBD5", "#00A087", "#8491B4", "#5C7A99", "#7E6148", "#B09C85"],
        "PRJNA1130896": ["#D55E00", "#E69F00", "#0072B2", "#009E73", "#CC79A7"],
    }
    markers = {"PRJNA1037681": "o", "PRJNA1130896": "s"}
    specs = []
    for project in PROJECT_ORDER:
        sites = []
        label_combo = None
        if table is not None and "project" in table.columns:
            sub = table[table["project"].eq(project)].copy()
            if "position" in sub.columns:
                for pos in sorted(sub["position"].dropna().astype(int).unique()):
                    sites.append({"position": int(pos)})
            elif "site_positions" in sub.columns and not sub.empty:
                positions = str(sub["site_positions"].dropna().iloc[0]).split(";")
                sites = [{"position": int(pos)} for pos in positions if str(pos).strip()]
            if "selected_TreeCluster_combo" in sub.columns and sub["selected_TreeCluster_combo"].notna().any():
                label_combo = str(sub["selected_TreeCluster_combo"].dropna().iloc[0])
        if label_combo is None:
            label_combo = ""
        if not sites:
            raise ValueError(f"No linked-site positions were available for {project}")
        specs.append({
            "project": project,
            "label": f"{PROJECTS[project]['label']} {label_combo}".strip(),
            "sites": sites,
            "palette": palettes[project],
            "marker": markers[project],
        })
    return specs

def covarying_site_screening(root: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = normalized_lofreq_raw(root)
    selected = selected_linked_validation_samples(root)
    site_sets = derive_linked_site_sets(root)
    screen_table = getattr(derive_linked_site_sets, "last_screen_table", pd.DataFrame())
    write_table(selected, out_dir, "selected_linked_isnv_validation_samples.csv")
    if not screen_table.empty:
        write_table(screen_table, out_dir, "linked_site_screen_from_lofreq.csv")

    raw_idx = raw.set_index(["project", "SampleID", "position", "Reference", "Allele"], drop=False)
    rows = []
    for sample_row in selected.itertuples(index=False):
        project = sample_row.project
        for site in site_sets[project]:
            pos, ref, alt = site["position"], site["ref"], site["alt"]
            key = (project, sample_row.SampleID, pos, ref, alt)
            status = "not_called_target_alt"
            af = np.nan
            dp = np.nan
            ref_depth = np.nan
            alt_depth = np.nan
            qual = np.nan
            filt = np.nan
            other_alts = ""
            at_pos = raw[(raw["project"].eq(project)) & (raw["SampleID"].eq(sample_row.SampleID)) & (raw["position"].eq(pos))]
            if not at_pos.empty:
                other_alts = ";".join(sorted(at_pos["Allele"].dropna().astype(str).unique()))
            if key in raw_idx.index:
                hit = raw_idx.loc[key]
                if isinstance(hit, pd.DataFrame):
                    hit = hit.iloc[0]
                af = float(hit["AF"])
                dp = float(hit.get("DP", np.nan))
                ref_depth = float(hit.get("ref_depth", np.nan)) if "ref_depth" in hit.index else np.nan
                alt_depth = float(hit.get("alt_depth", np.nan)) if "alt_depth" in hit.index else np.nan
                qual = hit.get("qual", np.nan)
                filt = hit.get("filter", np.nan)
                status = "called_target_alt"
            elif not at_pos.empty:
                status = "other_alt_at_position"

            plot_af = 0.0 if pd.isna(af) else af
            aa_position = int(np.floor((pos - G_START) / 3 + 1))
            rows.append({
                "project": project,
                "project_label": PROJECTS[project]["label"],
                "project_short": PROJECTS[project]["short"],
                "SampleID": sample_row.SampleID,
                "global_short_label": getattr(sample_row, "global_short_label", sample_row.SampleID),
                "validation_sample_set": getattr(sample_row, "validation_sample_set", f"{PROJECTS[project]['short']} linked iSNV validation set"),
                "selected_TreeCluster_combo": getattr(sample_row, "selected_TreeCluster_combo", target_branch_combo(root, project)),
                "collection_date": getattr(sample_row, "collection_date", np.nan),
                "year_month": getattr(sample_row, "year_month", np.nan),
                "clade": getattr(sample_row, "clade", np.nan),
                "position": pos,
                "aa_position": aa_position,
                "Reference": ref,
                "Allele": alt,
                "lofreq_status": status,
                "other_alts_at_position": other_alts,
                "AF": af,
                "plot_AF": plot_af,
                "target_allele_frequency_percent": plot_af * 100,
                "reference_allele_frequency_percent": (1 - plot_af) * 100,
                "alt_frequency_percent": plot_af * 100,
                "ref_frequency_percent": (1 - plot_af) * 100,
                "DP": dp,
                "ref_depth": ref_depth,
                "alt_depth": alt_depth,
                "QUAL": qual,
                "FILTER": filt,
            })
    freq = pd.DataFrame(rows).sort_values(["project", "collection_date", "SampleID", "position"])
    summary = freq.groupby(["project", "project_label", "validation_sample_set", "selected_TreeCluster_combo", "position", "aa_position", "Reference", "Allele", "lofreq_status"], as_index=False).agg(
        samples=("SampleID", "nunique"),
        median_alt_frequency_percent=("alt_frequency_percent", "median"),
        min_alt_frequency_percent=("alt_frequency_percent", "min"),
        max_alt_frequency_percent=("alt_frequency_percent", "max"),
        median_depth=("DP", "median"),
    )
    corr_rows = []
    for project in PROJECT_ORDER:
        sub = freq[freq["project"].eq(project)]
        pivot = sub.pivot_table(index="SampleID", columns="position", values="plot_AF", aggfunc="first")
        positions = list(pivot.columns)
        for i, p1 in enumerate(positions):
            for p2 in positions[i + 1:]:
                tmp = pivot[[p1, p2]].dropna()
                r = tmp[p1].corr(tmp[p2]) if len(tmp) >= 3 else np.nan
                corr_rows.append({"project": project, "project_label": PROJECTS[project]["label"], "position_a": int(p1), "position_b": int(p2), "n_samples": len(tmp), "pearson_r": r, "abs_pearson_r": abs(r) if pd.notna(r) else np.nan})
    correlations = pd.DataFrame(corr_rows).sort_values(["project", "abs_pearson_r"], ascending=[True, False])
    write_table(freq, out_dir, "targeted_site_frequencies_from_lofreq.csv")
    write_table(summary, out_dir, "targeted_site_frequency_summary.csv")
    write_table(correlations, out_dir, "targeted_site_pairwise_pearson_correlations.csv")
    write_table(correlations[correlations["abs_pearson_r"] >= 0.80], out_dir, "targeted_site_pairs_abs_r_ge_0p8.csv")
    return freq, summary, correlations

def draw_covarying_sites(freq: pd.DataFrame, correlations: pd.DataFrame, fig_dir: Path) -> tuple[plt.Figure, dict[str, str]]:
    setup_style(10.0)
    selected = (
        freq[["project", "SampleID", "global_short_label", "collection_date"]]
        .drop_duplicates()
        .assign(collection_dt=lambda d: pd.to_datetime(d["collection_date"], errors="coerce"))
        .sort_values(["project", "collection_dt", "SampleID"])
    )
    plot_samples = selected["SampleID"].tolist()
    x_map = {sample: i for i, sample in enumerate(plot_samples)}
    plot_labels = selected["global_short_label"].tolist()

    fig = plt.figure(figsize=(max(16.8, len(plot_samples) * 0.245), 5.2), constrained_layout=False)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.28], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    ax_key = fig.add_subplot(gs[0, 1])
    ax_key.set_axis_off()

    for spec in _linked_site_plot_specs(freq):
        project = spec["project"]
        samples = selected.loc[selected["project"].eq(project), "SampleID"].tolist()
        if not samples:
            continue
        start = x_map[samples[0]] - 0.5
        end = x_map[samples[-1]] + 0.5
        center = (start + end) / 2
        ax.axvspan(start, end, color=PROJECTS[project]["light"], alpha=0.58, zorder=0)
        ax.text(center, 103.2, PROJECTS[project]["label"], ha="center", va="bottom", fontsize=11.0, color="#333333")

    cursor = 0
    for spec in _linked_site_plot_specs(freq)[:-1]:
        cursor += len(selected.loc[selected["project"].eq(spec["project"]), "SampleID"])
        ax.axvline(cursor - 0.5, color="#BDBDBD", lw=0.85, ls="--", zorder=1)

    site_handles = []
    for spec in _linked_site_plot_specs(freq):
        project = spec["project"]
        for idx, site in enumerate(spec["sites"]):
            pos = site["position"]
            color = spec["palette"][idx % len(spec["palette"])]
            sub = freq[(freq["project"].eq(project)) & (freq["position"].eq(pos))].copy()
            sub["x"] = sub["SampleID"].map(x_map)
            sub = sub.sort_values("x")
            ax.plot(sub["x"], sub["target_allele_frequency_percent"], color=color, lw=2.0, alpha=0.94, ls="-", marker=spec["marker"], markersize=4.7, markeredgecolor="white", markeredgewidth=0.45, zorder=3)
            ax.plot(sub["x"], sub["reference_allele_frequency_percent"], color=color, lw=1.75, alpha=0.72, ls="--", marker=spec["marker"], markersize=4.2, markeredgecolor="white", markeredgewidth=0.45, zorder=2)
            site_handles.append(Line2D([0], [0], color=color, lw=2.0, marker=spec["marker"], markersize=4.8, label=f"{PROJECTS[project]['short']} nt {pos}"))

    ax.set_ylim(-2, 108)
    ax.set_xlim(-0.5, len(plot_samples) - 0.5)
    ax.set_ylabel("Allele frequency (%)")
    ax.set_title("Allele frequencies at correlation-selected linked iSNV sites", loc="left", fontweight="bold", fontsize=13.4)
    ax.set_xticks(np.arange(len(plot_samples)))
    ax.set_xticklabels(plot_labels, rotation=90, ha="center", fontsize=6.8)
    ax.set_xlabel("Linked-iSNV validation samples")
    ax.grid(axis="y", color="#E8E8E8", lw=0.62, zorder=0)
    ax.set_axisbelow(True)
    clean_axis(ax)
    panel_label(ax, "A", x=-0.052, y=1.03, size=14.6)

    style_handles = [
        Line2D([0], [0], color="#333333", lw=2.1, ls="-", label="Target allele"),
        Line2D([0], [0], color="#333333", lw=1.8, ls="--", label="Reference allele"),
    ]
    leg_style = ax_key.legend(handles=style_handles, title="Line type", frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.98), fontsize=9.0, title_fontsize=9.8, handlelength=2.4, labelspacing=0.46)
    ax_key.add_artist(leg_style)
    ax_key.legend(handles=site_handles, title="Candidate sites", frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.02), fontsize=8.4, title_fontsize=9.5, handlelength=2.2, labelspacing=0.34)
    fig.tight_layout()
    paths = save_pub_figure(fig, fig_dir, "Fig2A_allele_frequency_validation")
    return fig, paths



def haplotype_reconstruction(root: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = selected_linked_validation_samples(root)
    selected_keep = selected[["project", "SampleID", "global_short_label", "collection_date", "selected_TreeCluster_combo"]].drop_duplicates()
    hap = build_haplotype_site_combinations(root, selected_keep)
    hap = hap.merge(selected_keep, on=["project", "SampleID"], how="left")
    hap = hap[hap["project"].isin(PROJECT_ORDER)].copy()
    hap["collection_dt"] = pd.to_datetime(hap["collection_date"], errors="coerce")

    combo_matrix = hap.pivot_table(index=["project", "SampleID", "global_short_label"], columns="site_combo", values="haplotype_frequency", aggfunc="sum", fill_value=0).reset_index()
    count_values = haplotype_count_distribution_from_input(root)
    count_values["haplotype_count_class"] = count_values["haplotype_count_class"].astype(str).replace({"4": "4+"})
    count_values["project_label"] = count_values["project"].map(lambda p: PROJECTS[p]["label"])

    state_rows = []
    for row in hap.itertuples(index=False):
        sites = str(row.site_positions).split(";")
        bases = str(row.site_combo).split("-")
        rec = {"project": row.project, "SampleID": row.SampleID, "SequenceName": row.SequenceName, "haplotype_id": row.haplotype_id, "haplotype_frequency": row.haplotype_frequency, "site_combo": row.site_combo}
        for site, base in zip(sites, bases):
            rec[f"nt{site}"] = base
        state_rows.append(rec)
    state_matrix = pd.DataFrame(state_rows)
    write_table(hap.sort_values(["project", "collection_dt", "SampleID", "haplotype_id"]), out_dir, "haplotype_site_combos_per_haplotype.csv")
    write_table(combo_matrix, out_dir, "haplotype_combo_frequency_matrix.csv")
    write_table(count_values.sort_values(["project", "haplotype_count_class"]), out_dir, "haplotype_count_distribution.csv")
    write_table(state_matrix, out_dir, "haplotype_site_state_matrix.csv")
    return hap, combo_matrix, count_values

def draw_haplotype_reconstruction(hap: pd.DataFrame, count_values: pd.DataFrame, fig_dir: Path) -> tuple[plt.Figure, dict[str, str]]:
    setup_style(10.0)
    selected = (
        hap[["project", "SampleID", "global_short_label", "collection_date"]]
        .drop_duplicates()
        .assign(collection_dt=lambda d: pd.to_datetime(d["collection_date"], errors="coerce"))
        .sort_values(["project", "collection_dt", "SampleID"])
    )
    plot_samples = selected["SampleID"].tolist()
    x_map = {sample: i for i, sample in enumerate(plot_samples)}
    plot_labels = selected["global_short_label"].tolist()
    hap_combo_freq = {sample: {} for sample in plot_samples}
    for row in hap.itertuples(index=False):
        hap_combo_freq.setdefault(row.SampleID, {})[row.site_combo] = hap_combo_freq.setdefault(row.SampleID, {}).get(row.site_combo, 0.0) + float(row.haplotype_frequency)

    fig_width = max(16.8, len(plot_samples) * 0.245)
    fig = plt.figure(figsize=(fig_width, 7.7), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.12, 0.78], width_ratios=[1.0, 0.28], hspace=0.46, wspace=0.08)
    ax_b = fig.add_subplot(gs[0, 0])
    ax_b_key = fig.add_subplot(gs[0, 1])
    ax_blank = fig.add_subplot(gs[1, 1])
    ax_b_key.set_axis_off()
    ax_blank.set_axis_off()
    bottom_gs = gs[1, 0].subgridspec(1, 2, wspace=0.30)
    ax_c = fig.add_subplot(bottom_gs[0, 0])
    ax_d = fig.add_subplot(bottom_gs[0, 1], sharey=ax_c)

    for ax in [ax_b]:
        for spec in _linked_site_plot_specs(hap):
            project = spec["project"]
            samples = selected.loc[selected["project"].eq(project), "SampleID"].tolist()
            if not samples:
                continue
            start = x_map[samples[0]] - 0.5
            end = x_map[samples[-1]] + 0.5
            center = (start + end) / 2
            ax.axvspan(start, end, color=PROJECTS[project]["light"], alpha=0.58, zorder=0)
            ax.text(center, 103.2, PROJECTS[project]["label"], ha="center", va="bottom", fontsize=11.0, color="#333333")
        cursor = 0
        for spec in _linked_site_plot_specs(hap)[:-1]:
            cursor += len(selected.loc[selected["project"].eq(spec["project"]), "SampleID"])
            ax.axvline(cursor - 0.5, color="#BDBDBD", lw=0.85, ls="--", zorder=1)
        ax.grid(axis="y", color="#E8E8E8", lw=0.62, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.5, len(plot_samples) - 0.5)
        clean_axis(ax)

    cohort_hap_styles = {
        "PRJNA1037681": {"colors": ["#6A51A3", "#1B9E77", "#7A8FB3"], "markers": ["o", "s", "D"]},
        "PRJNA1130896": {"colors": ["#D55E00", "#0072B2", "#B75D4D"], "markers": ["o", "s", "D"]},
    }
    hap_handles = []
    for spec in _linked_site_plot_specs(hap):
        project = spec["project"]
        project_samples = selected.loc[selected["project"].eq(project), "SampleID"].tolist()
        project_combo_total = {}
        for sample in project_samples:
            for combo, freq in hap_combo_freq.get(sample, {}).items():
                project_combo_total[combo] = project_combo_total.get(combo, 0.0) + freq
        project_combos = sorted(project_combo_total, key=lambda c: (-project_combo_total[c], c))[:2]
        styles = cohort_hap_styles[project]
        for rank, combo in enumerate(project_combos):
            xs, ys = [], []
            for sample in project_samples:
                freq = hap_combo_freq.get(sample, {}).get(combo, 0.0)
                if freq > 0:
                    xs.append(x_map[sample])
                    ys.append(freq * 100.0)
            if not xs:
                continue
            order_idx = np.argsort(xs)
            xs = np.array(xs)[order_idx]
            ys = np.array(ys)[order_idx]
            linestyle = "-" if rank == 0 else "--"
            color = styles["colors"][rank % len(styles["colors"])]
            marker = styles["markers"][rank % len(styles["markers"])]
            ax_b.plot(xs, ys, color=color, lw=2.05, alpha=0.94, ls=linestyle, marker=marker, markersize=4.9, markeredgecolor="white", markeredgewidth=0.45, zorder=3)
            hap_handles.append(Line2D([0], [0], color=color, lw=2.05, ls=linestyle, marker=marker, markersize=4.9, label=f"{PROJECTS[project]['short']} {combo}"))
    ax_b.set_ylim(-2, 108)
    ax_b.set_ylabel("Haplotype frequency (%)")
    ax_b.set_title("Reconstructed haplotype frequencies", loc="left", fontweight="bold", fontsize=13.4)
    ax_b.set_xticks(np.arange(len(plot_samples)))
    ax_b.set_xticklabels(plot_labels, rotation=90, ha="center", fontsize=6.8)
    ax_b.set_xlabel("Linked-iSNV validation samples")
    panel_label(ax_b, "B", x=-0.052, y=1.03, size=14.6)
    ax_b_key.legend(handles=hap_handles, title="Reconstructed haplotypes", frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.98), fontsize=8.8, title_fontsize=9.7, handlelength=2.35, labelspacing=0.42)

    base_classes = ["1", "2", "3", "4+"]
    class_totals = count_values.groupby("haplotype_count_class")["n_samples"].sum().to_dict()
    classes = [c for c in base_classes if c != "4+" or class_totals.get("4+", 0) > 0]
    ymax = 0
    for project in PROJECT_ORDER:
        vals = count_values[count_values["project"].eq(project)].set_index("haplotype_count_class")["n_samples"].reindex(classes, fill_value=0)
        ymax = max(ymax, int(vals.max()))

    for ax, project, letter in [(ax_c, "PRJNA1037681", "C"), (ax_d, "PRJNA1130896", "D")]:
        cfg = PROJECTS[project]
        vals = count_values[count_values["project"].eq(project)].set_index("haplotype_count_class")["n_samples"].reindex(classes, fill_value=0)
        x = np.arange(len(classes))
        ax.bar(x, vals.to_numpy(), width=0.58, color=cfg["color"], edgecolor="#333333", linewidth=0.70)
        for xi, value in zip(x, vals.to_numpy()):
            ax.text(xi, value + max(0.6, ymax * 0.018), str(int(value)), ha="center", va="bottom", fontsize=10.7)
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.set_xlabel("Reconstructed G-gene\nhaplotypes per sample", fontsize=11.2)
        ax.set_title(f"{cfg['label']}: haplotypes per sample", loc="left", fontweight="bold", fontsize=12.4)
        ax.grid(axis="y", color="#E8E8E8", lw=0.62)
        ax.set_axisbelow(True)
        ax.set_ylim(0, ymax * 1.18 + 1.5)
        clean_axis(ax)
        panel_label(ax, letter, x=-0.12 if letter == "C" else -0.10, y=1.04, size=14.2)
    ax_c.set_ylabel("Samples")
    ax_d.set_ylabel("")
    ax_d.tick_params(axis="y", labelleft=False)
    fig.tight_layout()
    paths = save_pub_figure(fig, fig_dir, "Fig2BCD_haplotype_reconstruction")
    return fig, paths


def phylogeny_branch_assignment(root: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    combos, assignments, manifest = get_sample_branch_combinations(root)
    cache = treecluster_cache_dir(root)
    write_table(combos, cache, "sample_treecluster_branch_combinations.csv")
    commands = treecluster_generation_commands(root)

    hap = load_haplotype_records(root)
    valid = assignments.dropna(subset=["SampleID"]).copy()
    valid = valid[valid["TreeCluster_branch"].ne("Unclustered")].copy()
    valid = valid.merge(hap[["SequenceName", "G_seq", "Frequency"]], on="SequenceName", how="left")
    branch_matrix = valid.pivot_table(index=["project", "SampleID"], columns="TreeCluster_branch", values="haplotype_frequency", aggfunc="sum", fill_value=0).reset_index()
    branch_matrix = branch_matrix.merge(combos[["project", "SampleID", "clade", "TreeCluster_combo"]], on=["project", "SampleID"], how="left")
    branch_cols = sorted_branches([c for c in branch_matrix.columns if str(c).startswith("C") or str(c) == "Unclustered"])
    branch_matrix = branch_matrix[["project", "SampleID", "clade", "TreeCluster_combo"] + branch_cols]

    manifest = manifest.merge(commands.groupby("project", as_index=False).agg(commands=("command", lambda x: " then ".join(x))), on="project", how="left")
    write_table(assignments, out_dir, "treecluster_haplotype_assignments.csv")
    write_table(branch_matrix, out_dir, "sample_branch_frequency_matrix.csv")
    write_table(combos, out_dir, "sample_treecluster_branch_combinations.csv")
    write_table(commands, out_dir, "treecluster_generation_commands.csv")
    write_table(manifest, out_dir, "phylogeny_branch_input_manifest.csv")
    return assignments, branch_matrix, manifest

def draw_branch_frequency_heatmaps(branch_matrix: pd.DataFrame, fig_dir: Path) -> tuple[plt.Figure, dict[str, str]]:
    setup_style(9.5)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.4), gridspec_kw={"wspace": 0.30})
    last_im = None
    for ax, project, panel in zip(axes, PROJECT_ORDER, ["B", "D"]):
        target = target_combo_from_table(branch_matrix, project).split("+")
        sub = branch_matrix[branch_matrix["project"].eq(project)].copy()
        sub["target_frequency"] = sub[target].sum(axis=1) if all(t in sub.columns for t in target) else 0
        sub = sub.sort_values(["TreeCluster_combo", "target_frequency", "SampleID"], ascending=[False, False, True])
        matrix = sub[target].to_numpy(float) * 100
        cmap = mcolors.LinearSegmentedColormap.from_list(PROJECTS[project]["label"], ["#F7F7F7", PROJECTS[project]["color"]])
        last_im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=100)
        ax.set_xticks(np.arange(len(target)))
        ax.set_xticklabels(target)
        y_labels = sub["SampleID"].tolist()
        if len(y_labels) > 45:
            tick_idx = np.linspace(0, len(y_labels) - 1, 12, dtype=int)
            ax.set_yticks(tick_idx)
            ax.set_yticklabels([y_labels[i] for i in tick_idx], fontsize=6.2)
        else:
            ax.set_yticks(np.arange(len(y_labels)))
            ax.set_yticklabels(y_labels, fontsize=6.0)
        ax.set_xlabel("TreeCluster branch")
        ax.set_ylabel("Samples ordered by branch composition")
        ax.set_title(f"{PROJECTS[project]['label']}: focal branch composition", loc="left", pad=7)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        panel_label(ax, panel, x=-0.10)
    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.022, pad=0.020)
    cbar.set_label("Haplotype frequency (%)")
    paths = save_pub_figure(fig, fig_dir, "Fig3BD_branch_composition_heatmaps")
    return fig, paths


def _eligible_exact_combos(df: pd.DataFrame) -> list[str]:
    eligible = set()
    for _, g in df.groupby("exchange_group", sort=False):
        branches = sorted(g["TreeCluster_branch"].astype(str).unique(), key=branch_sort_key)
        sample_h = g.groupby("SampleID").size()
        max_h = int(sample_h.max()) if len(sample_h) else 0
        max_k = min(max_h, len(branches))
        for k in range(2, max_k + 1):
            for comb in combinations(branches, k):
                eligible.add("+".join(comb))
    return sorted(eligible, key=lambda c: (combo_size(c), [branch_sort_key(x) for x in c.split("+")]))


def _sample_combo_counts(sample_to_labels: dict[str, list[str]]) -> Counter:
    counts = Counter()
    for labels in sample_to_labels.values():
        combo = combo_from_branches(labels)
        if combo_size(combo) >= 2:
            counts[combo] += 1
    return counts


def _build_sample_to_labels(df: pd.DataFrame, label_col: str = "TreeCluster_branch") -> dict[str, list[str]]:
    out = defaultdict(list)
    for sample, label in zip(df["SampleID"], df[label_col]):
        out[str(sample)].append(str(label))
    return dict(out)


def _permute_once(df: pd.DataFrame, groups: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, list[str]]:
    labels = df["TreeCluster_branch"].astype(str).to_numpy().copy()
    permuted = labels.copy()
    for idx in groups.values():
        permuted[idx] = rng.permutation(labels[idx])
    sample_to_labels = defaultdict(list)
    samples = df["SampleID"].astype(str).to_numpy()
    for sample, label in zip(samples, permuted):
        sample_to_labels[sample].append(str(label))
    return dict(sample_to_labels)


def _empirical_p_ge(null_values: np.ndarray, observed: float) -> float:
    return float((1 + np.sum(null_values >= observed)) / (len(null_values) + 1))


def _run_exact_combo_permutation(project: str, hap_meta: pd.DataFrame, n_perm: int, rng: np.random.Generator, target_combo: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    cfg = PROJECTS[project]
    df = hap_meta[hap_meta["project"].eq(project)].copy().reset_index(drop=True)
    df["exchange_group"] = df["year_month"].astype(str) + "|" + df["geo_stratum"].astype(str)
    eligible = _eligible_exact_combos(df)
    eligible_index = {c: i for i, c in enumerate(eligible)}
    groups = {str(k): v.to_numpy() for k, v in df.groupby("exchange_group").groups.items()}

    observed_counts = _sample_combo_counts(_build_sample_to_labels(df))
    observed_target = int(observed_counts.get(target_combo, 0))
    observed_max = max([observed_counts.get(c, 0) for c in eligible], default=0)
    counts = np.zeros((n_perm, len(eligible)), dtype=np.int16)
    target_idx = eligible_index[target_combo]
    for i in range(n_perm):
        perm_counts = _sample_combo_counts(_permute_once(df, groups, rng))
        for combo, count in perm_counts.items():
            j = eligible_index.get(combo)
            if j is not None:
                counts[i, j] = count

    means = counts.mean(axis=0)
    sds = counts.std(axis=0, ddof=0)
    target_null = counts[:, target_idx]
    target_mean = float(means[target_idx])
    target_sd = float(sds[target_idx])
    target_z = (observed_target - target_mean) / target_sd if target_sd > 0 else np.inf
    max_count_null = counts.max(axis=1)
    z_counts = np.full_like(counts, fill_value=-np.inf, dtype=float)
    nonconstant = sds > 0
    z_counts[:, nonconstant] = (counts[:, nonconstant] - means[nonconstant]) / sds[nonconstant]
    maxT_null = z_counts.max(axis=1)

    family_rows = []
    for combo, j in eligible_index.items():
        observed = int(observed_counts.get(combo, 0))
        family_rows.append({
            "project": project,
            "project_label": cfg["label"],
            "target_combo": target_combo,
            "TreeCluster_combo": combo,
            "combo_size": combo_size(combo),
            "observed_count": observed,
            "null_mean": means[j],
            "null_sd": sds[j],
            "null_q025": float(np.quantile(counts[:, j], 0.025)),
            "null_median": float(np.quantile(counts[:, j], 0.5)),
            "null_q975": float(np.quantile(counts[:, j], 0.975)),
            "observed_expected_ratio": observed / means[j] if means[j] > 0 else np.nan,
            "unadjusted_p_count": _empirical_p_ge(counts[:, j], observed),
            "observed_z": (observed - means[j]) / sds[j] if sds[j] > 0 else np.nan,
            "is_target_combo": combo == target_combo,
            "observed_nonzero": observed > 0,
        })
    family = pd.DataFrame(family_rows).sort_values(["observed_count", "combo_size", "TreeCluster_combo"], ascending=[False, True, True]).reset_index(drop=True)
    family["observed_rank_all_eligible"] = np.arange(1, len(family) + 1)

    summary = {
        "project": project,
        "project_label": cfg["label"],
        "target_combo": target_combo,
        "analysis_mode": "month_region_all_eligible_exact_combo",
        "stratification": "year_month + geo_stratum",
        "n_permutations": n_perm,
        "n_samples": df["SampleID"].nunique(),
        "n_haplotypes": len(df),
        "n_exchangeability_strata": len(groups),
        "n_eligible_exact_cross_cluster_combos": len(eligible),
        "n_observed_cross_cluster_combos": sum(v > 0 for v in observed_counts.values()),
        "observed_target_count": observed_target,
        "observed_max_all_eligible_count": int(observed_max),
        "target_null_mean": target_mean,
        "target_null_sd": target_sd,
        "target_null_q025": float(np.quantile(target_null, 0.025)),
        "target_null_median": float(np.quantile(target_null, 0.5)),
        "target_null_q975": float(np.quantile(target_null, 0.975)),
        "target_observed_expected_ratio": observed_target / target_mean if target_mean > 0 else np.nan,
        "target_unadjusted_p_count": _empirical_p_ge(target_null, observed_target),
        "target_z": target_z,
        "all_eligible_max_count_null_mean": float(max_count_null.mean()),
        "all_eligible_max_count_null_q025": float(np.quantile(max_count_null, 0.025)),
        "all_eligible_max_count_null_median": float(np.quantile(max_count_null, 0.5)),
        "all_eligible_max_count_null_q975": float(np.quantile(max_count_null, 0.975)),
        "all_eligible_max_count_adjusted_p": _empirical_p_ge(max_count_null, observed_target),
        "all_eligible_maxT_null_q025": float(np.quantile(maxT_null, 0.025)),
        "all_eligible_maxT_null_median": float(np.quantile(maxT_null, 0.5)),
        "all_eligible_maxT_null_q975": float(np.quantile(maxT_null, 0.975)),
        "all_eligible_maxT_adjusted_p": _empirical_p_ge(maxT_null, target_z),
    }
    draws = pd.DataFrame({
        "project": project,
        "project_label": cfg["label"],
        "target_combo": target_combo,
        "iteration": np.arange(1, n_perm + 1),
        "permuted_target_count": target_null,
        "permuted_max_all_eligible_exact_combo_count": max_count_null,
        "permuted_max_all_eligible_exact_combo_z": maxT_null,
    })
    return summary, family, draws


def cross_branch_permutation(root: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    combos, assignments, _ = get_sample_branch_combinations(root)
    hap = load_haplotype_records(root)
    meta = load_metadata(root)
    valid = assignments.dropna(subset=["SampleID"]).copy()
    valid = valid[valid["TreeCluster_branch"].ne("Unclustered")].copy()
    hap_meta = valid.merge(hap[["SequenceName", "G_seq"]], on="SequenceName", how="left")
    hap_meta = hap_meta.merge(meta[["project", "SampleID", "year_month", "geo_stratum", "clade"]], on=["project", "SampleID"], how="left")

    observed = combos[combos["project"].isin(PROJECT_ORDER)].groupby(["project", "project_label", "TreeCluster_combo"], as_index=False).agg(n_samples=("SampleID", "nunique"), median_n_haplotypes=("n_haplotypes", "median"), median_n_branches=("n_TreeCluster_branches", "median"))
    targets = select_target_branch_combos(root).set_index("project")
    observed["is_target_combo"] = observed.apply(lambda r: r["TreeCluster_combo"] == targets.loc[r["project"], "target_combo"], axis=1)

    n_perm = int(os.environ.get("RSVA_N_PERMUTATIONS", "20000"))
    rng = np.random.default_rng(20260604)
    summaries = []
    families = []
    draw_tables = []
    for project in PROJECT_ORDER:
        summary, family, draws = _run_exact_combo_permutation(project, hap_meta, n_perm, rng, targets.loc[project, "target_combo"])
        summaries.append(summary)
        families.append(family)
        draw_tables.append(draws)
    summary = pd.DataFrame(summaries)
    draws = pd.concat(draw_tables, ignore_index=True)
    family = pd.concat(families, ignore_index=True)
    write_table(observed.sort_values(["project", "n_samples"], ascending=[True, False]), out_dir, "observed_treecluster_combo_counts.csv")
    write_table(summary, out_dir, "month_region_permutation_publication_summary.csv")
    write_table(draws, out_dir, "month_region_permutation_draws.csv")
    write_table(family, out_dir, "all_eligible_exact_combo_family_results.csv")
    return observed, summary, draws

def draw_permutation(summary: pd.DataFrame, draws: pd.DataFrame, fig_dir: Path) -> tuple[plt.Figure, dict[str, str]]:
    setup_style(10.0)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), gridspec_kw={"wspace": 0.28})
    for ax, project, panel in zip(axes, PROJECT_ORDER, ["E", "F"]):
        cfg = PROJECTS[project]
        sub_draws = draws[draws["project"].eq(project)].copy()
        row = summary[summary["project"].eq(project)].iloc[0]
        obs = int(row["observed_target_count"])
        bins = np.arange(0, max(obs, sub_draws["permuted_target_count"].max()) + 3) - 0.5
        ax.hist(sub_draws["permuted_target_count"], bins=bins, color=lighten_color(cfg["color"], 0.55), edgecolor="white", linewidth=0.35)
        ax.axvline(obs, color=cfg["color"], lw=2.3)
        ax.text(obs + 0.25, ax.get_ylim()[1] * 0.88, f"Observed = {obs}", color=cfg["color"], fontsize=10.2, fontweight="bold", ha="left")
        ax.set_xlabel(f"Permuted {row['target_combo']} sample count")
        ax.set_ylabel("Permutation draws")
        ax.set_title(f"{cfg['label']}: month-region permutation", loc="left", pad=7)
        p = float(row["target_unadjusted_p_count"])
        ax.text(0.98, 0.86, f"P = {p:.5f}\nO/E = {row['target_observed_expected_ratio']:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=9.4, color="#333333")
        clean_axis(ax)
        ax.grid(axis="y", color="#EAEAEA", lw=0.55)
        ax.set_axisbelow(True)
        panel_label(ax, panel, x=-0.12)
    fig.tight_layout()
    paths = save_pub_figure(fig, fig_dir, "Fig3EF_cross_branch_permutation")
    return fig, paths


def branch_defining_sites(root: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, freq_summary, _ = covarying_site_screening(root, out_dir / "covarying_site_screening")
    core = freq_summary[freq_summary["lofreq_status"].eq("called_target_alt")].copy()
    core = core.rename(columns={
        "selected_TreeCluster_combo": "TreeCluster_combo",
        "position": "nt_position_genome",
        "Reference": "ref_nt",
        "Allele": "target_nt",
    })
    keep = ["project", "validation_sample_set", "TreeCluster_combo", "nt_position_genome", "ref_nt", "target_nt", "aa_position", "lofreq_status", "samples", "median_alt_frequency_percent", "min_alt_frequency_percent", "max_alt_frequency_percent", "median_depth"]
    core = core[keep].sort_values(["project", "nt_position_genome"]).reset_index(drop=True)
    domain = g_protein_domain_layout()

    combos, assignments, _ = get_sample_branch_combinations(root)
    hap = load_haplotype_records(root)
    valid = assignments.dropna(subset=["SampleID"]).copy()
    valid = valid[valid["TreeCluster_branch"].ne("Unclustered")].copy()
    valid = valid.merge(hap[["SequenceName", "G_seq", "Frequency"]], on="SequenceName", how="left")
    aa_positions = sorted(core["aa_position"].dropna().astype(int).unique())
    matrix_rows = []
    targets = select_target_branch_combos(root).set_index("project")
    for project in PROJECT_ORDER:
        cfg = PROJECTS[project]
        target_combo = targets.loc[project, "target_combo"]
        target_branches = list(targets.loc[project, "target_branches"])
        target_samples = combos[(combos["project"].eq(project)) & (combos["TreeCluster_combo"].eq(target_combo))]["SampleID"].astype(str).unique()
        for branch in target_branches:
            sub = valid[(valid["project"].eq(project)) & (valid["TreeCluster_branch"].eq(branch)) & (valid["SampleID"].astype(str).isin(target_samples))].copy()
            rec = {"project": project, "project_label": cfg["label"], "TreeCluster_combo": target_combo, "branch": branch, "n_samples_total": int(len(set(target_samples)))}
            for aa in aa_positions:
                states = []
                weights = []
                for row in sub.itertuples(index=False):
                    aa_state = aa_from_sequence(row.G_seq, aa)
                    if pd.notna(aa_state):
                        states.append(str(aa_state))
                        weights.append(float(row.Frequency) if pd.notna(row.Frequency) else 1.0)
                if states:
                    weight_by_state = pd.DataFrame({"state": states, "weight": weights}).groupby("state")["weight"].sum().sort_values(ascending=False)
                    rec[f"AA{aa}"] = weight_by_state.index[0]
                else:
                    rec[f"AA{aa}"] = np.nan
            matrix_rows.append(rec)
    matrix = pd.DataFrame(matrix_rows)
    aa_cols = [c for c in matrix.columns if c.startswith("AA")]
    for project in PROJECT_ORDER:
        cfg = PROJECTS[project]
        branch_a, branch_b = list(targets.loc[project, "target_branches"])
        sub_idx = matrix["project"].eq(project)
        sub = matrix[sub_idx].set_index("branch")
        for col in aa_cols:
            a = sub.loc[branch_a, col] if branch_a in sub.index else np.nan
            b = sub.loc[branch_b, col] if branch_b in sub.index else np.nan
            if pd.isna(a) or pd.isna(b) or str(a) == str(b):
                matrix.loc[sub_idx, col] = np.nan

    rows = []
    for project in PROJECT_ORDER:
        cfg = PROJECTS[project]
        sub = matrix[matrix["project"].eq(project)].set_index("branch")
        branch_a, branch_b = list(targets.loc[project, "target_branches"])
        for col in aa_cols:
            aa_pos = int(col.replace("AA", ""))
            a = sub.loc[branch_a, col] if branch_a in sub.index else np.nan
            b = sub.loc[branch_b, col] if branch_b in sub.index else np.nan
            if pd.notna(a) and pd.notna(b) and str(a) != str(b):
                rows.append({"project": project, "project_label": cfg["label"], "TreeCluster_combo": targets.loc[project, "target_combo"], "branch_a": branch_a, "branch_b": branch_b, "aa_position": aa_pos, "branch_a_state": a, "branch_b_state": b, "aa_change": f"{a}>{b}", "n_samples_total": int(sub.loc[branch_a, "n_samples_total"])})
    aa_summary = pd.DataFrame(rows)
    write_table(core, out_dir, "core_linked_nt_sites_used.csv")
    write_table(matrix, out_dir, "hvr2_core_aa_site_matrix.csv")
    write_table(aa_summary, out_dir, "branch_defining_aa_site_summary.csv")
    write_table(domain, out_dir, "g_protein_domain_layout.csv")
    return core, matrix, aa_summary

def draw_branch_defining_sites(root: Path, core: pd.DataFrame, matrix: pd.DataFrame, fig_dir: Path) -> tuple[plt.Figure, dict[str, str]]:
    setup_style(10.0)
    domain = g_protein_domain_layout()
    aa_cols = [c for c in matrix.columns if c.startswith("AA")]
    aa_positions = [int(c.replace("AA", "")) for c in aa_cols]
    fig = plt.figure(figsize=(11.2, 4.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.78, 1.70], hspace=0.33)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_a.set_xlim(0, 330)
    ax_a.set_ylim(0, 1)
    track_y = 0.42
    track_h = 0.24
    for row in domain.itertuples(index=False):
        rect = patches.FancyBboxPatch((row.start, track_y), row.end - row.start, track_h, boxstyle="round,pad=0.012,rounding_size=0.02", facecolor=row.color, edgecolor="#666666", linewidth=0.72)
        ax_a.add_patch(rect)
        text_color = "white" if row.domain == "TM" else "#333333"
        ax_a.text((row.start + row.end) / 2, track_y + track_h / 2, row.domain, ha="center", va="center", fontsize=9.2, color=text_color)
    for aa in sorted(core["aa_position"].dropna().astype(int).unique()):
        ax_a.plot([aa, aa], [track_y + track_h, 0.88], color="#D55E00", lw=1.15)
        ax_a.scatter([aa], [0.88], s=34, color="#D55E00", edgecolor="white", linewidth=0.55, zorder=3)
        ax_a.text(aa, 0.96, str(aa), ha="center", va="bottom", fontsize=8.0, color="#70331D")
    ax_a.text(202, 0.22, "HVR2", ha="left", va="center", fontsize=9.8, color="#B04A32", fontstyle="italic")
    ax_a.set_title("G protein domain map and core linked HVR2 sites", loc="left", fontsize=12.2, fontweight="bold")
    ax_a.axis("off")
    panel_label(ax_a, "A", x=-0.045, y=1.03)

    plot_rows = []
    for project in PROJECT_ORDER:
        sub = matrix[matrix["project"].eq(project)].copy()
        rank = {b: i for i, b in enumerate(target_branches_for_project(root, project))}
        sub["_rank"] = sub["branch"].map(rank)
        for _, row in sub.sort_values("_rank").iterrows():
            plot_rows.append(row)
    ax_b.set_xlim(-3.0, len(aa_cols) + 0.7)
    ax_b.set_ylim(len(plot_rows) - 0.5, -0.95)
    for x, aa in enumerate(aa_positions):
        ax_b.text(x, -0.66, str(aa), ha="center", va="center", fontsize=9.6, color="#5A5A5A", fontweight="bold")
    for y, row in enumerate(plot_rows):
        color = BRANCH_COLORS.get(row["branch"], "#666666")
        ax_b.text(-2.86, y, "USA" if row["project_label"] == "United States" else row["project_label"], ha="left", va="center", fontsize=9.0, color="#555555")
        ax_b.scatter([-1.10], [y], s=56, color=color, edgecolor="white", linewidth=0.5, zorder=3)
        ax_b.text(-0.88, y, row["branch"], ha="left", va="center", fontsize=9.5, fontweight="bold", color=color)
        for x, col in enumerate(aa_cols):
            val = row[col]
            face = lighten_color(color, 0.34) if pd.notna(val) else "#F4F4F4"
            edge = color if pd.notna(val) else "#DDDDDD"
            rect = patches.Rectangle((x - 0.39, y - 0.34), 0.78, 0.68, facecolor=face, edgecolor=edge, linewidth=0.72)
            ax_b.add_patch(rect)
            ax_b.text(x, y, "" if pd.isna(val) else str(val), ha="center", va="center", fontsize=9.8, fontweight="bold", color="#222222")
    ax_b.text(np.mean(np.arange(len(aa_cols))), -0.92, "Amino-acid position", ha="center", va="center", fontsize=9.5, color="#555555")
    ax_b.set_title("Branch-defining amino-acid states", loc="left", fontsize=12.2, fontweight="bold", pad=16)
    ax_b.axis("off")
    panel_label(ax_b, "B", x=-0.045, y=1.04)
    paths = save_pub_figure(fig, fig_dir, "Fig4AB_branch_defining_HVR2_sites")
    return fig, paths



def _global_records_from_fasta(root: Path, site_sets: dict[str, list[dict[str, object]]] | None = None) -> pd.DataFrame:
    fasta = input_global_fasta_path(root)
    site_sets = site_sets or derive_linked_site_sets(root)
    recs = read_fasta_records(fasta).rename(columns={"record_id": "accession", "sequence": "G_seq", "length_nt": "G_length"})
    recs["G_ambiguous_fraction"] = recs["G_seq"].map(lambda s: sum(base not in "ACGT" for base in str(s).upper()) / len(str(s)) if len(str(s)) else np.nan)
    for project, cfg in PROJECTS.items():
        col = f"{cfg['short']}_site_combo"
        recs[col] = recs["G_seq"].map(lambda seq, p=project: site_combo_from_sequence(seq, p, site_sets[p]))
    return recs


def _study_haplotype_records_with_branches(root: Path, site_sets: dict[str, list[dict[str, object]]] | None = None) -> pd.DataFrame:
    combos, assignments, _ = get_sample_branch_combinations(root)
    hap = load_haplotype_records(root)
    meta = load_metadata(root)
    site_sets = site_sets or derive_linked_site_sets(root)
    valid = assignments.dropna(subset=["SampleID"]).copy()
    valid = valid[valid["TreeCluster_branch"].ne("Unclustered")].copy()
    out = valid.merge(hap[["SequenceName", "G_seq", "G_length", "Frequency"]], on="SequenceName", how="left")
    out = out.merge(meta[["project", "SampleID", "clade", "year_month", "geo_stratum"]], on=["project", "SampleID"], how="left")
    out = out.merge(combos[["project", "SampleID", "TreeCluster_combo"]], on=["project", "SampleID"], how="left")
    for project, cfg in PROJECTS.items():
        out.loc[out["project"].eq(project), f"{cfg['short']}_site_combo"] = out.loc[out["project"].eq(project), "G_seq"].map(lambda seq, p=project: site_combo_from_sequence(seq, p, site_sets[p]))
    out["record_type"] = "study_haplotype"
    targets = select_target_branch_combos(root).set_index("project")
    out["is_focal_haplotype"] = out.apply(lambda r: r["TreeCluster_branch"] in targets.loc[r["project"], "target_branches"] and r["TreeCluster_combo"] == targets.loc[r["project"], "target_combo"], axis=1)
    return out


def _branch_barcode_modes(root: Path, study: pd.DataFrame) -> dict[str, dict[str, str]]:
    modes = {}
    focal = study[study["is_focal_haplotype"].astype(bool)].copy()
    targets = select_target_branch_combos(root).set_index("project")
    for project, cfg in PROJECTS.items():
        combo_col = f"{cfg['short']}_site_combo"
        for branch in list(targets.loc[project, "target_branches"]):
            sub = focal[(focal["project"].eq(project)) & (focal["TreeCluster_branch"].eq(branch))]
            mode = sub[combo_col].dropna().astype(str).mode()
            modes[branch] = {"project": project, "dataset": cfg["label"], "combo_col": combo_col, "expected_combo": mode.iloc[0] if len(mode) else "", "expected_barcode": f"{cfg['short']}_{branch}_like"}
    return modes


def global_consensus_context(root: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    site_sets = derive_linked_site_sets(root)
    study = _study_haplotype_records_with_branches(root, site_sets)
    global_df = _global_records_from_fasta(root, site_sets)
    branch_config = _branch_barcode_modes(root, study)

    threshold_nt = 5
    min_comparable_sites = 900
    global_df = global_df[global_df["G_seq"].astype(str).str.len().eq(966)].reset_index(drop=True)
    global_seqs = global_df["G_seq"].tolist()
    global_bytes = np.array([list(seq.encode("ascii")) for seq in global_seqs], dtype="S1")
    acgt = np.array(list(b"ACGT"), dtype="S1")
    global_valid = np.isin(global_bytes, acgt)

    focal = study[study["is_focal_haplotype"].astype(bool)].copy()
    focal = focal[focal["TreeCluster_branch"].isin(branch_config)].copy()
    rows = []
    for branch, cfg in branch_config.items():
        sub = focal[focal["TreeCluster_branch"].eq(branch)].copy()
        same_mask = global_df[cfg["combo_col"]].eq(cfg["expected_combo"]).to_numpy()
        for _, h in sub.iterrows():
            seq = str(h["G_seq"]).upper()
            if len(seq) != 966:
                continue
            hap_bytes = np.array(list(seq.encode("ascii")), dtype="S1")
            hap_valid = np.isin(hap_bytes, acgt)
            comparable = global_valid & hap_valid
            denom = comparable.sum(axis=1)
            dist = ((global_bytes != hap_bytes) & comparable).sum(axis=1).astype(float)
            dist[denom < min_comparable_sites] = np.nan
            any_within = dist <= threshold_nt
            same_within = any_within & same_mask
            min_any = float(np.nanmin(dist)) if np.isfinite(dist).any() else np.nan
            min_same = float(np.nanmin(dist[same_mask])) if same_mask.any() and np.isfinite(dist[same_mask]).any() else np.nan
            rows.append({
                "dataset": cfg["dataset"],
                "project": cfg["project"],
                "SampleID": h["SampleID"],
                "SequenceName": h["SequenceName"],
                "hap_id": h.get("hap_id", h.get("haplotype_id", np.nan)),
                "Frequency": h.get("Frequency", np.nan),
                "branch": branch,
                "expected_barcode": cfg["expected_barcode"],
                "expected_site_combo": cfg["expected_combo"],
                "threshold_nt": threshold_nt,
                "threshold_subs_per_site": threshold_nt / 966,
                "min_any_global_distance_nt": min_any,
                "min_same_barcode_global_distance_nt": min_same,
                "global_records_within_threshold_any_barcode": int(np.nansum(any_within)),
                "global_records_within_threshold_same_barcode": int(np.nansum(same_within)),
                "has_any_global_within_threshold": bool(np.nansum(any_within) > 0),
                "has_same_barcode_global_within_threshold": bool(np.nansum(same_within) > 0),
            })

    per_haplotype = pd.DataFrame(rows).sort_values(["dataset", "branch", "SampleID", "SequenceName"])
    sample_branch = (
        per_haplotype.groupby(["dataset", "project", "SampleID", "branch", "expected_barcode", "expected_site_combo", "threshold_nt"], as_index=False)
        .agg(
            n_haplotypes_in_sample_branch=("SequenceName", "count"),
            has_any_global_within_threshold=("has_any_global_within_threshold", "any"),
            has_same_barcode_global_within_threshold=("has_same_barcode_global_within_threshold", "any"),
            min_any_global_distance_nt=("min_any_global_distance_nt", "min"),
            min_same_barcode_global_distance_nt=("min_same_barcode_global_distance_nt", "min"),
            global_records_within_threshold_any_barcode=("global_records_within_threshold_any_barcode", "max"),
            global_records_within_threshold_same_barcode=("global_records_within_threshold_same_barcode", "max"),
        )
        .sort_values(["dataset", "branch", "SampleID"])
    )
    summary = (
        sample_branch.groupby(["dataset", "project", "branch", "expected_barcode", "expected_site_combo", "threshold_nt"], as_index=False)
        .agg(
            n_focal_sample_branches=("SampleID", "nunique"),
            n_focal_with_any_global_within_threshold=("has_any_global_within_threshold", "sum"),
            n_focal_with_same_barcode_global_within_threshold=("has_same_barcode_global_within_threshold", "sum"),
            total_global_records_any_barcode_within_threshold=("global_records_within_threshold_any_barcode", "sum"),
            total_global_records_same_barcode_within_threshold=("global_records_within_threshold_same_barcode", "sum"),
            median_min_any_global_distance_nt=("min_any_global_distance_nt", "median"),
            median_min_same_barcode_global_distance_nt=("min_same_barcode_global_distance_nt", "median"),
        )
        .sort_values(["dataset", "branch"])
    )

    barcode_counts = []
    for branch, cfg in branch_config.items():
        sub = global_df.groupby(cfg["combo_col"], as_index=False).size().rename(columns={cfg["combo_col"]: "site_combo", "size": "global_records"})
        sub["branch"] = branch
        sub["barcode_class"] = np.where(sub["site_combo"].eq(cfg["expected_combo"]), cfg["expected_barcode"], "other_complete")
        barcode_counts.append(sub)
    barcode_counts = pd.concat(barcode_counts, ignore_index=True) if barcode_counts else pd.DataFrame()

    write_table(study, out_dir, "study_haplotype_G_records_with_branches_rebuilt.csv")
    write_table(global_df[["accession", "G_seq", "G_length", "G_ambiguous_fraction", "AU_site_combo", "US_site_combo"]], out_dir, "global_G_records_from_fasta.csv")
    write_table(per_haplotype, out_dir, "focal_haplotype_global_context.csv")
    write_table(sample_branch, out_dir, "focal_sample_branch_global_context.csv")
    write_table(summary, out_dir, "global_consensus_background_branch_summary.csv")
    write_table(barcode_counts, out_dir, "global_consensus_barcode_class_counts.csv")
    return sample_branch, summary, barcode_counts, study, global_df

def draw_global_context(summary: pd.DataFrame, fig_dir: Path) -> tuple[plt.Figure, dict[str, str]]:
    setup_style(9.2)
    plot = summary.copy()
    plot["project_order"] = plot["project"].map({p: i for i, p in enumerate(PROJECT_ORDER)})
    plot["branch_order"] = plot["branch"].map(branch_sort_key)
    plot = plot.sort_values(["project_order", "branch_order", "branch"]).reset_index(drop=True)
    n_rows = max(len(plot), 1)
    fig_h = max(3.45, 1.05 + 0.53 * n_rows)
    fig, ax = plt.subplots(figsize=(4.15, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.00, 1.02, "E", transform=ax.transAxes, ha="left", va="bottom", fontsize=14.0, fontweight="bold")
    ax.text(0.095, 1.025, "Same-state recovery in global consensus", transform=ax.transAxes, ha="left", va="bottom", fontsize=10.4, fontweight="bold")
    ax.text(0.06, 0.88, "Branch", ha="left", va="center", fontsize=8.3, color="#666666", fontweight="bold")
    ax.text(0.88, 0.88, "Recovered", ha="right", va="center", fontsize=8.3, color="#666666", fontweight="bold")
    ax.plot([0.06, 0.90], [0.835, 0.835], color="#D9DEE2", lw=0.8)

    top_y = 0.72
    bottom_y = 0.22 if n_rows > 1 else top_y
    ys = np.linspace(top_y, bottom_y, n_rows)
    last_project = None
    for y, rec in zip(ys, plot.itertuples(index=False)):
        if rec.project != last_project:
            ax.text(0.06, min(y + 0.065, 0.80), rec.dataset, ha="left", va="center", fontsize=8.2, color="#777777")
            last_project = rec.project
        recovered = int(rec.n_focal_with_same_barcode_global_within_threshold)
        total = int(rec.n_focal_sample_branches)
        color = BRANCH_COLORS.get(rec.branch, PROJECTS.get(rec.project, {}).get("color", "#555555"))
        ax.scatter([0.075], [y], s=34, color=color, edgecolor="white", lw=0.4, zorder=3)
        ax.text(0.115, y, str(rec.branch), ha="left", va="center", fontsize=8.8, color="#333333")
        ax.text(0.88, y, f"{recovered}/{total}", ha="right", va="center", fontsize=9.4, fontweight="bold", color=color if recovered else "#777777")
        ax.plot([0.06, 0.90], [y - 0.070, y - 0.070], color="#EDF0F2", lw=0.6)
    ax.text(0.06, 0.075, "Same-state criterion: <=5 nt across the 966-nt G gene", ha="left", va="center", fontsize=7.2, color="#69757C")
    fig.tight_layout(pad=0.5)
    paths = save_pub_figure(fig, fig_dir, "Fig4E_global_consensus_recovery")
    return fig, paths

def de_novo_plausibility(root: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    core, _, _ = branch_defining_sites(root, out_dir / "branch_defining_sites")
    facts = []
    for (project, combo), group in core.groupby(["project", "TreeCluster_combo"], sort=False):
        facts.append({
            "project": project,
            "project_label": PROJECTS[project]["label"],
            "target_combo": combo,
            "k_linked_sites": int(group["nt_position_genome"].nunique()),
            "observed_target_samples": int(group["samples"].median()) if "samples" in group.columns else np.nan,
            "site_pattern": "; ".join(f"{int(r.nt_position_genome)} {r.ref_nt}>{r.target_nt}" for r in group.sort_values("nt_position_genome").itertuples(index=False)),
        })
    facts = pd.DataFrame(facts)
    mu_values = [1e-6, 1e-5, 1e-4]
    rounds = [3, 5, 10, 14]
    rows = []
    for rec in facts.itertuples(index=False):
        for mu in mu_values:
            for c in rounds:
                site_hit = 1.0 - math.exp(-mu * c)
                rows.append({
                    "project": rec.project,
                    "project_label": rec.project_label,
                    "target_combo": rec.target_combo,
                    "k_linked_sites": rec.k_linked_sites,
                    "mu": mu,
                    "serial_replication_rounds": c,
                    "p_specific_linked_pattern_upper": site_hit ** int(rec.k_linked_sites),
                })
    scenarios = pd.DataFrame(rows)
    write_table(facts, out_dir, "de_novo_linked_site_facts.csv")
    write_table(scenarios, out_dir, "de_novo_plausibility_sensitivity_grid.csv")
    return facts, scenarios

def draw_de_novo(scenarios: pd.DataFrame, fig_dir: Path) -> tuple[plt.Figure, dict[str, str]]:
    setup_style(9.8)
    mu_values = [1e-6, 1e-5, 1e-4]
    mu_panel = {1e-6: "A", 1e-5: "B", 1e-4: "C"}
    mu_label = {1e-6: r"$\mu=10^{-6}$", 1e-5: r"$\mu=10^{-5}$", 1e-4: r"$\mu=10^{-4}$"}
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.25), sharey=True, gridspec_kw={"wspace": 0.32})
    short = {"Australia": "AU", "United States": "US"}
    for ax, mu in zip(axes, mu_values):
        sub = scenarios[scenarios["mu"].eq(mu)]
        for project in PROJECT_ORDER:
            label = PROJECTS[project]["label"]
            group = sub[sub["project"].eq(project)].sort_values("serial_replication_rounds")
            color = PROJECTS[project]["color"]
            ax.plot(group["serial_replication_rounds"], group["p_specific_linked_pattern_upper"], marker="o", linewidth=2.05, markersize=5.0, color=color, markeredgecolor="white", markeredgewidth=0.6)
            last = group.iloc[-1]
            y_factor = 1.42 if project == "PRJNA1130896" else 0.65
            ax.text(last["serial_replication_rounds"] + 0.28, last["p_specific_linked_pattern_upper"] * y_factor, f"{short[label]}: {int(last['k_linked_sites'])} linked sites", color=color, va="center", ha="left", fontsize=9.0, fontweight="bold")
        ax.set_yscale("log")
        ax.set_ylim(1e-30, 1e-5)
        ax.set_xlim(2.4, 16.9)
        ax.set_xticks([3, 5, 10, 14])
        ax.set_xlabel("Assumed replication rounds (C)")
        ax.grid(axis="y", color="#E8ECEF", linewidth=0.85, which="both")
        ax.axhline(1e-6, color="#AEB8BF", linestyle=(0, (4, 3)), linewidth=1.05)
        clean_axis(ax)
        panel_label(ax, mu_panel[mu], x=-0.12, y=1.04)
        ax.set_title(mu_label[mu], loc="center", pad=8)
    axes[0].set_ylabel("Upper-bound probability of specified\nlinked-site pattern")
    fig.tight_layout()
    paths = save_pub_figure(fig, fig_dir, "FigS_de_novo_specific_linked_pattern_upper_bound")
    return fig, paths
