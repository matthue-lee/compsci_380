#!/usr/bin/env python3

"""Aggregate experimental and structural features into a master table."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


HYDROPHOBIC = set("AVLIMFWYGP")
POLAR = set("STNQYCH")
POSITIVE = set("KRH")
NEGATIVE = set("DE")
AMINO_ACIDS = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]
CHARGED_RESIDUES = set("DEHKR")
PKA_POSITIVE = {"K": 10.5, "R": 12.4, "H": 6.0}
PKA_NEGATIVE = {"D": 3.9, "E": 4.3, "C": 8.3, "Y": 10.1}
PKA_NTERM = 8.0
PKA_CTERM = 3.6
KYTE_DOOLITTLE = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}
PHYSIOLOGICAL_PH = 7.0
AMINO_ACID_FRACTION_COLUMNS = [f"sequence_frac_{aa}" for aa in AMINO_ACIDS]
HELIX_CHARS = set("HGI")
STRAND_CHARS = set("EB")


def normalize_pdb_id(value: str | None) -> str:
    if not value:
        return ""
    cleaned = value.strip().lower()
    if cleaned.endswith(".pdb"):
        cleaned = cleaned[:-4]
    cleaned = cleaned.replace("*", "")
    return cleaned


def coerce_number(value: str | None) -> int | float | str:
    if value is None:
        return ""
    text = value.strip()
    if not text:
        return ""
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def read_dataset(path: Path) -> Tuple[List[Dict[str, object]], List[str]]:
    rows: List[Dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        for raw_row in reader:
            row = {key: raw_row.get(key, "") for key in fieldnames}
            original_id = row.get("PDB", "")
            normalized = normalize_pdb_id(original_id)
            row["source_PDB"] = original_id
            row["PDB"] = normalized
            for key in fieldnames:
                if key in {"PDB", "Pattern", "CATH"}:
                    continue
                row[key] = coerce_number(row.get(key, ""))
            rows.append(row)
    ordered_fields = ["PDB", "source_PDB"] + [field for field in fieldnames if field != "PDB"]
    return rows, ordered_fields


def read_fasta_sequences(fasta_dir: Path) -> Dict[str, Dict[str, object]]:
    features: Dict[str, Dict[str, object]] = {}
    if not fasta_dir.exists():
        return features
    for fasta_path in sorted(fasta_dir.glob("*.fa")):
        seq_lines: List[str] = []
        with fasta_path.open(encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith(">"):
                    continue
                seq_lines.append(line.strip())
        raw_sequence = "".join(seq_lines).upper()
        chains = [segment for segment in raw_sequence.split("!") if segment]
        aa_sequence = "".join(chains)
        seq_length = len(aa_sequence)
        counts = Counter(aa_sequence)
        hydro_mean, hydro_var = hydrophobicity_moments(counts, seq_length)
        net_charge_value = net_charge(counts, seq_length)
        pi_value = estimate_isoelectric_point(counts, seq_length)
        composition = {
            f"sequence_frac_{aa}": fraction(counts, seq_length, {aa}) for aa in AMINO_ACIDS
        }
        entry = {
            "sequence_raw_length": len(raw_sequence.replace("\n", "")),
            "sequence_chain_count": max(len(chains), 1 if raw_sequence else 0),
            "sequence_length": seq_length,
            "sequence_frac_hydrophobic": fraction(counts, seq_length, HYDROPHOBIC),
            "sequence_frac_polar": fraction(counts, seq_length, POLAR),
            "sequence_frac_positive": fraction(counts, seq_length, POSITIVE),
            "sequence_frac_negative": fraction(counts, seq_length, NEGATIVE),
            "sequence_frac_glycine": fraction(counts, seq_length, {"G"}),
            "sequence_frac_proline": fraction(counts, seq_length, {"P"}),
            "sequence_frac_low_complexity": fraction(counts, seq_length, {"G", "P"}),
            "sequence_fraction_charged": fraction(counts, seq_length, CHARGED_RESIDUES),
            "sequence_net_charge": net_charge_value,
            "sequence_estimated_pI": pi_value,
            "sequence_hydrophobicity_mean": hydro_mean,
            "sequence_hydrophobicity_variance": hydro_var,
        }
        entry.update(composition)
        features[normalize_pdb_id(fasta_path.stem)] = entry
    return features


def fraction(counts: Counter, length: int, letters: Iterable[str]) -> float:
    if not length:
        return 0.0
    return sum(counts.get(letter, 0) for letter in letters) / length


def net_charge(counts: Counter, length: int, ph: float = PHYSIOLOGICAL_PH) -> float:
    if length == 0:
        return 0.0
    positive = sum(
        counts.get(residue, 0) * (1.0 / (1 + 10 ** (ph - pka))) for residue, pka in PKA_POSITIVE.items()
    )
    negative = -sum(
        counts.get(residue, 0) * (1.0 / (1 + 10 ** (pka - ph))) for residue, pka in PKA_NEGATIVE.items()
    )
    positive += 1.0 / (1 + 10 ** (ph - PKA_NTERM))
    negative -= 1.0 / (1 + 10 ** (PKA_CTERM - ph))
    return positive + negative


def estimate_isoelectric_point(counts: Counter, length: int) -> float:
    if length == 0:
        return 0.0
    low, high = 0.0, 14.0
    for _ in range(40):
        mid = (low + high) / 2
        charge = net_charge(counts, length, mid)
        if charge > 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def hydrophobicity_moments(counts: Counter, length: int) -> Tuple[float, float]:
    if length == 0:
        return 0.0, 0.0
    total = sum(KYTE_DOOLITTLE.get(residue, 0.0) * count for residue, count in counts.items())
    total_sq = sum(
        (KYTE_DOOLITTLE.get(residue, 0.0) ** 2) * count for residue, count in counts.items()
    )
    mean = total / length
    variance = max(total_sq / length - mean**2, 0.0)
    return mean, variance


def segment_lengths(sequence: str, target_chars: Iterable[str]) -> List[int]:
    lengths: List[int] = []
    current = 0
    target = set(target_chars)
    for char in sequence:
        if char in target:
            current += 1
        else:
            if current:
                lengths.append(current)
                current = 0
    if current:
        lengths.append(current)
    return lengths


def summarize_secondary_sequence(sequence: str) -> Dict[str, object]:
    sequence = sequence.strip()
    length = len(sequence)
    counts = Counter(sequence)
    helix_residues = sum(counts.get(char, 0) for char in HELIX_CHARS)
    strand_residues = sum(counts.get(char, 0) for char in STRAND_CHARS)
    coil_residues = max(length - helix_residues - strand_residues, 0)
    helix_segments = segment_lengths(sequence, HELIX_CHARS)
    strand_segments = segment_lengths(sequence, STRAND_CHARS)
    transitions = sum(1 for idx in range(1, length) if sequence[idx] != sequence[idx - 1])
    features = {
        "ss_fraction_helix": helix_residues / length if length else 0.0,
        "ss_fraction_strand": strand_residues / length if length else 0.0,
        "ss_fraction_coil": coil_residues / length if length else 0.0,
        "ss_helix_segment_count": len(helix_segments),
        "ss_strand_segment_count": len(strand_segments),
        "ss_helix_segment_mean_length": (sum(helix_segments) / len(helix_segments)) if helix_segments else 0.0,
        "ss_strand_segment_mean_length": (sum(strand_segments) / len(strand_segments)) if strand_segments else 0.0,
        "ss_longest_helix_length": max(helix_segments) if helix_segments else 0,
        "ss_longest_strand_length": max(strand_segments) if strand_segments else 0,
        "ss_transitions_per_residue": transitions / length if length else 0.0,
    }
    return features


def contact_statistics(
    coords: List[Tuple[float, float, float]], cutoff: float = 8.0, buried_threshold: int = 12
) -> Tuple[float, float, float]:
    count = len(coords)
    if count == 0:
        return 0.0, 0.0, 0.0
    cutoff_sq = cutoff * cutoff
    contact_counts = [0] * count
    for i in range(count):
        xi, yi, zi = coords[i]
        for j in range(i + 1, count):
            xj, yj, zj = coords[j]
            dx = xi - xj
            dy = yi - yj
            dz = zi - zj
            dist_sq = dx * dx + dy * dy + dz * dz
            if dist_sq <= cutoff_sq:
                contact_counts[i] += 1
                contact_counts[j] += 1
    mean_contact = sum(contact_counts) / count
    mean_relative_sasa = sum(1.0 / (1.0 + value) for value in contact_counts) / count
    fraction_buried = sum(1 for value in contact_counts if value >= buried_threshold) / count
    return mean_contact, mean_relative_sasa, fraction_buried


def radius_of_gyration(coords: List[Tuple[float, float, float]]) -> float:
    count = len(coords)
    if count == 0:
        return 0.0
    cx = sum(point[0] for point in coords) / count
    cy = sum(point[1] for point in coords) / count
    cz = sum(point[2] for point in coords) / count
    total_sq = 0.0
    for x, y, z in coords:
        dx = x - cx
        dy = y - cy
        dz = z - cz
        total_sq += dx * dx + dy * dy + dz * dz
    return math.sqrt(total_sq / count)


def read_dssp_summary(path: Path) -> Dict[str, Dict[str, object]]:
    summary: Dict[str, Dict[str, object]] = {}
    if not path.exists():
        return summary
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 12:
                continue
            base_id = normalize_pdb_id(parts[0])
            summary[base_id] = {
                "ss_total_residues": int(parts[1]),
                "ss_H": int(parts[2]),
                "ss_B": int(parts[3]),
                "ss_E": int(parts[4]),
                "ss_G": int(parts[5]),
                "ss_I": int(parts[6]),
                "ss_T": int(parts[7]),
                "ss_S": int(parts[8]),
                "ss_frac_helix": float(parts[9]),
                "ss_frac_beta": float(parts[10]),
                "ss_frac_secondary": float(parts[11]),
            }
    return summary


def read_secondary_structure_sequences(secondary_dir: Path) -> Dict[str, Dict[str, object]]:
    features: Dict[str, Dict[str, object]] = {}
    if not secondary_dir.exists():
        return features
    for ss_path in sorted(secondary_dir.glob("*.ss")):
        lines: List[str] = []
        with ss_path.open(encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith(">"):
                    continue
                lines.append(line.strip())
        sequence = "".join(lines).upper()
        if not sequence:
            continue
        features[normalize_pdb_id(ss_path.stem)] = summarize_secondary_sequence(sequence)
    return features


def parse_structure_features(pdb_dir: Path) -> Dict[str, Dict[str, object]]:
    features: Dict[str, Dict[str, object]] = {}
    if not pdb_dir.exists():
        return features
    for pdb_file in sorted(pdb_dir.glob("*.pdb")):
        chains = set()
        residues = set()
        chain_ranges: Dict[str, Tuple[int, int]] = {}
        atoms = hetatm = ssbonds = 0
        model_count = 0
        ca_coords: List[Tuple[float, float, float]] = []
        seen_ca = set()
        with pdb_file.open(encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                record = line[:6].strip().upper()
                if record == "MODEL":
                    model_count += 1
                if record == "ATOM":
                    atoms += 1
                    chain = (line[21].strip() or "_") if len(line) >= 22 else "_"
                    resseq = line[22:26].strip() if len(line) >= 26 else ""
                    icode = line[26].strip() if len(line) >= 27 else ""
                    residues.add((chain, resseq, icode))
                    chains.add(chain)
                    if resseq:
                        try:
                            resnum = int(resseq)
                        except ValueError:
                            resnum = None
                        if resnum is not None:
                            low, high = chain_ranges.get(chain, (resnum, resnum))
                            chain_ranges[chain] = (min(low, resnum), max(high, resnum))
                    atom_name = line[12:16].strip()
                    altloc = line[16].strip() if len(line) >= 17 else ""
                    if atom_name == "CA" and (not altloc or altloc == "A"):
                        try:
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                        except ValueError:
                            continue
                        key = (chain, resseq, icode)
                        if key in seen_ca:
                            continue
                        seen_ca.add(key)
                        ca_coords.append((x, y, z))
                elif record == "HETATM":
                    hetatm += 1
                elif record == "SSBOND":
                    ssbonds += 1
        if model_count == 0:
            model_count = 1
        contact_density, mean_relative_sasa, fraction_buried = contact_statistics(ca_coords)
        gyration = radius_of_gyration(ca_coords)
        chain_list = "|".join(sorted(chains)) if chains else ""
        domain_ranges = "|".join(
            f"{chain}:{bounds[0]}-{bounds[1]}" for chain, bounds in sorted(chain_ranges.items())
        )
        features[normalize_pdb_id(pdb_file.stem)] = {
            "chain_list": chain_list,
            "num_chains": len(chains),
            "num_residues": len(residues),
            "num_atoms": atoms,
            "num_hetatm": hetatm,
            "num_models": model_count,
            "num_disulfide_bonds": ssbonds,
            "domain_ranges": domain_ranges,
            "structure_contact_density": contact_density,
            "structure_radius_of_gyration": gyration,
            "structure_mean_relative_sasa": mean_relative_sasa,
            "structure_fraction_buried": fraction_buried,
        }
    return features


def merge_rows(
    dataset_rows: List[Dict[str, object]],
    sequence_features: Dict[str, Dict[str, object]],
    secondary_features: Dict[str, Dict[str, object]],
    structure_features: Dict[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    merged: List[Dict[str, object]] = []
    for row in dataset_rows:
        pdb_id = row.get("PDB", "")
        combined = row.copy()
        for source in (structure_features, sequence_features, secondary_features):
            if pdb_id in source:
                combined.update(source[pdb_id])
        beta_fraction = combined.get("ss_fraction_strand")
        contact_density = combined.get("structure_contact_density")
        if beta_fraction is not None and contact_density is not None:
            try:
                combined["structure_beta_topology_score"] = float(beta_fraction) * float(contact_density)
            except (TypeError, ValueError):
                combined.setdefault("structure_beta_topology_score", 0.0)
        else:
            combined.setdefault("structure_beta_topology_score", 0.0)
        merged.append(combined)
    return merged


def write_csv(rows: List[Dict[str, object]], columns: List[str], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset.csv", help="Path to the curated dataset CSV")
    parser.add_argument(
        "--work-dir",
        default="pdb_helpers",
        help="Directory that contains PDB_Files, Fasta_Files, and Analysis outputs",
    )
    parser.add_argument(
        "--pdb-dir",
        default=None,
        help="Override directory containing raw PDB files (defaults to WORK_DIR/PDB_Files)",
    )
    parser.add_argument(
        "--fasta-dir",
        default=None,
        help="Override directory with per-structure FASTA files (defaults to WORK_DIR/Fasta_Files)",
    )
    parser.add_argument(
        "--dssp-summary",
        default=None,
        help="Path to Analysis/DSSP/DSSP_outputs.txt (defaults relative to WORK_DIR)",
    )
    parser.add_argument(
        "--secondary-dir",
        default=None,
        help="Override directory with DSSP secondary structure FASTA files (defaults to WORK_DIR/DSSP/SecondaryStructure)",
    )
    parser.add_argument(
        "--output",
        default="master_table.csv",
        help="Destination CSV file for the consolidated table",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    work_dir = Path(args.work_dir)
    pdb_dir = Path(args.pdb_dir) if args.pdb_dir else work_dir / "PDB_Files"
    fasta_dir = Path(args.fasta_dir) if args.fasta_dir else work_dir / "Fasta_Files"
    dssp_path = Path(args.dssp_summary) if args.dssp_summary else work_dir / "Analysis" / "DSSP" / "DSSP_outputs.txt"
    secondary_dir = (
        Path(args.secondary_dir)
        if args.secondary_dir
        else work_dir / "DSSP" / "SecondaryStructure"
    )

    dataset_rows, ordered_fields = read_dataset(dataset_path)
    seq_features = read_fasta_sequences(fasta_dir)
    ss_features = read_dssp_summary(dssp_path)
    ss_sequence_features = read_secondary_structure_sequences(secondary_dir)
    for pdb_id, values in ss_sequence_features.items():
        if pdb_id not in ss_features:
            ss_features[pdb_id] = {}
        ss_features[pdb_id].update(values)
    struct_features = parse_structure_features(pdb_dir)

    merged_rows = merge_rows(dataset_rows, seq_features, ss_features, struct_features)

    sequence_columns = [
        "sequence_chain_count",
        "sequence_length",
        "sequence_raw_length",
        "sequence_frac_hydrophobic",
        "sequence_frac_polar",
        "sequence_frac_positive",
        "sequence_frac_negative",
        "sequence_frac_glycine",
        "sequence_frac_proline",
        "sequence_frac_low_complexity",
        "sequence_fraction_charged",
        "sequence_net_charge",
        "sequence_estimated_pI",
        "sequence_hydrophobicity_mean",
        "sequence_hydrophobicity_variance",
    ]
    sequence_columns += AMINO_ACID_FRACTION_COLUMNS
    secondary_columns = [
        "ss_total_residues",
        "ss_H",
        "ss_B",
        "ss_E",
        "ss_G",
        "ss_I",
        "ss_T",
        "ss_S",
        "ss_frac_helix",
        "ss_frac_beta",
        "ss_frac_secondary",
        "ss_fraction_helix",
        "ss_fraction_strand",
        "ss_fraction_coil",
        "ss_helix_segment_count",
        "ss_strand_segment_count",
        "ss_helix_segment_mean_length",
        "ss_strand_segment_mean_length",
        "ss_longest_helix_length",
        "ss_longest_strand_length",
        "ss_transitions_per_residue",
    ]
    structure_columns = [
        "chain_list",
        "domain_ranges",
        "num_chains",
        "num_residues",
        "num_atoms",
        "num_hetatm",
        "num_models",
        "num_disulfide_bonds",
        "structure_contact_density",
        "structure_radius_of_gyration",
        "structure_mean_relative_sasa",
        "structure_fraction_buried",
        "structure_beta_topology_score",
    ]
    all_columns = ordered_fields + sequence_columns + secondary_columns + structure_columns

    output_path = Path(args.output)
    write_csv(merged_rows, all_columns, output_path)

    print(f"Wrote master table with {len(merged_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
