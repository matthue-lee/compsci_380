#!/usr/bin/env python3

import argparse
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

AA3_TO_AA1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "ASX": "B",
    "GLX": "Z",
    "SEC": "U",
    "PYL": "O",
    "MSE": "M",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce supervisor DSSP workflow using mkdssp"
    )
    parser.add_argument(
        "--mkdssp-path",
        default="mkdssp",
        help="Path to mkdssp executable (defaults to mkdssp in PATH)",
    )
    parser.add_argument(
        "--work-dir",
        required=True,
        help="Working directory containing PDB_Files and where DSSP/Fasta outputs go",
    )
    parser.add_argument(
        "--analysis-root",
        default=None,
        help="Root directory that contains Analysis/DSSP/DSSP_outputs.txt (defaults to work dir)",
    )
    parser.add_argument(
        "--pdb-dir",
        default=None,
        help="Override path to PDB files (defaults to WORK_DIR/PDB_Files)",
    )
    return parser.parse_args()


def amino_acid_to_one(code):
    if not code:
        return "X"
    code = code.strip()
    if len(code) == 1:
        return code.upper()
    return AA3_TO_AA1.get(code.upper(), "X")


def ensure_clean_directory(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def gather_pdb_files(pdb_dir):
    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    if not pdb_files:
        print(f"No PDB files found in {pdb_dir}", file=sys.stderr)
    return pdb_files


def run_dssp_on_file(pdb_path, mkdssp_path):
    cmd = [mkdssp_path, "--output-format", "dssp", str(pdb_path)]
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"mkdssp executable not found: {mkdssp_path}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(
            f"mkdssp failed for {pdb_path}: {stderr or exc}"
        ) from exc

    entries = []
    lines = proc.stdout.splitlines()
    data_started = False
    for line in lines:
        if not data_started:
            if "#  RESIDUE" in line:
                data_started = True
            continue
        if not line.strip():
            continue

        aa_char = line[13:14].strip() if len(line) >= 14 else ""
        ss_char = line[16:17].strip() if len(line) >= 17 else ""

        aa = amino_acid_to_one(aa_char)
        ss = ss_char if ss_char else "-"
        entries.append((aa, ss))

    return entries


def write_fasta(path, identifier, sequence):
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"> {identifier}\n")
        handle.write(sequence)
        handle.write("\n")


def compute_summary(ss_sequence, length):
    counts = Counter(ss_sequence)
    H = counts.get("H", 0)
    B = counts.get("B", 0)
    E = counts.get("E", 0)
    G = counts.get("G", 0)
    I = counts.get("I", 0)
    T = counts.get("T", 0)
    S = counts.get("S", 0)

    if length == 0:
        return H, B, E, G, I, T, S, 0.0, 0.0, 0.0

    total_helix = (H + G + I) / length
    total_beta = (B + E) / length
    total_big = (H + G + I + B + E) / length
    return H, B, E, G, I, T, S, total_helix, total_beta, total_big


def main():
    args = parse_args()
    work_dir = Path(args.work_dir).resolve()
    if not work_dir.exists():
        print(f"Work directory not found: {work_dir}", file=sys.stderr)
        sys.exit(1)

    analysis_root = Path(args.analysis_root).resolve() if args.analysis_root else work_dir
    pdb_dir = Path(args.pdb_dir).resolve() if args.pdb_dir else work_dir / "PDB_Files"
    mkdssp_path = str(Path(args.mkdssp_path).expanduser())

    dssp_dir = work_dir / "DSSP"
    secondary_dir = dssp_dir / "SecondaryStructure"
    fasta_dir = work_dir / "Fasta_Files"
    analysis_dir = analysis_root / "Analysis" / "DSSP"

    ensure_clean_directory(dssp_dir)
    secondary_dir.mkdir(parents=True, exist_ok=True)
    ensure_clean_directory(fasta_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    pdb_files = gather_pdb_files(pdb_dir)
    if not pdb_files:
        sys.exit(1)

    summary_path = analysis_dir / "DSSP_outputs.txt"
    summary_handle = summary_path.open("a", encoding="utf-8")

    processed = 0
    for pdb_file in pdb_files:
        identifier = pdb_file.name.split(".")[0]
        try:
            entries = run_dssp_on_file(pdb_file, mkdssp_path)
        except Exception as exc:
            print(f"Failed to process {pdb_file}: {exc}", file=sys.stderr)
            continue

        if not entries:
            print(f"No DSSP entries for {pdb_file}", file=sys.stderr)
            continue

        aa_sequence = "".join(aa for aa, _ in entries)
        ss_sequence = "".join(ss for _, ss in entries)
        length = len(entries)

        write_fasta(fasta_dir / f"{identifier}.fa", identifier, aa_sequence)
        write_fasta(secondary_dir / f"{identifier}.ss", identifier, ss_sequence)

        stats = compute_summary(ss_sequence, length)
        H, B, E, G, I, T, S, helix_pct, beta_pct, big_pct = stats
        summary_handle.write(
            f"{identifier}\t{length}\t{H}\t{B}\t{E}\t{G}\t{I}\t{T}\t{S}\t"
            f"{helix_pct:.6f}\t{beta_pct:.6f}\t{big_pct:.6f}\n"
        )
        processed += 1

    summary_handle.close()

    if processed == 0:
        print("No structures were processed successfully", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
