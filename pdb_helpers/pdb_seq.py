#!/usr/bin/env python3

# Copyright 2007, Michael J. Harms
# This program is distributed under General Public License v. 3.  See the file
# COPYING for a copy of the license.  

__description__ = \
"""
pdb_seq.py

Reads sequence from pdb and returns in single amino acid code FASTA format.
First tries to read SEQRES entries; if this fails, uses pdb coordinates.
"""
__author__ = "Michael J. Harms"
__date__ = "080123"

import argparse
import os
import sys
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
    "MSE": "M",
}

class PdbSeqError(Exception):
    """
    Error class for this module.
    """

    pass


def pdbSeq(pdb,use_atoms=False):
    """
    Parse the SEQRES entries in a pdb file.  If this fails, use the ATOM 
    entries.  Return dictionary of sequences keyed to chain and type of
    sequence used.
    """

    # Try using SEQRES
    seq = [l for l in pdb if l[0:6] == "SEQRES"]
    if len(seq) != 0 and not use_atoms:
        seq_type = "SEQRES"
        chain_dict = {l[11]: [] for l in seq}
        for c in chain_dict.keys():
            chain_seq = [l[19:70].split() for l in seq if l[11] == c]
            for x in chain_seq:
                chain_dict[c].extend(x)

    # Otherwise, use ATOM
    else:

        seq_type = "ATOM  "

        # Check to see if there are multiple models.  If there are, only look
        # at the first model.
        models = [i for i, l in enumerate(pdb) if l.startswith("MODEL")]
        if len(models) > 1:
            pdb = pdb[models[0]:models[1]]     

        # Grab all CA from ATOM entries, as well as MSE from HETATM
        atoms = []
        for l in pdb:
            if l[0:6] == "ATOM  " and l[13:16] == "CA ":
                    
                # Check to see if this is a second conformation of the previous
                # atom
                if len(atoms) != 0:
                    if atoms[-1][17:26] == l[17:26]:
                        continue

                atoms.append(l)
            elif l[0:6] == "HETATM" and l[13:16] == "CA " and l[17:20] == "MSE":

                # Check to see if this is a second conformation of the previous
                # atom
                if len(atoms) != 0:
                    if atoms[-1][17:26] == l[17:26]:
                        continue

                atoms.append(l)

        chain_dict = {l[21]: [] for l in atoms}
        for c in chain_dict.keys():
            chain_dict[c] = [l[17:20] for l in atoms if l[21] == c]

    return chain_dict, seq_type 


def convertModifiedAA(chain_dict,pdb):
    """
    Convert modified amino acids to their normal counterparts.
    """

    # See if there are any non-standard amino acids in the pdb file.  If there
    # are not, return
    modres = [l for l in pdb if l[0:6] == "MODRES"]
    if len(modres) == 0:
        return chain_dict

    # Create list of modified residues
    mod_dict = {l[12:15]: l[24:27] for l in modres}

    # Replace all entries in chain_dict with their unmodified counterparts.
    for c in chain_dict.keys():
        for i, a in enumerate(chain_dict[c]):
            if a in mod_dict:
                chain_dict[c][i] = mod_dict[a]

    return chain_dict


def pdbSeq2Fasta(pdb,pdb_id="",chain="all",use_atoms=False):
    """
    Extract sequence from pdb file and write out in FASTA format.
    """

    # Grab sequences
    chain_dict, seq_type = pdbSeq(pdb,use_atoms)

    # Convert modified amino acids to their natural counterparts
    chain_dict = convertModifiedAA(chain_dict,pdb)

    # Determine which chains are being written out
    if chain == "all":
        chains_to_write = sorted(chain_dict.keys())
    else:
        if chain in chain_dict:
            chains_to_write = [chain]
        else:
            err = "Chain \"%s\" not in pdb!" % chain
            raise PdbSeqError(err)

    # Convert sequences to 1-letter format and join strings
    for c in chains_to_write:
        for aa_index, aa in enumerate(chain_dict[c]):
            try:
                chain_dict[c][aa_index] = AA3_TO_AA1[aa]
            except KeyError:
                chain_dict[c][aa_index] = "X"

    out = []
    for c in chains_to_write:
        out.append("> %s%s_%s\n" % (pdb_id,c,seq_type))

        # Write output in lines 80 characters long
        seq_length = len(chain_dict[c])
        num_lines = seq_length // 80
        
        for i in range(num_lines+1):
            out.append("".join([aa for aa in chain_dict[c][80*i:80*(i+1)]]))
            out.append("\n")
        out.append("".join([aa for aa in chain_dict[c][80*(i+1):]]))
        out.append("\n")


    return "".join(out)
      
            
def parse_args():
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument(
        "pdb_inputs",
        nargs="+",
        help="Paths to PDB files or directories containing .pdb files",
    )
    parser.add_argument(
        "-c",
        "--chain",
        default="all",
        help="Chain identifier to select (default: all)",
    )
    parser.add_argument(
        "-a",
        "--atomseq",
        action="store_true",
        default=False,
        help="Use ATOM sequence when SEQRES is unavailable",
    )
    return parser.parse_args()


def resolve_pdb_files(paths):
    resolved = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            dir_files = sorted(
                f for f in path.iterdir() if f.is_file() and f.suffix.lower() == ".pdb"
            )
            if not dir_files:
                print(f"No .pdb files found in directory: {path}", file=sys.stderr)
            resolved.extend(dir_files)
        elif path.is_file():
            resolved.append(path)
        else:
            print(f"Path does not exist: {path}", file=sys.stderr)
    return resolved


def main():
    """Function to execute if called from command line."""

    args = parse_args()

    pdb_paths = resolve_pdb_files(args.pdb_inputs)
    if not pdb_paths:
        print("No PDB files to process", file=sys.stderr)
        sys.exit(1)

    for pdb_path in pdb_paths:
        pdb_id = pdb_path.stem
        with pdb_path.open(encoding="utf-8") as f:
            pdb = f.readlines()

        seq = pdbSeq2Fasta(pdb, pdb_id, args.chain, args.atomseq)

        print(seq)
    


if __name__ == "__main__":
    main()
