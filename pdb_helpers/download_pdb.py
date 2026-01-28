#!/usr/bin/env python3

import os
import sys
from urllib import error, request

from tqdm import tqdm

LIST_FILE = "pdb.list.to.download"


def main():
    try:
        os.system(f"wc -l {LIST_FILE}")
        with open(LIST_FILE, encoding="utf-8") as f:
            for line in tqdm(f, desc="Downloading PDB files"):
                line = line.strip()
                if not line:
                    continue

                token = line.split()[0]
                pdb_id = token.split(".")[0]
                url = f"https://files.rcsb.org/view/{pdb_id}.pdb"
                req = request.Request(url)

                try:
                    with request.urlopen(req) as resp:
                        data = resp.read()
                except error.URLError as exc:
                    print(f"Failed to download {pdb_id}: {exc}", file=sys.stderr)
                    continue

                with open(f"{pdb_id}.pdb", "wb") as out_file:
                    out_file.write(data)
    except FileNotFoundError:
        print(f"Could not open {LIST_FILE}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
