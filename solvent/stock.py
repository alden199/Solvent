from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from solvent.chem_utils import canonical_parent_smiles


def validate_stock(path: Path) -> dict[str, int]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Stock database does not exist: {path}")
    try:
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
            columns = {row[1]: row for row in conn.execute("PRAGMA table_info(emolecules)")}
            if "canonical_parent" not in columns:
                raise ValueError("Stock must contain emolecules(canonical_parent TEXT).")
            if columns["canonical_parent"][2].upper() != "TEXT":
                raise ValueError("emolecules.canonical_parent must have TEXT affinity.")
            count = conn.execute("SELECT count(*) FROM emolecules").fetchone()[0]
            if not count:
                raise ValueError("Stock database is empty.")
            if conn.execute(
                "SELECT 1 FROM emolecules WHERE canonical_parent IS NULL OR canonical_parent = '' LIMIT 1"
            ).fetchone():
                raise ValueError("Stock contains empty canonical_parent values.")
            indexes = conn.execute("PRAGMA index_list(emolecules)").fetchall()
            indexed = False
            for row in indexes:
                name = row[1].replace('"', '""')
                info = conn.execute(f'PRAGMA index_info("{name}")').fetchall()
                indexed |= bool(info and info[0][2] == "canonical_parent" and not row[4])
            if not indexed:
                raise ValueError("Stock needs an index starting with canonical_parent.")
    except sqlite3.DatabaseError as exc:
        raise ValueError(f"Invalid stock database: {exc}") from exc
    return {"molecules": count}


def build_stock(smiles_file: Path, output: Path, *, skip_invalid: bool = False) -> dict[str, int]:
    smiles_file = Path(smiles_file).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace an existing stock database: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".stock-", suffix=".sqlite", dir=output.parent)
    os.close(fd)
    temporary_path = Path(temporary)
    conn = sqlite3.connect(temporary)
    read = invalid = 0
    try:
        conn.execute("CREATE TABLE emolecules (canonical_parent TEXT PRIMARY KEY NOT NULL) WITHOUT ROWID")
        with smiles_file.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                smiles = line.strip()
                if not smiles:
                    continue
                read += 1
                parent = canonical_parent_smiles(smiles) if not any(c.isspace() for c in smiles) else None
                if not parent:
                    invalid += 1
                    if not skip_invalid:
                        raise ValueError(f"Invalid SMILES at line {line_number}; expected one SMILES per line.")
                    continue
                conn.execute("INSERT OR IGNORE INTO emolecules VALUES (?)", (parent,))
                if read % 10000 == 0:
                    conn.commit()
        conn.commit()
        conn.close()
        summary = validate_stock(temporary_path)
        os.link(temporary_path, output)
        return {"input_rows": read, "invalid_rows": invalid, **summary}
    finally:
        conn.close()
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an indexed inventory from one SMILES per line.")
    parser.add_argument("--smiles", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-invalid", action="store_true")
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args()
    try:
        if args.validate:
            if args.smiles or args.output or args.skip_invalid:
                parser.error("--validate cannot be combined with build options")
            result = validate_stock(args.validate)
        else:
            if not args.smiles or not args.output:
                parser.error("--smiles and --output are required when building stock")
            result = build_stock(args.smiles, args.output, skip_invalid=args.skip_invalid)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
