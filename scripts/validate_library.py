from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from library import load_library, validate_library_document


def main() -> int:
    try:
        document = load_library()
        report = validate_library_document(document)
    except Exception as exc:
        print(f"FALHA: {exc}")
        return 1
    print(f"VALIDAÇÃO OK — {report['item_count']} itens editoriais válidos.")
    for category, count in report["category_counts"].items():
        print(f"  {category}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
