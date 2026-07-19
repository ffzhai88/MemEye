from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.memlens.official import prepare_official_memlens_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the official MEMLENS agent subset.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-image-check", action="store_true")
    args = parser.parse_args()
    result = prepare_official_memlens_records(
        args.dataset,
        args.subset,
        args.image_root,
        args.output_dir,
        check_images=not args.skip_image_check,
    )
    print(
        f"Converted {result['converted_items']}/{result['requested_ids']} records; "
        f"invalid={result['invalid_items']}"
    )


if __name__ == "__main__":
    main()
