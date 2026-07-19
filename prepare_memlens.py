from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.memlens import prepare_memlens_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and convert the official MEMLENS 32K agent subset to MemEye records."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset_32k.json")
    parser.add_argument("--subset", type=Path, required=True, help="Path to agent_subset_195.json")
    parser.add_argument(
        "--image-root", type=Path, required=True, help="Path to the MEMLENS release_images directory"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data_external/memlens/converted_32k_agent195"))
    parser.add_argument("--skip-image-check", action="store_true")
    parser.add_argument("--expected-subset-size", type=int, default=195)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare_memlens_records(
        args.dataset,
        args.subset,
        args.image_root,
        args.output_dir,
        check_images=not args.skip_image_check,
        expected_subset_size=args.expected_subset_size,
    )
    print(
        f"Converted {manifest['converted_items']}/{manifest['requested_ids']} records; "
        f"invalid={manifest['invalid_items']}; manifest={args.output_dir / 'manifest.json'}"
    )


if __name__ == "__main__":
    main()
