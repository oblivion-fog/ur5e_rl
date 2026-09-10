from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path


ASSET_NAMES: tuple[str, ...] = (
    "base_0.obj",
    "base_1.obj",
    "forearm_0.obj",
    "forearm_1.obj",
    "forearm_2.obj",
    "forearm_3.obj",
    "shoulder_0.obj",
    "shoulder_1.obj",
    "shoulder_2.obj",
    "upperarm_0.obj",
    "upperarm_1.obj",
    "upperarm_2.obj",
    "upperarm_3.obj",
    "wrist1_0.obj",
    "wrist1_1.obj",
    "wrist1_2.obj",
    "wrist2_0.obj",
    "wrist2_1.obj",
    "wrist2_2.obj",
    "wrist3.obj",
)

RAW_BASE = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/universal_robots_ur5e"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备 MuJoCo Menagerie UR5e mesh assets")
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="已有 UR5e assets 目录；不提供则从官方 Menagerie 下载",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets",
        help="工程 assets 输出目录",
    )
    return parser.parse_args()


def copy_assets(source: Path, destination: Path) -> None:
    for name in ASSET_NAMES:
        source_file = source / name
        if not source_file.exists():
            raise FileNotFoundError(f"缺少 UR5e mesh: {source_file}")
        shutil.copy2(source_file, destination / name)


def download_assets(destination: Path) -> None:
    for index, name in enumerate(ASSET_NAMES, start=1):
        url = f"{RAW_BASE}/assets/{name}"
        print(f"[{index:02d}/{len(ASSET_NAMES):02d}] {name}")
        urllib.request.urlretrieve(url, destination / name)

    urllib.request.urlretrieve(f"{RAW_BASE}/LICENSE", destination.parent / "UR5E_LICENSE")


def main() -> None:
    args = parse_args()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    if args.source is not None:
        copy_assets(args.source.resolve(), destination)
    else:
        download_assets(destination)

    print("UR5e assets 已准备：", destination)


if __name__ == "__main__":
    main()
