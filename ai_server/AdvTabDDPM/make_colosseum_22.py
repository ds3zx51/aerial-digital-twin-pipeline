"""Convert the legacy 26-numerical + 5-categorical arrays to 20+3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NUMERICAL_KEEP = [
    0,   # Timestamp
    1,   # num_ues
    2,   # slice_prb
    4,   # dl_mcs
    5,   # dl_n_samples
    6,   # dl_buffer [bytes]
    7,   # tx_brate downlink [Mbps]
    8,   # tx_pkts downlink
    10,  # dl_cqi
    11,  # ul_mcs
    12,  # ul_n_samples
    13,  # ul_buffer [bytes]
    14,  # rx_brate uplink [Mbps]
    15,  # rx_pkts uplink
    16,  # rx_errors uplink (%)
    18,  # ul_sinr
    19,  # phr
    20,  # sum_requested_prbs
    21,  # sum_granted_prbs
    25,  # ul_turbo_iters
]

# Legacy categorical order: IMSI, RNTI, slicing_enabled, slice_id,
# scheduling_policy.
CATEGORICAL_KEEP = [1, 3, 4]


def convert(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    split_sizes: dict[str, int] = {}

    for split in ("train", "val", "test"):
        x_num = np.load(src / f"X_num_{split}.npy", allow_pickle=True)
        x_cat = np.load(src / f"X_cat_{split}.npy", allow_pickle=True)
        y = np.load(src / f"y_{split}.npy", allow_pickle=True)

        if x_num.ndim != 2 or x_num.shape[1] != 26:
            raise ValueError(
                f"{split}: expected X_num shape [N, 26], got {x_num.shape}"
            )
        if x_cat.ndim != 2 or x_cat.shape[1] != 5:
            raise ValueError(
                f"{split}: expected X_cat shape [N, 5], got {x_cat.shape}"
            )
        if len(x_num) != len(x_cat) or len(x_num) != len(y):
            raise ValueError(f"{split}: array lengths do not match")

        x_num_22 = x_num[:, NUMERICAL_KEEP]
        x_cat_22 = x_cat[:, CATEGORICAL_KEEP]

        np.save(dst / f"X_num_{split}.npy", x_num_22)
        np.save(dst / f"X_cat_{split}.npy", x_cat_22)
        np.save(dst / f"y_{split}.npy", y)
        split_sizes[split] = int(len(y))

        print(
            f"{split}: X_num={x_num_22.shape}, "
            f"X_cat={x_cat_22.shape}, y={y.shape}"
        )

    info_path = src / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
    else:
        info = {
            "name": "Network Traffic",
            "id": "network-traffic--v3",
            "task_type": "multiclass",
        }

    info.update(
        {
            "n_num_features": 20,
            "n_cat_features": 3,
            "train_size": split_sizes["train"],
            "val_size": split_sizes["val"],
            "test_size": split_sizes["test"],
        }
    )
    (dst / "info.json").write_text(
        json.dumps(info, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Created: {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("data") / "colosseum",
        help="Directory containing the legacy X_num/X_cat/y arrays",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("data") / "colosseum_22",
        help="Output directory for the 20+3 arrays",
    )
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == "__main__":
    main()
