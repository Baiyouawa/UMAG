from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent / "data"
    # 改用合并后的原始文件
    raw_path = base_dir / "physio_set_ab_raw.csv"
    output_path = base_dir / "Physio_norm.csv"
    labels_path = base_dir / "Physio_labels.csv"

    print(f"Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path)
    print(f"Raw shape: {df.shape}")

    # Keep a label file (one row per record).
    labels = (
        df[["RecordID", "In-hospital_death"]]
        .drop_duplicates(subset="RecordID")
        .sort_values("RecordID")
    )
    labels.to_csv(labels_path, index=False)
    print(f"Saved {len(labels)} labels to: {labels_path}")

    # Drop identifier/time/label columns to get pure feature matrix.
    features = df.drop(columns=["RecordID", "Time", "In-hospital_death"])

    # Convert sentinel -1 values to NaN (appear in Height/Weight/Gender).
    features = features.replace(-1, np.nan).astype(float)

    # Column-wise standardization ignoring NaNs.
    means = features.mean(skipna=True)
    stds = features.std(skipna=True, ddof=0)
    stds = stds.replace(0, 1.0)  # avoid division by zero

    normalized = (features - means) / stds

    normalized = normalized.to_numpy()
    normalized[np.isnan(normalized)] = -200

    np.savetxt(output_path, normalized, delimiter=",", fmt="%.6f")
    print(f"Saved normalized features to: {output_path} with shape {normalized.shape}")


if __name__ == "__main__":
    main()
