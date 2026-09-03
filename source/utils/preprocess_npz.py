from pathlib import Path

import numpy as np
import pandas as pd


ras_input_dir = Path(r"E:\Tasks\PINN\Wuda\Generalization_Ability\Joint\runs\joint_20260831_164351\hydrodynamics")
geo_input_dir = Path(r"E:\Tasks\PINN\Wuda\generalization\demo_data\geo")
ras_output_dir = Path(r"E:\Tasks\PINN\Wuda\generalization\data\ras")
geo_output_dir = Path(r"E:\Tasks\PINN\Wuda\generalization\data\geo")
OVERWRITE = False


def dataframe_to_npz(csv_file, output_file, overwrite=False):
    """将 CSV 原始表格逐列保存为 NPZ"""
    if output_file.exists() and not overwrite:
        return "skipped"

    frame = pd.read_csv(csv_file, encoding="utf-8-sig")
    arrays = {}
    for column in frame.columns:
        series = frame[column]
        if (
            pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)
        ):
            arrays[column] = series.fillna("").astype(str).to_numpy(dtype=str)
        else:
            arrays[column] = series.to_numpy(copy=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(".npz.tmp")
    with temporary_file.open("wb") as file_handle:
        np.savez_compressed(file_handle, **arrays)
    temporary_file.replace(output_file)
    return "written"


def convert_directory(input_dir, output_dir, pattern, overwrite=False, limit=None):
    csv_files = sorted(input_dir.glob(pattern))
    if limit is not None:
        csv_files = csv_files[:limit]
    if not csv_files:
        raise FileNotFoundError(f"No files matching {pattern} under {input_dir}")

    written = 0
    skipped = 0
    for index, csv_file in enumerate(csv_files, start=1):
        output_file = output_dir / f"{csv_file.stem}.npz"
        status = dataframe_to_npz(csv_file, output_file, overwrite=overwrite)
        written += status == "written"
        skipped += status == "skipped"
        print(f"[{index}/{len(csv_files)}] {status}: {output_file.name}", flush=True)

    print(f"complete: written={written}, skipped={skipped}, output={output_dir}")


def main():
    convert_directory(
        ras_input_dir,
        ras_output_dir,
        "*_hydrodynamics.csv",
        overwrite=OVERWRITE,
    )
    convert_directory(
        geo_input_dir,
        geo_output_dir,
        "*_cross_section_geometry.csv",
        overwrite=OVERWRITE,
    )


if __name__ == "__main__":
    main()
