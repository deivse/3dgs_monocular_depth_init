from __future__ import annotations
import argparse
import csv
from typing import List, Sequence, Union, Tuple, Optional
import math
from matplotlib import pyplot as plt
import numpy as np
import os
import re

NumberLike = Union[int, float, str]
Row = List[NumberLike]
Table = List[Row]


def visualize_preset_table(
    table: Table,
    figsize: Tuple[float, float] = (10, 5),
    rotation: int = 15,
    bar_width: Optional[float] = None,
    ax=None,
):
    if not table or not table[0]:
        raise ValueError("Table must be non-empty")

    import matplotlib.pyplot as plt

    header = table[0]
    if len(header) < 2:
        raise ValueError("Header must contain at least one label column")

    title = str(header[0])
    labels = [str(l) for l in header[1:]]

    presets: List[str] = []
    data_matrix: List[List[float]] = []
    for r_i, row in enumerate(table[1:], start=1):
        if len(row) != len(header):
            raise ValueError(
                f"Row {r_i} length {len(row)} != header length {len(header)}")
        preset_name = str(row[0])
        presets.append(preset_name)
        numeric_row: List[float] = []
        for c_i, cell in enumerate(row[1:], start=1):
            try:
                val = float(cell)
            except (TypeError, ValueError):
                val = float("nan")
            numeric_row.append(val)
        data_matrix.append(numeric_row)

    if not data_matrix:
        raise ValueError("No data rows found")

    num_presets = len(presets)
    num_labels = len(labels)
    data_np = np.array(data_matrix, dtype=float)

    if bar_width is None:
        bar_width = min(0.8 / num_presets, 0.25)

    import matplotlib.pyplot as plt
    x = np.arange(num_labels)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    prop_cycle = plt.rcParams.get("axes.prop_cycle")
    colors = (prop_cycle.by_key().get("color") or [])[:num_presets]
    if len(colors) < num_presets:
        cmap = plt.get_cmap("tab20")
        for i in range(num_presets - len(colors)):
            colors.append(cmap(i / max(1, num_presets - len(colors) - 1)))

    offsets_centered = (np.arange(num_presets) -
                        (num_presets - 1) / 2.0) * bar_width
    for p_idx, (preset, offset) in enumerate(zip(presets, offsets_centered)):
        row_vals = data_np[p_idx]
        valid_mask = ~np.isnan(row_vals)
        if not np.any(valid_mask):
            continue
        ax.bar(
            x[valid_mask] + offset,
            row_vals[valid_mask],
            width=bar_width * 0.95,
            label=preset,
            color=colors[p_idx],
            edgecolor="black",
            linewidth=0.5,
        )
        if num_presets * num_labels <= 60:
            for xi, val in zip(x[valid_mask] + offset, row_vals[valid_mask]):
                ax.text(
                    xi,
                    val,
                    f"{val:.2g}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90 if num_presets * num_labels > 25 else 0,
                )

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotation,
                       ha="right" if rotation else "center")
    ax.set_xlim(-0.6, num_labels - 0.4)
    ax.set_ylabel("Value")
    ax.legend(title="Preset", frameon=False, ncol=math.ceil(num_presets / 6))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.margins(y=0.1)
    fig.tight_layout()
    return fig, ax


def _sanitize_title(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_.-]", "", s)
    return s or "table"


def _split_csv_into_tables(path: str) -> List[Table]:
    tables: List[Table] = []
    current: Table = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for raw_row in reader:
            # Separator: completely empty row or all cells blank
            if not raw_row or all((c.strip() == "" for c in raw_row)):
                if current:
                    tables.append(current)
                    current = []
                continue
            parsed_row: Row = []
            for cell in raw_row:
                try:
                    parsed_row.append(float(cell))
                except ValueError:
                    parsed_row.append(cell)
            current.append(parsed_row)
    if current:
        tables.append(current)
    return tables


def main():
    argparser = argparse.ArgumentParser(
        description="Visualize one or more patches results tables (blank-line separated) as grouped bar charts."
    )
    argparser.add_argument("input_file", type=str,
                           help="Path to the input CSV file.")
    argparser.add_argument("output_file", type=str,
                           help="Path to the output image file (base if multiple tables).")
    args = argparser.parse_args()

    tables = _split_csv_into_tables(args.input_file)
    if not tables:
        raise SystemExit("No tables found in CSV.")

    base, ext = os.path.splitext(args.output_file)
    used_names = set()

    multiple = len(tables) > 1
    for idx, table in enumerate(tables):
        title_cell = str(table[0][0]) if table and table[0] else f"table{idx+1}"
        safe = _sanitize_title(title_cell)
        if safe in used_names:
            # disambiguate
            suffix_counter = 2
            candidate = f"{safe}_{suffix_counter}"
            while candidate in used_names:
                suffix_counter += 1
                candidate = f"{safe}_{suffix_counter}"
            safe = candidate
        used_names.add(safe)

        if multiple:
            out_path = f"{base}_{safe}{ext or '.png'}"
        else:
            out_path = args.output_file if ext else f"{base}.png"

        fig, ax = visualize_preset_table(table)
        fig.savefig(out_path)
        plt.show()
        print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()
