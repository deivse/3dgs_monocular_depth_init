

import re

from tabulate import tabulate
from results_processing_scripts.parameters import NerfbaselinesJSONParameter, ParamOrdering, Parameter, TensorboardParameter


def make_pretty_preset_name(preset_name: str) -> str:
    # preset_name = re.sub(
    #     r"depth_downsample_(\d+|adaptive)", r"[\1]", preset_name)

    # name = preset_name.replace("_", " ")
    # substitutions = {
    #     "metric3d": "Metric3Dv2",
    #     "unidepth": "UniDepth",
    #     "depth anything v2": "DA V2",
    #     "moge": "MoGe",
    # }
    # explicitly_capitalized = []
    # for sub in substitutions.values():
    #     explicitly_capitalized.extend(sub.split(" "))

    # for key, value in substitutions.items():
    #     name = name.replace(key, value)
    # name = " ".join(
    #     [
    #         word.capitalize() if word not in explicitly_capitalized else word
    #         for word in name.split(" ")
    #     ]
    # )
    return preset_name


PARAMS: dict[str, Parameter] = {
    "psnr": NerfbaselinesJSONParameter(
        name="PSNR", json_name="psnr", ordering=ParamOrdering.HIGHER_IS_BETTER
    ),
    "ssim": NerfbaselinesJSONParameter(
        name="SSIM", json_name="ssim", ordering=ParamOrdering.HIGHER_IS_BETTER
    ),
    "lpips": NerfbaselinesJSONParameter(
        name="LPIPS", json_name="lpips", ordering=ParamOrdering.LOWER_IS_BETTER
    ),
    "lpips_vgg": NerfbaselinesJSONParameter(
        name="LPIPS(VGG)",
        json_name="lpips_vgg",
        ordering=ParamOrdering.LOWER_IS_BETTER,
    ),
    "num_sfm_points": NerfbaselinesJSONParameter(
        name="Num Sfm Points",
        json_name="num_sfm_points",
        formatter=lambda val: f"{int(val):,}",
    ),
    "num_gaussians": TensorboardParameter(
        name="Num Gaussians",
        tensorboard_id="train/num-gaussians",
        formatter=lambda val: f"{int(float(val) / 1000):,}K",
        ordering=ParamOrdering.LOWER_IS_BETTER,
        should_highlight_best=False,
    ),
}

SCENES = {
    "mipnerf360": [
        "garden",
        "bonsai",
        "stump",
        "flowers",
        "bicycle",
        "kitchen",
        "treehill",
        "room",
        "counter",
    ],
    "tanksandtemples": [
        "auditorium",
        "ballroom",
        "courtroom",
        "museum",
        "palace",
        "temple",
        "family",
        "francis",
        "horse",
        "lighthouse",
        "m60",
        "panther",
        "playground",
        "train",
        "barn",
        "caterpillar",
        "church",
        "courthouse",
        "ignatius",
        "meetingroom",
        "truck",
    ],
}


class PresetFilter:
    def __init__(self, in_regex: str | None, out_regex: str | None):
        self.in_regex = re.compile(in_regex) if in_regex else None
        self.out_regex = re.compile(out_regex) if out_regex else None

    def allows(self, preset_name: str) -> bool:
        if self.in_regex and not self.in_regex.search(preset_name):
            return False
        if self.out_regex and self.out_regex.search(preset_name):
            return False
        return True


def preset_without_predictor(preset_id: str):
    KNOWN_PREDICTOR_IDS = [
        "metric3d",
        "unidepth",
        "depth_anything_v2_indoor",
        "depth_anything_v2_outdoor",
        "moge",
    ]
    for predictor_id in KNOWN_PREDICTOR_IDS:
        if predictor_id in preset_id:
            return preset_id.replace(f"{predictor_id}_", "")
    return preset_id


def format_best(val, output_fmt):
    MARKDOWN_FORMATS = ["github", "grid", "pipe",
                        "jira", "presto", "pretty", "rst"]
    if output_fmt in MARKDOWN_FORMATS:
        return f"***{val}***"
    if output_fmt == "latex":
        return f"\\textbf{{{val}}}"
    return f"*{val}"


Table = list[list[str]]


def table_to_csv_string(table: Table) -> str:
    import csv
    from io import StringIO

    output = StringIO()
    writer = csv.writer(output)
    for row in table:
        writer.writerow(row)
    return output.getvalue().strip()  # Remove trailing newline


def output_table(args, table: Table):
    if args.output_format == "latex":
        args.output_format = "latex_raw"

    if args.output_format == "csv":
        table_str = table_to_csv_string(table)
    else:
        table_str = tabulate(table, headers="firstrow",
                             tablefmt=args.output_format)

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(table_str + "\n")
    else:
        print(table_str + "\n")
