import subprocess
from pathlib import Path


def url4name(name):
    return f"https://huggingface.co/spaces/JUGGHM/Metric3D/resolve/main/weight/{name}?download=true"


if __name__ == "__main__":
    DIR = "weight"
    Path("weights").mkdir(exist_ok=True)
    for name in [
        "convlarge_hourglass_0.3_150_step750k_v1.1.pth",
        "metric_depth_vit_giant2_800k.pth",
        "metric_depth_vit_large_800k.pth",
        "metric_depth_vit_small_800k.pth",
    ]:
        if Path(f"{DIR}/{name}").exists():
            print(f"Skipping {name} (already exists)")
            continue
        url = url4name(name)
        print(f"Downloading {name} from {url}")
        subprocess.run(["wget", url, "-O", f"{DIR}/{name}"], check=True)
