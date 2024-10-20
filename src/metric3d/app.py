import os
import os.path as osp

import torch

try:
    from mmcv.utils import Config
except ImportError:
    from mmengine import Config
import cv2
import gradio as gr
import numpy as np
import plotly.graph_objects as go
from metric3d.mono.model.monodepth_model import get_configured_monodepth_model
from metric3d.mono.utils.do_test import get_prediction, transform_test_data_scalecano
from metric3d.mono.utils.running import load_ckpt
from metric3d.mono.utils.transform import gray_to_colormap
from metric3d.mono.utils.unproj_pcd import reconstruct_pcd, save_point_cloud
from metric3d.mono.utils.visualization import vis_surface_normal
from PIL import ExifTags, Image

CODE_SPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# torch.hub.download_url_to_file('https://images.unsplash.com/photo-1437622368342-7a3d73a34c8f', 'turtle.jpg')
# torch.hub.download_url_to_file('https://images.unsplash.com/photo-1519066629447-267fffa62d4b', 'lions.jpg')

cfg_large = Config.fromfile("./mono/configs/HourglassDecoder/vit.raft5.large.py")
model_large = get_configured_monodepth_model(
    cfg_large,
)
model_large, _, _, _ = load_ckpt(
    "./weight/metric_depth_vit_large_800k.pth", model_large, strict_match=False
)
model_large.eval()

cfg_small = Config.fromfile("./mono/configs/HourglassDecoder/vit.raft5.small.py")
model_small = get_configured_monodepth_model(
    cfg_small,
)
model_small, _, _, _ = load_ckpt(
    "./weight/metric_depth_vit_small_800k.pth", model_small, strict_match=False
)
model_small.eval()

device = "cuda"
model_large.to(device)
model_small.to(device)


def predict_depth_normal(
    img, model_selection="vit-small", fx=1000.0, fy=1000.0, state_cache={}
):
    if model_selection == "vit-small":
        model = model_small
        cfg = cfg_small
    elif model_selection == "vit-large":
        model = model_large
        cfg = cfg_large
        pass
    else:
        return None, None, None, None, state_cache, "Not implemented model."

    if img is None:
        return (
            None,
            None,
            None,
            None,
            state_cache,
            "Please upload an image and wait for the upload to complete.",
        )

    cv_image = np.array(img)
    img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    intrinsic = [fx, fy, img.shape[1] / 2, img.shape[0] / 2]
    rgb_input, cam_models_stacks, pad, label_scale_factor = (
        transform_test_data_scalecano(img, intrinsic, cfg.data_basic)
    )

    with torch.no_grad():
        pred_depth, pred_depth_scale, scale, output, confidence = get_prediction(
            model=model,
            input=rgb_input,
            cam_model=cam_models_stacks,
            pad_info=pad,
            scale_info=label_scale_factor,
            gt_depth=None,
            normalize_scale=cfg.data_basic.depth_range[1],
            ori_shape=[img.shape[0], img.shape[1]],
        )

        pred_normal = output["normal_out_list"][0][:, :3, :, :]
        H, W = pred_normal.shape[2:]
        pred_normal = pred_normal[:, :, pad[0] : H - pad[1], pad[2] : W - pad[3]]

    pred_depth = pred_depth.squeeze().cpu().numpy()
    pred_depth[pred_depth < 0] = 0
    pred_color = gray_to_colormap(pred_depth)

    pred_normal = torch.nn.functional.interpolate(
        pred_normal, [img.shape[0], img.shape[1]], mode="bilinear"
    ).squeeze()
    pred_normal = pred_normal.permute(1, 2, 0)
    pred_color_normal = vis_surface_normal(pred_normal)
    pred_normal = pred_normal.cpu().numpy()

    # Storing depth and normal map in state for potential 3D reconstruction
    state_cache["depth"] = pred_depth
    state_cache["normal"] = pred_normal
    state_cache["img"] = img
    state_cache["intrinsic"] = intrinsic
    state_cache["confidence"] = confidence

    # save depth and normal map to .npy file
    if "save_dir" not in state_cache:
        cache_id = np.random.randint(0, 100000000000)
        while osp.exists(f"recon_cache/{cache_id:08d}"):
            cache_id = np.random.randint(0, 100000000000)
        state_cache["save_dir"] = f"recon_cache/{cache_id:08d}"
        os.makedirs(state_cache["save_dir"], exist_ok=True)
    depth_file = f"{state_cache['save_dir']}/depth.npy"
    normal_file = f"{state_cache['save_dir']}/normal.npy"
    np.save(depth_file, pred_depth)
    np.save(normal_file, pred_normal)

    # formatted = (output * 255 / np.max(output)).astype('uint8')
    img = Image.fromarray(pred_color)
    img_normal = Image.fromarray(pred_color_normal)
    return img, depth_file, img_normal, normal_file, state_cache, "Success!"


def get_camera(img):
    if img is None:
        return (
            None,
            None,
            None,
            "Please upload an image and wait for the upload to complete.",
        )
    try:
        exif = img.getexif()
        exif.update(exif.get_ifd(ExifTags.IFD.Exif))
    except:
        exif = {}
    sensor_width = exif.get(ExifTags.Base.FocalPlaneYResolution, None)
    sensor_height = exif.get(ExifTags.Base.FocalPlaneXResolution, None)
    focal_length = exif.get(ExifTags.Base.FocalLength, None)

    # convert sensor size to mm, see https://photo.stackexchange.com/questions/40865/how-can-i-get-the-image-sensor-dimensions-in-mm-to-get-circle-of-confusion-from
    w, h = img.size
    sensor_width = w / sensor_width * 25.4 if sensor_width is not None else None
    sensor_height = h / sensor_height * 25.4 if sensor_height is not None else None
    focal_length = focal_length * 1.0 if focal_length is not None else None

    message = "Success!"
    if focal_length is None:
        message = "Focal length not found in EXIF. Please manually input."
    elif sensor_width is None and sensor_height is None:
        sensor_width = 16
        sensor_height = h / w * sensor_width
        message = f"Sensor size not found in EXIF. Using {sensor_width}x{sensor_height:.2f} mm as default."

    return sensor_width, sensor_height, focal_length, message


def get_intrinsic(img, sensor_width, sensor_height, focal_length):
    if img is None:
        return None, None, "Please upload an image and wait for the upload to complete."
    if sensor_width is None or sensor_height is None or focal_length is None:
        return (
            1000,
            1000,
            "Insufficient information. Try detecting camera first or use default 1000 for fx and fy.",
        )
    if sensor_width == 0 or sensor_height == 0 or focal_length == 0:
        return (
            1000,
            1000,
            "Insufficient information. Try detecting camera first or use default 1000 for fx and fy.",
        )

    # calculate focal length in pixels
    w, h = img.size
    fx = w / sensor_width * focal_length if sensor_width is not None else None
    fy = h / sensor_height * focal_length if sensor_height is not None else None

    # if fx is None:
    #     return fy, fy, "Sensor width not provided, using fy for both fx and fy"
    # if fy is None:
    #     return fx, fx, "Sensor height not provided, using fx for both fx and fy"

    return fx, fy, "Success!"


def unprojection_pcd(state_cache):
    depth_map = state_cache.get("depth", None)
    normal_map = state_cache.get("normal", None)
    img = state_cache.get("img", None)
    intrinsic = state_cache.get("intrinsic", None)

    if depth_map is None or img is None:
        return None, "Please predict depth and normal first."

    # # downsample/upsample the depth map to confidence map size
    # confidence = state_cache.get('confidence', None)
    # if confidence is not None:
    #     H, W = confidence.shape
    #     # intrinsic[0] *= W / depth_map.shape[1]
    #     # intrinsic[1] *= H / depth_map.shape[0]
    #     # intrinsic[2] *= W / depth_map.shape[1]
    #     # intrinsic[3] *= H / depth_map.shape[0]
    #     depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)
    #     img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

    #     # filter out depth map by confidence
    #     mask = confidence.cpu().numpy() > 0

    # downsample the depth map if too large
    if depth_map.shape[0] > 1080:
        scale = 1080 / depth_map.shape[0]
        depth_map = cv2.resize(
            depth_map, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
        )
        img = cv2.resize(
            img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
        )
        intrinsic = [
            intrinsic[0] * scale,
            intrinsic[1] * scale,
            intrinsic[2] * scale,
            intrinsic[3] * scale,
        ]

    if "save_dir" not in state_cache:
        cache_id = np.random.randint(0, 100000000000)
        while osp.exists(f"recon_cache/{cache_id:08d}"):
            cache_id = np.random.randint(0, 100000000000)
        state_cache["save_dir"] = f"recon_cache/{cache_id:08d}"
        os.makedirs(state_cache["save_dir"], exist_ok=True)

    pcd_ply = f"{state_cache['save_dir']}/output.ply"
    pcd_obj = pcd_ply.replace(".ply", ".obj")

    pcd = reconstruct_pcd(
        depth_map, intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
    )
    # if mask is not None:
    #     pcd_filtered = pcd[mask]
    #     img_filtered = img[mask]
    pcd_filtered = pcd.reshape(-1, 3)
    img_filtered = img.reshape(-1, 3)

    save_point_cloud(pcd_filtered, img_filtered, pcd_ply, binary=False)
    # ply_to_obj(pcd_ply, pcd_obj)

    # downsample the point cloud for visualization
    num_samples = 250000
    if pcd_filtered.shape[0] > num_samples:
        indices = np.random.choice(pcd_filtered.shape[0], num_samples, replace=False)
        pcd_downsampled = pcd_filtered[indices]
        img_downsampled = img_filtered[indices]
    else:
        pcd_downsampled = pcd_filtered
        img_downsampled = img_filtered

    # plotly show
    color_str = np.array([f"rgb({r},{g},{b})" for b, g, r in img_downsampled])
    data = [
        go.Scatter3d(
            x=pcd_downsampled[:, 0],
            y=pcd_downsampled[:, 1],
            z=pcd_downsampled[:, 2],
            mode="markers",
            marker=dict(
                size=1,
                color=color_str,
                opacity=0.8,
            ),
        )
    ]
    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            camera=dict(eye=dict(x=0, y=0, z=-1), up=dict(x=0, y=-1, z=0)),
            xaxis=dict(showgrid=False, showticklabels=False, visible=False),
            yaxis=dict(showgrid=False, showticklabels=False, visible=False),
            zaxis=dict(showgrid=False, showticklabels=False, visible=False),
        ),
    )
    fig = go.Figure(data=data, layout=layout)

    return fig, pcd_ply, "Success!"


title = "Metric3D"
description = """# Metric3Dv2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation
Gradio demo for Metric3D v1/v2 which takes in a single image for computing metric depth and surface normal. To use it, simply upload your image, or click one of the examples to load them. Learn more from our paper linked below."""
article = "<p style='text-align: center'><a href='https://arxiv.org/pdf/2307.10984.pdf'>Metric3D arxiv</a> | <a href='https://arxiv.org/abs/2404.15506'>Metric3Dv2 arxiv</a> | <a href='https://github.com/YvanYin/Metric3D'>Github Repo</a></p>"

custom_css = """#button1, #button2 {
    width: 20px;
}"""

examples = [
    # ["turtle.jpg"],
    # ["lions.jpg"]
    # ["files/gundam.jpg"],
    "files/p50_pro.jpg",
    "files/iphone13.JPG",
    "files/canon_cat.JPG",
    "files/canon_dog.JPG",
    "files/museum.jpg",
    "files/terra.jpg",
    "files/underwater.jpg",
    "files/venue.jpg",
]


with gr.Blocks(title=title, css=custom_css) as demo:
    gr.Markdown(description + article)

    # input and control components
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Original Image")
            _ = gr.Examples(examples=examples, inputs=[image_input])
        with gr.Column():
            model_dropdown = gr.Dropdown(
                ["vit-small", "vit-large"], label="Model", value="vit-small"
            )

            with gr.Accordion("Advanced options (beta)", open=True):
                with gr.Row():
                    sensor_width = gr.Number(
                        None, label="Sensor Width in mm", precision=2
                    )
                    sensor_height = gr.Number(
                        None, label="Sensor Height in mm", precision=2
                    )
                    focal_len = gr.Number(None, label="Focal Length in mm", precision=2)
                    camera_detector = gr.Button(
                        "Detect Camera from EXIF", elem_id="#button1"
                    )
                with gr.Row():
                    fx = gr.Number(1000.0, label="fx in pixels", precision=2)
                    fy = gr.Number(1000.0, label="fy in pixels", precision=2)
                    focal_detector = gr.Button(
                        "Calculate Intrinsic", elem_id="#button2"
                    )

            message_box = gr.Textbox(label="Messages")

    # depth and normal
    submit_button = gr.Button("Predict Depth and Normal")
    with gr.Row():
        with gr.Column():
            depth_output = gr.Image(label="Output Depth")
            depth_file = gr.File(label="Depth (.npy)")
        with gr.Column():
            normal_output = gr.Image(label="Output Normal")
            normal_file = gr.File(label="Normal (.npy)")

    # 3D reconstruction
    reconstruct_button = gr.Button("Reconstruct 3D")
    pcd_output = gr.Plot(label="3D Point Cloud (Sampled sparse version)")
    pcd_ply = gr.File(label="3D Point Cloud (.ply)")

    # cache for depth, normal maps and other states
    state_cache = gr.State({})

    # detect focal length in pixels
    camera_detector.click(
        fn=get_camera,
        inputs=[image_input],
        outputs=[sensor_width, sensor_height, focal_len, message_box],
    )
    focal_detector.click(
        fn=get_intrinsic,
        inputs=[image_input, sensor_width, sensor_height, focal_len],
        outputs=[fx, fy, message_box],
    )

    submit_button.click(
        fn=predict_depth_normal,
        inputs=[image_input, model_dropdown, fx, fy, state_cache],
        outputs=[
            depth_output,
            depth_file,
            normal_output,
            normal_file,
            state_cache,
            message_box,
        ],
    )
    reconstruct_button.click(
        fn=unprojection_pcd,
        inputs=[state_cache],
        outputs=[pcd_output, pcd_ply, message_box],
    )

demo.launch()


# iface = gr.Interface(
#     depth_normal,
#     inputs=[
#         gr.Image(type='pil', label="Original Image"),
#         gr.Dropdown(["vit-small", "vit-large"], label="Model", info="Select a model type", value="vit-large")
#     ],
#     outputs=[
#         gr.Image(type="pil", label="Output Depth"),
#         gr.Image(type="pil", label="Output Normal"),
#         gr.Textbox(label="Messages")
#     ],
#     title=title,
#     description=description,
#     article=article,
#     examples=examples,
#     analytics_enabled=False
# )

# iface.launch()
