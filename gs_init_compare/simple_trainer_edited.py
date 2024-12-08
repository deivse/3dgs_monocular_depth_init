import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import yaml
from datasets.colmap import Dataset, Parser
from datasets.traj import generate_interpolated_path, generate_ellipse_path_z
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy

@dataclass
class Config:
    disable_viewer: bool = False
    ckpt: Optional[List[str]] = None
    compression: Optional[Literal['png']] = None
    render_traj_path: str = 'interp'
    data_dir: str = 'data/360_v2/garden'
    data_factor: int = 4
    result_dir: str = 'results/garden'
    test_every: int = 8
    patch_size: Optional[int] = None
    global_scale: float = 1.0
    normalize_world_space: bool = True
    port: int = 8080
    batch_size: int = 1
    steps_scaler: float = 1.0
    max_steps: int = 30000
    eval_steps: List[int] = field(default_factory=lambda : [7000, 30000])
    save_steps: List[int] = field(default_factory=lambda : [7000, 30000])
    init_type: str = 'sfm'
    init_num_pts: int = 100000
    init_extent: float = 3.0
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    init_opa: float = 0.1
    init_scale: float = 1.0
    ssim_lambda: float = 0.2
    near_plane: float = 0.01
    far_plane: float = 10000000000.0
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(default_factory=DefaultStrategy)
    packed: bool = False
    sparse_grad: bool = False
    antialiased: bool = False
    random_bkgd: bool = False
    opacity_reg: float = 0.0
    scale_reg: float = 0.0
    pose_opt: bool = False
    pose_opt_lr: float = 1e-05
    pose_opt_reg: float = 1e-06
    pose_noise: float = 0.0
    app_opt: bool = False
    app_embed_dim: int = 16
    app_opt_lr: float = 0.001
    app_opt_reg: float = 1e-06
    depth_loss: bool = False
    depth_lambda: float = 0.01
    tb_every: int = 100
    tb_save_image: bool = False
    lpips_net: Literal['vgg', 'alex'] = 'alex'
    app_test_opt_steps: int = 128
    app_test_opt_lr: float = 0.1
    background_color: Optional[Tuple[float, float, float]] = None

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)

def create_splats_with_optimizers(parser: Parser, init_type: str='sfm', init_num_pts: int=100000, init_extent: float=3.0, init_opacity: float=0.1, init_scale: float=1.0, scene_scale: float=1.0, sh_degree: int=3, sparse_grad: bool=False, batch_size: int=1, feature_dim: Optional[int]=None, device: str='cuda', world_rank: int=0, world_size: int=1) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == 'sfm':
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == 'random':
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError('Please specify a correct init_type: sfm or random')
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]
    N = points.shape[0]
    quats = torch.rand((N, 4))
    opacities = torch.logit(torch.full((N,), init_opacity))
    params = [('means', torch.nn.Parameter(points), 0.00016 * scene_scale), ('scales', torch.nn.Parameter(scales), 0.005), ('quats', torch.nn.Parameter(quats), 0.001), ('opacities', torch.nn.Parameter(opacities), 0.05)]
    if feature_dim is None:
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(('sh0', torch.nn.Parameter(colors[:, :1, :]), 0.0025))
        params.append(('shN', torch.nn.Parameter(colors[:, 1:, :]), 0.0025 / 20))
    else:
        features = torch.rand(N, feature_dim)
        params.append(('features', torch.nn.Parameter(features), 0.0025))
        colors = torch.logit(rgbs)
        params.append(('colors', torch.nn.Parameter(colors), 0.0025))
    splats = torch.nn.ParameterDict({n: v for (n, v, _) in params}).to(device)
    BS = batch_size * world_size
    optimizers = {name: (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)([{'params': splats[name], 'lr': lr * math.sqrt(BS), 'name': name}], eps=1e-15 / math.sqrt(BS), betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999))) for (name, _, lr) in params}
    return (splats, optimizers)

class Runner:
    """Engine for training and testing."""

    def __init__(self, local_rank: int, world_rank, world_size: int, cfg: Config, Parser, Dataset) -> None:
        set_random_seed(42 + local_rank)
        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f'cuda:{local_rank}'
        self.parser = Parser(data_dir=cfg.data_dir, factor=cfg.data_factor, normalize=cfg.normalize_world_space, test_every=cfg.test_every)
        self.trainset = Dataset(self.parser, split='train', patch_size=cfg.patch_size, load_depths=cfg.depth_loss)
        self.valset = Dataset(self.parser, split='val')
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print('Scene scale:', self.scene_scale)
        feature_dim = 32 if cfg.app_opt else None
        (self.splats, self.optimizers) = create_splats_with_optimizers(self.parser, init_type=cfg.init_type, init_num_pts=cfg.init_num_pts, init_extent=cfg.init_extent, init_opacity=cfg.init_opa, init_scale=cfg.init_scale, scene_scale=self.scene_scale, sh_degree=cfg.sh_degree, sparse_grad=cfg.sparse_grad, batch_size=cfg.batch_size, feature_dim=feature_dim, device=self.device, world_rank=world_rank, world_size=world_size)
        print('Model initialized. Number of GS:', len(self.splats['means']))
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)
        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(scene_scale=self.scene_scale)
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == 'png':
                self.compression_method = PngCompression()
            else:
                raise ValueError(f'Unknown compression strategy: {cfg.compression}')
        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [torch.optim.Adam(self.pose_adjust.parameters(), lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size), weight_decay=cfg.pose_opt_reg)]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)
        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)
        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree).to(self.device)
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [torch.optim.Adam(self.app_module.embeds.parameters(), lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0, weight_decay=cfg.app_opt_reg), torch.optim.Adam(self.app_module.color_head.parameters(), lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size))]
            if world_size > 1:
                self.app_module = DDP(self.app_module)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

    def rasterize_splats(self, camtoworlds: Tensor, Ks: Tensor, width: int, height: int, **kwargs) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats['means']
        quats = self.splats['quats']
        scales = torch.exp(self.splats['scales'])
        opacities = torch.sigmoid(self.splats['opacities'])
        image_ids = kwargs.pop('image_ids', None)
        if self.cfg.app_opt:
            colors = self.app_module(features=self.splats['features'], embed_ids=image_ids, dirs=means[None, :, :] - camtoworlds[:, None, :3, 3], sh_degree=kwargs.pop('sh_degree', self.cfg.sh_degree))
            colors = colors + self.splats['colors']
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats['sh0'], self.splats['shN']], 1)
        rasterize_mode = 'antialiased' if self.cfg.antialiased else 'classic'
        (render_colors, render_alphas, info) = rasterization(means=means, quats=quats, scales=scales, opacities=opacities, colors=colors, viewmats=torch.linalg.inv(camtoworlds), Ks=Ks, width=width, height=height, packed=self.cfg.packed, absgrad=self.cfg.strategy.absgrad if isinstance(self.cfg.strategy, DefaultStrategy) else False, sparse_grad=self.cfg.sparse_grad, rasterize_mode=rasterize_mode, distributed=self.world_size > 1, **kwargs)
        return (render_colors, render_alphas, info)

    def setup_train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size
        max_steps = cfg.max_steps
        init_step = 0
        schedulers = [torch.optim.lr_scheduler.ExponentialLR(self.optimizers['means'], gamma=0.01 ** (1.0 / max_steps))]
        if cfg.pose_opt:
            schedulers.append(torch.optim.lr_scheduler.ExponentialLR(self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)))
        trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        trainloader_iter = iter(trainloader)
        self.trainloader = trainloader
        self.trainloader_iter = trainloader_iter
        self.schedulers = schedulers

    @torch.no_grad()
    def eval(self, step: int, stage: str='val'):
        """Entry for evaluation."""
        print('Running evaluation...')
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size
        valloader = torch.utils.data.DataLoader(self.valset, batch_size=1, shuffle=False, num_workers=0)
        ellipse_time = 0
        metrics = {'psnr': [], 'ssim': [], 'lpips': []}
        for (i, data) in enumerate(valloader):
            camtoworlds = data['camtoworld'].to(device)
            Ks = data['K'].to(device)
            pixels = data['image'].to(device) / 255.0
            (height, width) = pixels.shape[1:3]
            torch.cuda.synchronize()
            tic = time.time()
            (colors, _, _) = self.rasterize_splats(camtoworlds=camtoworlds, Ks=Ks, width=width, height=height, sh_degree=cfg.sh_degree, near_plane=cfg.near_plane, far_plane=cfg.far_plane)
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic
            colors = torch.clamp(colors, 0.0, 1.0)
            canvas_list = [pixels, colors]
            if world_rank == 0:
                canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(f'{self.render_dir}/{stage}_step{step}_{i:04d}.png', canvas)
                pixels = pixels.permute(0, 3, 1, 2)
                colors = colors.permute(0, 3, 1, 2)
                metrics['psnr'].append(self.psnr(colors, pixels))
                metrics['ssim'].append(self.ssim(colors, pixels))
                metrics['lpips'].append(self.lpips(colors, pixels))
        if world_rank == 0:
            ellipse_time /= len(valloader)
            psnr = torch.stack(metrics['psnr']).mean()
            ssim = torch.stack(metrics['ssim']).mean()
            lpips = torch.stack(metrics['lpips']).mean()
            print(f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f} Time: {ellipse_time:.3f}s/image Number of GS: {len(self.splats['means'])}")
            stats = {'psnr': psnr.item(), 'ssim': ssim.item(), 'lpips': lpips.item(), 'ellipse_time': ellipse_time, 'num_GS': len(self.splats['means'])}
            with open(f'{self.stats_dir}/{stage}_step{step:04d}.json', 'w') as f:
                json.dump(stats, f)
            for (k, v) in stats.items():
                self.writer.add_scalar(f'{stage}/{k}', v, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        print('Running trajectory rendering...')
        cfg = self.cfg
        device = self.device
        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == 'interp':
            camtoworlds_all = generate_interpolated_path(camtoworlds_all, 1)
        elif cfg.render_traj_path == 'ellipse':
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(camtoworlds_all, height=height)
        else:
            raise ValueError(f'Render trajectory type not supported: {cfg.render_traj_path}')
        camtoworlds_all = np.concatenate([camtoworlds_all, np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0)], axis=1)
        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        (width, height) = list(self.parser.imsize_dict.values())[0]
        canvas_all = []
        for i in tqdm.trange(len(camtoworlds_all), desc='Rendering trajectory'):
            camtoworlds = camtoworlds_all[i:i + 1]
            Ks = K[None]
            (renders, _, _) = self.rasterize_splats(camtoworlds=camtoworlds, Ks=Ks, width=width, height=height, sh_degree=cfg.sh_degree, near_plane=cfg.near_plane, far_plane=cfg.far_plane, render_mode='RGB+ED')
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)
            depths = renders[..., 3:4]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            canvas_all.append(canvas)
        video_dir = f'{cfg.result_dir}/videos'
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f'{video_dir}/traj_{step}.mp4', fps=30)
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f'Video saved to {video_dir}/traj_{step}.mp4')

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print('Running compression...')
        world_rank = self.world_rank
        compress_dir = f'{cfg.result_dir}/compression/rank{world_rank}'
        os.makedirs(compress_dir, exist_ok=True)
        self.compression_method.compress(compress_dir, self.splats)
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage='compress')

    @torch.no_grad()
    def _viewer_render_fn(self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]):
        """Callable function for the viewer."""
        (W, H) = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)
        (render_colors, _, _) = self.rasterize_splats(camtoworlds=c2w[None], Ks=K[None], width=W, height=H, sh_degree=self.cfg.sh_degree, radius_clip=3.0)
        return render_colors[0].cpu().numpy()

    def train_iteration(self, step):
        trainloader_iter = self.trainloader_iter
        trainloader = self.trainloader
        schedulers = self.schedulers
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size
        try:
            data = next(trainloader_iter)
        except StopIteration:
            trainloader_iter = iter(trainloader)
            data = next(trainloader_iter)
        camtoworlds = camtoworlds_gt = data['camtoworld'].to(device)
        Ks = data['K'].to(device)
        pixels = data['image'].to(device) / 255.0
        num_train_rays_per_step = pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
        image_ids = data['image_id'].to(device)
        if cfg.depth_loss:
            points = data['points'].to(device)
            depths_gt = data['depths'].to(device)
        (height, width) = pixels.shape[1:3]
        if cfg.pose_noise:
            camtoworlds = self.pose_perturb(camtoworlds, image_ids)
        if cfg.pose_opt:
            camtoworlds = self.pose_adjust(camtoworlds, image_ids)
        sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)
        (renders, alphas, info) = self.rasterize_splats(camtoworlds=camtoworlds, Ks=Ks, width=width, height=height, sh_degree=sh_degree_to_use, near_plane=cfg.near_plane, far_plane=cfg.far_plane, image_ids=image_ids, render_mode='RGB+ED' if cfg.depth_loss else 'RGB')
        if data.get('sampling_mask') is not None:
            sampling_mask = data['sampling_mask'].to(self.device)
            renders = renders * sampling_mask + renders.detach() * (1 - sampling_mask)
            alphas = alphas * sampling_mask + alphas.detach() * (1 - sampling_mask)
        if renders.shape[-1] == 4:
            (colors, depths) = (renders[..., 0:3], renders[..., 3:4])
        else:
            (colors, depths) = (renders, None)
        if cfg.random_bkgd:
            bkgd = torch.rand(1, 3, device=device)
            colors = colors + bkgd * (1.0 - alphas)
        if not cfg.random_bkgd and cfg.background_color is not None:
            bkgd = torch.tensor(cfg.background_color, dtype=colors.dtype, device=self.device).view(1, 1, 1, 3)
            colors = colors + bkgd * (1.0 - alphas)
        self.cfg.strategy.step_pre_backward(params=self.splats, optimizers=self.optimizers, state=self.strategy_state, step=step, info=info)
        l1loss = F.l1_loss(colors, pixels)
        ssimloss = 1.0 - self.ssim(pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2))
        loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
        if cfg.depth_loss:
            points = torch.stack([points[:, :, 0] / (width - 1) * 2 - 1, points[:, :, 1] / (height - 1) * 2 - 1], dim=-1)
            grid = points.unsqueeze(2)
            depths = F.grid_sample(depths.permute(0, 3, 1, 2), grid, align_corners=True)
            depths = depths.squeeze(3).squeeze(1)
            disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
            disp_gt = 1.0 / depths_gt
            depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
            loss += depthloss * cfg.depth_lambda
        if cfg.opacity_reg > 0.0:
            loss = loss + cfg.opacity_reg * torch.abs(torch.sigmoid(self.splats['opacities'])).mean()
        if cfg.scale_reg > 0.0:
            loss = loss + cfg.scale_reg * torch.abs(torch.exp(self.splats['scales'])).mean()
        loss.backward()
        desc = f'loss={loss.item():.3f}| sh degree={sh_degree_to_use}| '
        if cfg.depth_loss:
            desc += f'depth loss={depthloss.item():.6f}| '
        if cfg.pose_opt and cfg.pose_noise:
            pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
            desc += f'pose err={pose_err.item():.6f}| '
        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.cfg.strategy.step_post_backward(params=self.splats, optimizers=self.optimizers, state=self.strategy_state, step=step, info=info, packed=cfg.packed)
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.cfg.strategy.step_post_backward(params=self.splats, optimizers=self.optimizers, state=self.strategy_state, step=step, info=info, lr=schedulers[0].get_last_lr()[0])
        else:
            assert_never(self.cfg.strategy)
        if cfg.sparse_grad:
            assert cfg.packed, 'Sparse gradients only work with packed mode.'
            gaussian_ids = info['gaussian_ids']
            for k in self.splats.keys():
                grad = self.splats[k].grad
                if grad is None or grad.is_sparse:
                    continue
                self.splats[k].grad = torch.sparse_coo_tensor(indices=gaussian_ids[None], values=grad[gaussian_ids], size=self.splats[k].size(), is_coalesced=len(Ks) == 1)
        for optimizer in self.optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for optimizer in self.pose_optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for optimizer in self.app_optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for scheduler in schedulers:
            scheduler.step()
        if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
            self.run_compression(step=step)
        self.trainloader_iter = trainloader_iter
        out = {'loss': loss.item(), 'l1loss': l1loss.item(), 'ssim': ssimloss.item(), 'num_gaussians': len(self.splats['means'])}
        if cfg.depth_loss:
            out['depthloss'] = depthloss.item()
        return out

    def save(self, step, path):
        world_size = self.world_size
        cfg = self.cfg
        data = {'step': step, 'splats': self.splats.state_dict()}
        if cfg.pose_opt:
            if world_size > 1:
                data['pose_adjust'] = self.pose_adjust.module.state_dict()
            else:
                data['pose_adjust'] = self.pose_adjust.state_dict()
        if cfg.app_opt:
            if world_size > 1:
                data['app_module'] = self.app_module.module.state_dict()
            else:
                data['app_module'] = self.app_module.state_dict()
        torch.save(data, f'{path}/ckpt_{step}_rank{self.world_rank}.pt')

def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and (not cfg.disable_viewer):
        cfg.disable_viewer = True
        if world_rank == 0:
            print('Viewer is disabled in distributed training.')
    runner = Runner(local_rank, world_rank, world_size, cfg)
    if cfg.ckpt is not None:
        ckpts = [torch.load(file, map_location=runner.device, weights_only=True) for file in cfg.ckpt]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt['splats'][k] for ckpt in ckpts])
        step = ckpts[0]['step']
        runner.eval(step=step)
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()
    if not cfg.disable_viewer:
        print('Viewer running... Ctrl+C to exit.')
        time.sleep(1000000)
if __name__ == '__main__':
    '\n    Usage:\n\n    ```bash\n    # Single GPU training\n    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default\n\n    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.\n    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25\n\n    '
    configs = {'default': ('Gaussian splatting training using densification heuristics from the original paper.', Config(strategy=DefaultStrategy(verbose=True))), 'mcmc': ("Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.", Config(init_opa=0.5, init_scale=0.1, opacity_reg=0.01, scale_reg=0.01, strategy=MCMCStrategy(verbose=True)))}
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)
    if cfg.compression == 'png':
        try:
            import plas
            import torchpq
        except:
            raise ImportError("To use PNG compression, you need to install torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') ")
    cli(main, cfg, verbose=True)