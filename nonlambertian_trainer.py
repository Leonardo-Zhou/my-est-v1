from __future__ import absolute_import, division, print_function

import time
import json
import datasets
import networks
import torch.optim as optim
from utils import *
from layers import *
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.nn.functional as F

class NonLambertianTrainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        # Initialize networks
        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        # Use the new non-Lambertian decompose decoder
        self.models["decompose_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["decompose_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["decompose_encoder"].parameters())
        
        self.models["decompose"] = networks.nonlambertian_decompose_decoder(
            self.models["decompose_encoder"].num_ch_enc, self.opt.scales)
        self.models["decompose"].to(self.device)
        self.parameters_to_train += list(self.models["decompose"].parameters())

        self.models["adjust_net"] = networks.adjust_net()
        self.models["adjust_net"].to(self.device)
        self.parameters_to_train += list(self.models["adjust_net"].parameters())

        self.models["pose_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers,
            self.opt.weights_init == "pretrained",
            num_input_images=self.num_pose_frames)
        self.models["pose_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["pose_encoder"].parameters())

        self.models["pose"] = networks.PoseDecoder(
            self.models["pose_encoder"].num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2)
        self.models["pose"].to(self.device)
        self.parameters_to_train += list(self.models["pose"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.model_optimizer, [self.opt.scheduler_step_size], 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training Non-Lambertian model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"endovis": datasets.SCAREDRAWDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=min(self.opt.num_workers, 6),  # 限制worker数量避免CPU瓶颈
            pin_memory=True, 
            drop_last=True,
            prefetch_factor=2,      # 减少预取因子
            persistent_workers=True # 保持worker进程，减少启动开销
        )
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, False,
            num_workers=1, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.ssim = SSIM()
        self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode"""
        for model_name in self.models:
            self.models[model_name].train()
            for param in self.models[model_name].parameters():
                param.requires_grad = True

    def set_eval(self):
        """Convert all models to testing/evaluation mode"""
        for model_name in self.models:
            self.models[model_name].eval()

    def train(self):
        """Run the entire training pipeline"""
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation"""
        print("Training")
        print(self.model_optimizer.param_groups[0]['lr'])
        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            # depth, pose, decompose
            self.set_train()
            outputs, losses = self.process_batch(inputs)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            phase = batch_idx % self.opt.log_frequency == 0

            if phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

        self.model_lr_scheduler.step()

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses"""
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device, non_blocking=True)  # 异步传输，减少等待时间

        # depth estimation
        features = self.models["encoder"](inputs["color_aug", 0, 0])
        outputs = self.models["depth"](features)

        # pose estimation
        outputs.update(self.predict_poses(inputs))

        # Non-Lambertian decomposition (I = A × S + R)
        self.nonlambertian_decompose(inputs, outputs)

        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs):
        """Predict poses between input frames for monocular sequences."""
        outputs = {}
        if self.num_pose_frames == 2:
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
                
            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    if f_i < 0:
                        inputs_all = [pose_feats[f_i], pose_feats[0]]
                    else:
                        inputs_all = [pose_feats[0], pose_feats[f_i]]

                    # pose
                    pose_inputs = [self.models["pose_encoder"](torch.cat(inputs_all, 1))]
                    axisangle, translation = self.models["pose"](pose_inputs)

                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
         
        return outputs
    
    def nonlambertian_decompose(self, inputs, outputs):
        """Perform non-Lambertian decomposition: I = A × S + R"""
        # 批量处理特征提取，减少重复计算
        decompose_features = {}
        for f_i in self.opt.frame_ids:
            decompose_features[f_i] = self.models["decompose_encoder"](inputs[("color_aug", f_i, 0)])
        
        # 批量分解和重建
        for f_i in self.opt.frame_ids:
            albedo, shading, specular = self.models["decompose"](decompose_features[f_i])
            
            outputs[("albedo", 0, f_i)] = albedo
            outputs[("shading", 0, f_i)] = shading  
            outputs[("specular", 0, f_i)] = specular
            
            # 使用就地操作减少内存分配: I = A × S + R
            outputs[("reprojection_color", 0, f_i)] = torch.addcmul(
                specular, albedo, shading
            )
        
        # Warping operations
        disp = outputs[("disp", 0)]
        disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)
        _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
        
        for i, frame_id in enumerate(self.opt.frame_ids[1:]):
            T = outputs[("cam_T_cam", 0, frame_id)]
            cam_points = self.backproject_depth[0](depth, inputs[("inv_K", 0)])
            pix_coords = self.project_3d[0](cam_points, inputs[("K", 0)], T)

            outputs[("warp", 0, frame_id)] = pix_coords

            # Warp albedo (should be consistent across frames)
            outputs[("albedo_warp", 0, frame_id)] = F.grid_sample(
                outputs[("albedo", 0, frame_id)],
                outputs[("warp", 0, frame_id)],
                padding_mode="border", align_corners=True)
            
            # Warp shading (can change due to lighting)
            outputs[("shading_warp", 0, frame_id)] = F.grid_sample(
                outputs[("shading", 0, frame_id)],
                outputs[("warp", 0, frame_id)],
                padding_mode="border", align_corners=True)
            
            # Warp specular (can change significantly)
            outputs[("specular_warp", 0, frame_id)] = F.grid_sample(
                outputs[("specular", 0, frame_id)],
                outputs[("warp", 0, frame_id)],
                padding_mode="border", align_corners=True)
            
            # Warp original color
            outputs[("color_warp", 0, frame_id)] = F.grid_sample(
                inputs[("color_aug", frame_id, 0)],
                outputs[("warp", 0, frame_id)],
                padding_mode="border", align_corners=True)
            
            # masking zero values
            mask_ones = torch.ones_like(inputs[("color_aug", frame_id, 0)])
            mask_warp = F.grid_sample(
                mask_ones,
                outputs[("warp", 0, frame_id)],
                padding_mode="zeros", align_corners=True)
            valid_mask = (mask_warp.abs().mean(dim=1, keepdim=True) > 0.0).float()
            outputs[("valid_mask", 0, frame_id)] = valid_mask

            # Compute difference for adjustment
            outputs[("warp_diff_color", 0, frame_id)] = (
                torch.abs(inputs[("color_aug", 0, 0)] - outputs[("color_warp", 0, frame_id)]) * valid_mask
            )
            
            # Adjust shading based on illumination changes
            outputs[("transform", 0, frame_id)] = self.models["adjust_net"](
                outputs[("warp_diff_color", 0, frame_id)]
            )
            outputs[("shading_adjust_warp", 0, frame_id)] = (
                outputs[("transform", 0, frame_id)] + outputs[("shading_warp", 0, frame_id)]
            )
            outputs[("shading_adjust_warp", 0, frame_id)] = torch.clamp(
                outputs[("shading_adjust_warp", 0, frame_id)], min=0.0, max=1.0
            )

            # Final reconstruction with adjusted shading
            outputs[("reprojection_color_warp", 0, frame_id)] = (
                outputs[("albedo_warp", 0, frame_id)] * outputs[("shading_adjust_warp", 0, frame_id)] +
                outputs[("specular_warp", 0, frame_id)]
            )

    def compute_reprojection_loss(self, pred, target):
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute losses for non-Lambertian model"""
        losses = {}
        # 使用tensor而不是标量，避免重复的GPU-CPU传输
        device = next(iter(outputs.values())).device
        loss_albedo_consistency = torch.tensor(0.0, device=device)
        loss_reprojection = torch.tensor(0.0, device=device)
        loss_disp_smooth = torch.tensor(0.0, device=device)
        loss_reconstruction = torch.tensor(0.0, device=device)
        loss_specular_smooth = torch.tensor(0.0, device=device)

        # Decomposition-Synthesis Loss: Ensure I = A × S + R
        for frame_id in self.opt.frame_ids:
            loss_reconstruction += (
                self.compute_reprojection_loss(
                    inputs[("color_aug", frame_id, 0)], 
                    outputs[("reprojection_color", 0, frame_id)]
                )
            ).mean()

        # Albedo Consistency Loss: Albedo should be consistent across frames
        for frame_id in self.opt.frame_ids[1:]: 
            mask = outputs[("valid_mask", 0, frame_id)]
            mask_sum = mask.sum()
            if mask_sum > 0:  # 避免除零错误
                loss_albedo_consistency += (
                    torch.abs(outputs[("albedo", 0, 0)] - outputs[("albedo_warp", 0, frame_id)]).mean(1, True) * mask
                ).sum() / mask_sum

        # Mapping-Synthesis Loss: Final reconstruction should match target
        for frame_id in self.opt.frame_ids[1:]:
            mask = outputs[("valid_mask", 0, frame_id)]
            mask_sum = mask.sum()
            if mask_sum > 0:  # 避免除零错误
                loss_reprojection += (
                    self.compute_reprojection_loss(
                        inputs[("color_aug", 0, 0)], 
                        outputs[("reprojection_color_warp", 0, frame_id)]
                    ) * mask
                ).sum() / mask_sum

        # Depth smoothness
        disp = outputs[("disp", 0)]
        color = inputs[("color_aug", 0, 0)]
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        loss_disp_smooth = get_smooth_loss(norm_disp, color)

        # Specular smoothness (specular should be sparse and smooth in non-specular regions)
        for frame_id in self.opt.frame_ids:
            specular = outputs[("specular", 0, frame_id)]
            # Encourage sparsity in specular component
            loss_specular_smooth += torch.mean(specular)

        # Total loss with weights - 使用更高效的计算
        num_frames = len(self.opt.frame_ids[1:])
        num_all_frames = len(self.opt.frame_ids)
        
        total_loss = (
            self.opt.reprojection_constraint * loss_reprojection / max(num_frames, 1) +
            self.opt.albedo_constraint * loss_albedo_consistency / max(num_frames, 1) +
            self.opt.disparity_smoothness * loss_disp_smooth +
            self.opt.reconstruction_constraint * loss_reconstruction / num_all_frames +
            self.opt.specular_smoothness * loss_specular_smooth / num_all_frames
        )

        losses["loss"] = total_loss
        losses["reconstruction_loss"] = loss_reconstruction
        losses["albedo_consistency_loss"] = loss_albedo_consistency
        losses["reprojection_loss"] = loss_reprojection
        losses["specular_smoothness_loss"] = loss_specular_smooth
        
        return losses
    
    def val(self):
        """Validate the model on a single minibatch"""
        self.set_eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)
            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses
            # 清理GPU缓存，避免内存碎片
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.set_train()

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal"""
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file"""
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):
            writer.add_image("disp/{}".format(j), visualize_depth(outputs[("disp", 0)][j]), self.step)
            writer.add_image("input/{}".format(j), inputs[("color", 0, 0)][j].data, self.step)
            
            # Log decomposition components
            if ("albedo", 0, 0) in outputs:
                writer.add_image("albedo/{}".format(j), outputs[("albedo", 0, 0)][j].data, self.step)
            if ("shading", 0, 0) in outputs:
                writer.add_image("shading/{}".format(j), outputs[("shading", 0, 0)][j].data, self.step)
            if ("specular", 0, 0) in outputs:
                writer.add_image("specular/{}".format(j), outputs[("specular", 0, 0)][j].data, self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with"""
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk"""
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk"""
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)