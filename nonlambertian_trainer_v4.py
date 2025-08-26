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


class NonLambertianTrainerV4:
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

        # Use the new non-Lambertian decompose decoder v4
        self.models["decompose_encoder"] = networks.NonLambertianResnetEncoderV4(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["decompose_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["decompose_encoder"].parameters())
        
        self.models["decompose"] = networks.NonLambertianDecomposeDecoderV4(
            self.models["decompose_encoder"].num_ch_enc, self.opt.scales)
        self.models["decompose"].to(self.device)
        self.parameters_to_train += list(self.models["decompose"].parameters())

        self.models["adjust_net_v4"] = networks.adjust_net_v4_lite()
        self.models["adjust_net_v4"].to(self.device)
        self.parameters_to_train += list(self.models["adjust_net_v4"].parameters())

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

        print("Training Non-Lambertian v4 model named:\n  ", self.opt.model_name)
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
            num_workers=min(self.opt.num_workers, 4),  # 减少worker数量
            pin_memory=False,  # 关闭pin_memory减少内存使用
            drop_last=True,
            prefetch_factor=1,  # 减少预取
            persistent_workers=False  # 关闭持久worker
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
            # V4: Progressive specular component importance
            if self.opt.progressive_specular_weight:
                self.current_specular_weight = min(1.0, self.epoch / self.opt.specular_warmup_epochs)
            else:
                self.current_specular_weight = 1.0
            
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation"""
        print("Training")
        print("Learning rate:", self.model_optimizer.param_groups[0]['lr'])
        print("Specular weight:", self.current_specular_weight)
        
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
            
            # 清理内存 - 在日志记录之后
            del outputs, losses
            torch.cuda.empty_cache()

            self.step += 1

        self.model_lr_scheduler.step()

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

        self.set_train()

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses"""
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device, non_blocking=True)

        # depth estimation
        features = self.models["encoder"](inputs["color_aug", 0, 0])
        outputs = self.models["depth"](features)

        # decompose using V4 networks
        decompose_features = self.models["decompose_encoder"](inputs["color_aug", 0, 0])
        outputs[("decompose_A", 0)], outputs[("decompose_S", 0)], outputs[("decompose_R", 0)] = self.models["decompose"](decompose_features)
        
        # Apply adjust net to get final intrinsic images
        outputs[("albedo", 0)], outputs[("shading", 0)], outputs[("specular", 0)] = self.models["adjust_net_v4"](
            outputs[("decompose_A", 0)], outputs[("decompose_S", 0)], outputs[("decompose_R", 0)])

        # pose
        outputs.update(self.predict_poses(inputs, features))

        # reconstruct images
        self.generate_images_pred(inputs, outputs)

        # compute losses
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences."""
        outputs = {}
        if self.num_pose_frames == 2:
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            pose_inputs = torch.cat(
                [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

            pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch."""
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth
            outputs[("color", 0, scale)] = inputs[("color", 0, source_scale)]

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]
                    
    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a prediction and target"""
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch"""
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs["color_aug", 0, source_scale]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    identity_reprojection_loss = identity_reprojection_losses

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss / self.num_scales
            losses["loss/{}".format(scale)] = loss

        # V4: Enhanced Non-Lambertian specific losses
        albedo = outputs[("albedo", 0)]
        shading = outputs[("shading", 0)]
        specular = outputs[("specular", 0)]
        
        # 1. Reconstruction constraint loss: I = A*S + R
        reconstructed = albedo * shading + specular
        target = inputs["color_aug", 0, 0]
        
        loss_decompose_l1 = torch.mean(torch.abs(reconstructed - target))
        loss_decompose_ssim = self.ssim(reconstructed, target).mean()
        loss_decompose = 0.85 * loss_decompose_ssim + 0.15 * loss_decompose_l1
        
        total_loss += self.opt.reconstruction_constraint * loss_decompose
        
        # 2. V4: 极简化的albedo一致性约束 (大幅减少内存使用)
        loss_albedo_consistency = torch.tensor(0.0, device=self.device)
        
        # 只在训练后期且每20个batch启用一次，并且权重很小
        if (hasattr(self, 'epoch') and self.epoch > 10 and 
            hasattr(self, 'step') and self.step % 20 == 0 and
            len(self.opt.frame_ids) > 1):
            
            # 简化的一致性约束：只比较albedo的均值
            try:
                frame_id = self.opt.frame_ids[1]  # 只取第一个非参考帧
                frame_color = inputs["color_aug", frame_id, 0]
                
                # 只处理前2个样本
                frame_color_small = frame_color[:2]
                albedo_small = albedo[:2]
                
                with torch.no_grad():
                    frame_decompose_features = self.models["decompose_encoder"](frame_color_small)
                    frame_albedo, _, _ = self.models["decompose"](frame_decompose_features)
                    frame_albedo_adj, _, _ = self.models["adjust_net_v4"](frame_albedo, 
                                                                         torch.ones_like(frame_albedo[:, :1]), 
                                                                         torch.zeros_like(frame_albedo))
                
                # 只比较全局均值，减少计算量
                loss_albedo_consistency = torch.abs(albedo_small.mean() - frame_albedo_adj.mean())
                
            except Exception as e:
                # 如果出现内存错误，跳过这个约束
                loss_albedo_consistency = torch.tensor(0.0, device=self.device)
        
        total_loss += 0.01 * self.opt.albedo_constraint * loss_albedo_consistency  # 进一步减小权重
        
        # 3. V4: Enhanced specular constraints with progressive weighting
        loss_specular_smooth = torch.mean(specular ** 2)
        loss_specular_l1 = torch.mean(torch.abs(specular))
        
        # V4: Add shading smoothness constraint (shading should be smooth)
        loss_shading_smooth = get_smooth_loss(shading, target)
        
        # V4: Add albedo smoothness constraint (moderate smoothness)
        loss_albedo_smooth = get_smooth_loss(albedo, target)
        
        # Apply progressive weight based on training epoch
        specular_weight = self.current_specular_weight if hasattr(self, 'current_specular_weight') else 1.0
        num_all_frames = len(self.opt.frame_ids)  # 定义缺失的变量
        
        total_loss += specular_weight * self.opt.specular_smoothness * loss_specular_smooth / num_all_frames
        total_loss += specular_weight * self.opt.specular_l1_sparsity * loss_specular_l1 / num_all_frames
        total_loss += self.opt.shading_smoothness * loss_shading_smooth
        total_loss += self.opt.albedo_smoothness * loss_albedo_smooth
        
        losses["loss"] = total_loss
        
        # Record individual losses for logging
        losses["decompose_loss"] = loss_decompose
        losses["albedo_consistency_loss"] = loss_albedo_consistency
        losses["specular_smooth_loss"] = loss_specular_smooth
        losses["specular_l1_loss"] = loss_specular_l1
        losses["shading_smooth_loss"] = loss_shading_smooth
        losses["albedo_smooth_loss"] = loss_albedo_smooth
        
        return losses

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
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id == 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

            # Log decomposed images
            if j < outputs[("albedo", 0)].shape[0]:
                writer.add_image(
                    "decomposed/albedo_{}".format(j),
                    outputs[("albedo", 0)][j].data, self.step)
                writer.add_image(
                    "decomposed/shading_{}".format(j),
                    outputs[("shading", 0)][j].data, self.step)
                writer.add_image(
                    "decomposed/specular_{}".format(j),
                    outputs[("specular", 0)][j].data, self.step)
                writer.add_image(
                    "decomposed/reconstructed_{}".format(j),
                    (outputs[("albedo", 0)] * outputs[("shading", 0)] + outputs[("specular", 0)])[j].data, self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with"""
        models_dir = os.path.join(self.log_path, "models")
        os.makedirs(models_dir, exist_ok=True)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk"""
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        os.makedirs(save_folder, exist_ok=True)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.eval_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "adam.pth")
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

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")