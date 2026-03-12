# built-in libraries
import json
import os
import pickle
import time
from copy import deepcopy

# third party libraries
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# from ml project manager
from mlproj_manager.problems import CifarDataSet
from mlproj_manager.util import access_dict, turn_off_debugging_processes

# project files
from src import (initialize_layer_norm_module, initialize_memory_vit, initialize_mlp_block,
                 initialize_multihead_self_attention_module, initialize_vit, initialize_vit_heads,
                 parse_terminal_arguments)
from src.networks import MemoryVisionTransformer, ReparameterizedLayerNorm, perturb_weights
from src.networks.torchvision_modified_vit import VisionTransformer
from src.swr_functions import SelectiveWeightReinitialization, get_network_init_parameters
from src.utils import IncrementalCIFARExperimentBase
from src.utils import compute_accuracy_from_batch, get_cifar_data
from src.utils import save_model_parameters, set_random_seed
from src.utils.tensorboard_logging import TensorBoardLogger


class IncrementalCIFARExperiment(IncrementalCIFARExperimentBase):

    def __init__(self, exp_params: dict, results_dir: str, run_index: int, verbose=True, gpu_index: int = 0):
        super().__init__(exp_params, results_dir, run_index, verbose)

        # set debugging options for pytorch
        turn_off_debugging_processes(access_dict(exp_params, key="debug", default=True, val_type=bool))
        # define torch device
        gpu_index = access_dict(exp_params, "gpu_index", default=gpu_index, val_type=int)
        self.device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")

        # For reproducibility
        set_random_seed(self.run_index)

        # Experiment parameters
        self.data_path = exp_params["data_path"]

        # problem definition parameters
        self.num_epochs = access_dict(exp_params, "num_epochs", default=1, val_type=int)
        self.current_num_classes = access_dict(exp_params, "initial_num_classes", default=2, val_type=int)
        self.fixed_classes = access_dict(exp_params, "fixed_classes", default=True, val_type=bool)
        self.use_best_network = access_dict(exp_params, "use_best_network", default=True, val_type=bool)
        self.compare_loss = access_dict(exp_params, "compare_loss", default=False, val_type=bool)

        # optimization parameters
        self.stepsize = exp_params["stepsize"]
        self.weight_decay = exp_params["weight_decay"]
        self.rescaled_wd = access_dict(exp_params, "rescaled_wd", default=False, val_type=bool)
        self.wd_on_1d_params = access_dict(exp_params, "wd_on_1d_params", default=True, val_type=bool)
        self.momentum = exp_params["momentum"]
        self.reset_momentum = access_dict(exp_params, "reset_momentum", default=False, val_type=bool)
        self.use_lr_schedule = access_dict(exp_params, "use_lr_schedule", default=True, val_type=bool)
        self.dropout_prob = access_dict(exp_params, "dropout_prob", default=0.05, val_type=float)

        # summary parameters
        self.extended_summaries = access_dict(exp_params, "extended_summaries", default=False, val_type=bool)
        self.use_tensorboard = access_dict(exp_params, "use_tensorboard", default=False, val_type=bool)
        self.tensorboard_log_dir = access_dict(exp_params,
                                               "tensorboard_log_dir",
                                               default=os.path.join(self.results_dir, "tensorboard"),
                                               val_type=str)
        self.tensorboard_flush_secs = access_dict(exp_params, "tensorboard_flush_secs", default=30, val_type=int)
        self.log_rmt_minibatch_losses = access_dict(exp_params,
                                                    "log_rmt_minibatch_losses",
                                                    default=False,
                                                    val_type=bool)
        if self.tensorboard_flush_secs <= 0:
            raise ValueError("tensorboard_flush_secs must be >= 1.")

        # model selection
        self.model_family = access_dict(exp_params, "model_family", default="vit", val_type=str,
                                        choices=["vit", "rmt"])

        # network resetting parameters
        self.reset_head = access_dict(exp_params, "reset_head", default=False, val_type=bool)
        self.reset_network = access_dict(exp_params, "reset_network", default=False, val_type=bool)
        self.reset_layer_norm = access_dict(exp_params, "reset_layer_norm", default=False, val_type=bool)
        self.reset_attention_layers = access_dict(exp_params, "reset_attention_layers", default=False, val_type=bool)
        self.reset_mlp_blocks = access_dict(exp_params, "reset_mlp_blocks", default=False, val_type=bool)

        # other network parameters
        self.reparam_ln = access_dict(exp_params, "reparam_ln", default=False, val_type=bool)

        # SWR parameters
        self.reinit_freq = access_dict(exp_params, "reinit_freq", default=0, val_type=int)
        self.reinit_factor = access_dict(exp_params, "reinit_factor", default=0.0, val_type=float)
        self.utility_function = access_dict(exp_params, "utility_function", default="none", val_type=str,
                                            choices=["none", "magnitude", "gradient"])
        self.pruning_method = access_dict(exp_params, "pruning_method", default="none", val_type=str,
                                          choices=["none", "proportional", "threshold"])
        self.reinit_strat = access_dict(exp_params, "reinit_strat", default="none", val_type=str,
                                        choices=["none", "mean", "resample"])
        self.use_swr = (self.pruning_method != "none") and (self.reinit_strat != "none") and (
            self.reinit_freq > 0) and (self.reinit_factor > 0.0)

        # CBP parameters
        self.replacement_rate = access_dict(exp_params, "replacement_rate", default=None, val_type=float)
        self.maturity_threshold = access_dict(exp_params, "maturity_threshold", default=None, val_type=int)
        self.use_cbp = (self.replacement_rate is not None) and (self.maturity_threshold is not None)

        # ReDO parameters
        self.redo_reinit_frequency = access_dict(exp_params, "redo_reinit_frequency", default=None, val_type=int)
        self.redo_reinit_threshold = access_dict(exp_params, "redo_reinit_threshold", default=None, val_type=float)
        self.use_redo = (self.redo_reinit_frequency is not None) and (self.redo_reinit_threshold is not None)

        # RMT parameters
        self.rmt_variant = access_dict(exp_params, "rmt_variant", default="fast_memory", val_type=str,
                                       choices=["baseline", "fast_memory"])
        self.rmt_patch_size = access_dict(exp_params, "rmt_patch_size", default=4, val_type=int)
        self.rmt_d_model = access_dict(exp_params, "rmt_d_model", default=384, val_type=int)
        self.rmt_n_layers = access_dict(exp_params, "rmt_n_layers", default=8, val_type=int)
        self.rmt_n_heads = access_dict(exp_params, "rmt_n_heads", default=12, val_type=int)
        self.rmt_mlp_ratio = access_dict(exp_params, "rmt_mlp_ratio", default=4.0, val_type=float)
        self.rmt_n_mem = access_dict(exp_params, "rmt_n_mem", default=2, val_type=int)
        self.rmt_fast_lr = access_dict(exp_params, "rmt_fast_lr", default=0.1, val_type=float)
        self.rmt_inner_memory_l2 = access_dict(exp_params, "rmt_inner_memory_l2", default=0.0, val_type=float)
        self.rmt_slow_update_freq = access_dict(exp_params, "rmt_slow_update_freq", default=10, val_type=int)
        self.rmt_memory_reset_at_task_boundary = access_dict(exp_params, "rmt_memory_reset_at_task_boundary",
                                                             default=True, val_type=bool)
        # access_dict enforces exact types for existing keys; allow explicit null in JSON for this optional field.
        clip_grad_value = exp_params["rmt_clip_memory_grad"] if "rmt_clip_memory_grad" in exp_params else None
        if clip_grad_value is not None and not isinstance(clip_grad_value, (float, int)):
            raise ValueError("rmt_clip_memory_grad must be null or a numeric value.")
        self.rmt_clip_memory_grad = None if clip_grad_value is None else float(clip_grad_value)
        if self.rmt_slow_update_freq <= 0:
            raise ValueError("rmt_slow_update_freq must be >= 1.")

        # shrink and perturb parameters
        self.parameter_noise_var = access_dict(exp_params, "parameter_noise_var", default=0.0, val_type=float)
        self.use_parameter_noise = self.parameter_noise_var > 0.0

        # Training constants
        self.batch_sizes = {"train": 90, "test": 100, "validation": 50}
        self.num_classes = 100
        self.image_dims = (32, 32, 3)
        self.num_images_per_epoch = 50000
        self.num_images_per_class = 450
        self.num_workers = 1 if self.device.type == "cpu" else 12

        # Explicitly disable reinitialization methods in RMT mode.
        if self.model_family == "rmt":
            self.use_swr = False
            self.use_cbp = False
            self.use_redo = False

        # Network set up
        self.net = self._build_vit_model() if self.model_family == "vit" else self._build_rmt_model()
        self.net.to(self.device)

        # initialize optimizer and loss function
        self.optim = self._get_optimizer()
        self.lr_scheduler = None
        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")

        # initialize selective weight reinitialization
        self.swr_optim = None
        if self.use_swr:
            means, std, normal_reinit = get_network_init_parameters(self.net, self.reinit_strat, reparam_ln=self.reparam_ln)
            self.swr_optim = SelectiveWeightReinitialization(self.net.parameters(),
                                                             utility_function=self.utility_function,
                                                             pruning_method=self.pruning_method,
                                                             param_means=means,
                                                             param_stds=std,
                                                             normal_reinit=normal_reinit,
                                                             reinit_freq=self.reinit_freq,
                                                             reinit_factor=self.reinit_factor,
                                                             decay_rate=0.0)
        # initialize training counters
        self.current_epoch = 0
        self.current_minibatch = 0

        # For data partitioning
        self.class_increase = access_dict(exp_params, "class_increase", default=5, val_type=int)
        self.class_increase_frequency = access_dict(exp_params, "class_increase_frequency", default=100, val_type=int)
        self.all_classes = np.random.permutation(self.num_classes)
        self.best_accuracy = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        self.best_loss = torch.ones_like(self.best_accuracy) * torch.inf
        self.best_model_parameters = {}

        # RMT state
        self.rmt_memory = None
        self.rmt_global_step = 0
        self.rmt_optimizer_step_count = 0
        self.rmt_running_memory_norm = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        self.rmt_running_memory_update_norm = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        # For creating experiment checkpoints
        self.experiment_checkpoints_dir_path = os.path.join(self.results_dir, "experiment_checkpoints")
        self.checkpoint_identifier_name = "current_epoch"
        self.checkpoint_save_frequency = self.class_increase_frequency
        self.delete_old_checkpoints = True

        # For summaries
        self.running_avg_window = 25
        self.current_running_avg_step, self.running_loss, self.running_accuracy = (0, 0.0, 0.0)
        self._initialize_summaries()
        self._initialize_rmt_summaries_if_needed()
        self.tb_logger = TensorBoardLogger(enabled=self.use_tensorboard,
                                           log_dir=self.tensorboard_log_dir,
                                           run_index=self.run_index,
                                           flush_secs=self.tensorboard_flush_secs)
        self.tb_logger.log_text("run/config", json.dumps(exp_params, indent=2, sort_keys=True, default=str), step=0)

    def _build_vit_model(self) -> VisionTransformer:
        net = VisionTransformer(
            image_size=32,
            patch_size=4,
            num_layers=8,
            num_heads=12,
            hidden_dim=384,
            mlp_dim=1536,
            num_classes=self.num_classes,
            dropout=self.dropout_prob,
            attention_dropout=self.dropout_prob,
            replacement_rate=self.replacement_rate,
            maturity_threshold=self.maturity_threshold,
            reinit_frequency=self.redo_reinit_frequency,
            reinit_threshold=self.redo_reinit_threshold,
            norm_layer=ReparameterizedLayerNorm if self.reparam_ln else torch.nn.LayerNorm,
        )
        initialize_vit(net)
        return net

    def _build_rmt_model(self) -> MemoryVisionTransformer:
        net = MemoryVisionTransformer(
            image_size=32,
            patch_size=self.rmt_patch_size,
            num_layers=self.rmt_n_layers,
            num_heads=self.rmt_n_heads,
            hidden_dim=self.rmt_d_model,
            mlp_dim=int(self.rmt_d_model * self.rmt_mlp_ratio),
            num_classes=self.num_classes,
            n_mem=self.rmt_n_mem,
            dropout=self.dropout_prob,
            attention_dropout=self.dropout_prob,
            norm_layer=ReparameterizedLayerNorm if self.reparam_ln else torch.nn.LayerNorm,
        )
        initialize_memory_vit(net)
        return net

    def _initialize_rmt_summaries_if_needed(self):
        if self.model_family != "rmt" or not self.extended_summaries:
            return
        total_ckpts = self.results_dict["train_loss_per_checkpoint"].shape[0]
        defaults = {"device": self.device, "dtype": torch.float32}
        self.results_dict["rmt_memory_norm_per_checkpoint"] = torch.zeros(total_ckpts, **defaults)
        self.results_dict["rmt_memory_update_norm_per_checkpoint"] = torch.zeros(total_ckpts, **defaults)

    def _initialize_rmt_memory(self, requires_grad: bool = None) -> torch.Tensor:
        if requires_grad is None:
            requires_grad = self.rmt_variant == "fast_memory"
        memory = torch.zeros(self.rmt_n_mem, self.rmt_d_model, device=self.device, dtype=torch.float32)
        return memory.requires_grad_(requires_grad)

    def _store_rmt_extended_summaries(self, memory_norm: torch.Tensor, memory_update_norm: torch.Tensor):
        if not self.extended_summaries:
            return
        self.rmt_running_memory_norm += memory_norm.detach()
        self.rmt_running_memory_update_norm += memory_update_norm.detach()
   # log memory loss and memory loss improvement at each minibatch for fast_memory variant
    def _log_rmt_minibatch_losses(self, memory_loss: torch.Tensor, memory_loss_improvement: torch.Tensor):
        if (not self.log_rmt_minibatch_losses or
                self.tb_logger is None or
                self.model_family != "rmt" or
                self.rmt_variant != "fast_memory"):
            return
        self.tb_logger.log_scalar("rmt/memory_loss_per_minibatch", memory_loss, self.rmt_global_step)
        self.tb_logger.log_scalar("rmt/memory_loss_improvement_per_minibatch",
                                  memory_loss_improvement,
                                  self.rmt_global_step)

    def _store_training_summaries(self):
        super()._store_training_summaries()
        if self.model_family != "rmt" or not self.extended_summaries:
            return
        idx = self.current_running_avg_step - 1
        avg_memory_norm = self.rmt_running_memory_norm / self.running_avg_window
        avg_memory_update_norm = self.rmt_running_memory_update_norm / self.running_avg_window
        self.results_dict["rmt_memory_norm_per_checkpoint"][idx] += avg_memory_norm
        self.results_dict["rmt_memory_update_norm_per_checkpoint"][idx] += avg_memory_update_norm
        if self.tb_logger is not None:
            self.tb_logger.log_scalar("rmt/memory_norm_per_checkpoint", avg_memory_norm, idx)
            self.tb_logger.log_scalar("rmt/memory_update_norm_per_checkpoint", avg_memory_update_norm, idx)
        self.rmt_running_memory_norm *= 0.0
        self.rmt_running_memory_update_norm *= 0.0

    def _get_optimizer(self):
        """ Creates optimizer object based on the experiment parameters """
        if self.wd_on_1d_params:
            wd = self.weight_decay if self.rescaled_wd else self.weight_decay / self.stepsize
            params = self.net.parameters()
            return torch.optim.SGD(params, lr=self.stepsize, momentum=self.momentum, weight_decay=wd)

        param_dict = {pn: p for pn, p in self.net.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [{"params": decay_params, "weight_decay": self.weight_decay},
                        {"params": nodecay_params, "weight_decay": 0.0}]
        return torch.optim.SGD(optim_groups, lr=self.stepsize, momentum=self.momentum)

    def get_experiment_checkpoint(self):
        checkpoint = super().get_experiment_checkpoint()
        if self.model_family != "rmt":
            return checkpoint

        checkpoint["rmt_memory"] = None if self.rmt_memory is None else self.rmt_memory.detach().cpu()
        checkpoint["rmt_global_step"] = self.rmt_global_step
        checkpoint["rmt_optimizer_step_count"] = self.rmt_optimizer_step_count
        checkpoint["rmt_running_memory_norm"] = self.rmt_running_memory_norm.detach().cpu()
        checkpoint["rmt_running_memory_update_norm"] = self.rmt_running_memory_update_norm.detach().cpu()
        return checkpoint

    def load_checkpoint_data_and_update_experiment_variables(self, file_path):
        super().load_checkpoint_data_and_update_experiment_variables(file_path)
        if self.model_family != "rmt":
            return

        with open(file_path, mode="rb") as experiment_checkpoint_file:
            checkpoint = pickle.load(experiment_checkpoint_file)

        rmt_memory = checkpoint.get("rmt_memory", None)
        if rmt_memory is not None:
            requires_grad = self.rmt_variant == "fast_memory"
            self.rmt_memory = rmt_memory.to(self.device).detach().requires_grad_(requires_grad)
        else:
            self.rmt_memory = None

        self.rmt_global_step = checkpoint.get("rmt_global_step", 0)
        self.rmt_optimizer_step_count = checkpoint.get("rmt_optimizer_step_count", 0)
        self.rmt_running_memory_norm = checkpoint.get(
            "rmt_running_memory_norm", torch.tensor(0.0, dtype=torch.float32)
        ).to(self.device)
        self.rmt_running_memory_update_norm = checkpoint.get(
            "rmt_running_memory_update_norm", torch.tensor(0.0, dtype=torch.float32)
        ).to(self.device)

    @torch.no_grad()
    def _evaluate_rmt_network(self, data_loader: DataLoader, memory_tokens: torch.Tensor):
        avg_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        avg_acc = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        num_batches = 0
        active_classes = self.all_classes[:self.current_num_classes]

        frozen_memory = memory_tokens.detach()
        for sample in data_loader:
            images = sample["image"].to(self.device)
            labels = sample["label"].to(self.device)
            preds = self.net.forward(images, frozen_memory)[:, active_classes]
            avg_loss += self.loss(preds, labels)
            avg_acc += compute_accuracy_from_batch(preds, labels)
            num_batches += 1

        return avg_loss / num_batches, avg_acc / num_batches

    def _store_test_summaries(self, test_data: DataLoader, val_data: DataLoader, epoch_number: int,
                              epoch_runtime: float):
        if self.model_family == "vit":
            super()._store_test_summaries(test_data, val_data, epoch_number, epoch_runtime)
            return

        self.results_dict["epoch_runtime"][epoch_number] += torch.tensor(epoch_runtime, dtype=torch.float32)
        if self.tb_logger is not None:
            self.tb_logger.log_scalar("epoch/runtime_sec", epoch_runtime, epoch_number)
            self.tb_logger.log_scalar("task/current_num_classes", self.current_num_classes, epoch_number)
        eval_memory = self.rmt_memory.detach().clone() if self.rmt_memory is not None else self._initialize_rmt_memory(False)

        self.net.eval()
        for data_name, data_loader, compare_to_best in [("test", test_data, False), ("validation", val_data, True)]:
            evaluation_start_time = time.perf_counter()
            loss, accuracy = self._evaluate_rmt_network(data_loader, eval_memory)
            evaluation_time = time.perf_counter() - evaluation_start_time

            if compare_to_best:
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    if not self.compare_loss:
                        self.best_model_parameters = deepcopy(self.net.state_dict())
                if loss < self.best_loss:
                    self.best_loss = loss
                    if self.compare_loss:
                        self.best_model_parameters = deepcopy(self.net.state_dict())

            self.results_dict[data_name + "_evaluation_runtime"][epoch_number] += torch.tensor(evaluation_time, dtype=torch.float32)
            self.results_dict[data_name + "_loss_per_epoch"][epoch_number] += loss
            self.results_dict[data_name + "_accuracy_per_epoch"][epoch_number] += accuracy
            if self.tb_logger is not None:
                self.tb_logger.log_scalar(f"{data_name}/eval_runtime_sec", evaluation_time, epoch_number)
                self.tb_logger.log_scalar(f"{data_name}/loss_per_epoch", loss, epoch_number)
                self.tb_logger.log_scalar(f"{data_name}/accuracy_per_epoch", accuracy, epoch_number)
            self._print(f"\t{data_name} accuracy: {accuracy:.4f}")

        self.net.train()

    def run(self):
        training_data, training_dl = get_cifar_data(self.data_path, train=True, validation=False,
                                                    batch_size=self.batch_sizes["train"], num_workers=self.num_workers)
        val_data, val_dl = get_cifar_data(self.data_path, train=True, validation=True,
                                          batch_size=self.batch_sizes["validation"], num_workers=self.num_workers)
        test_data, test_dl = get_cifar_data(self.data_path, train=False, batch_size=self.batch_sizes["test"],
                                            num_workers=self.num_workers)
        self.load_experiment_checkpoint()
        self.train(train_dataloader=training_dl, test_dataloader=test_dl, val_dataloader=val_dl,
                   test_data=test_data, training_data=training_data, val_data=val_data)
        self.post_process_results()

    def train(self, train_dataloader: DataLoader, test_dataloader: DataLoader, val_dataloader: DataLoader,
              test_data: CifarDataSet, training_data: CifarDataSet, val_data: CifarDataSet):
        training_data.select_new_partition(self.all_classes[:self.current_num_classes])
        test_data.select_new_partition(self.all_classes[:self.current_num_classes])
        val_data.select_new_partition(self.all_classes[:self.current_num_classes])

        if self.use_lr_schedule:
            if self.model_family == "rmt":
                effective_steps = int(np.ceil(len(train_dataloader) / self.rmt_slow_update_freq))
                self.lr_scheduler = self.get_lr_scheduler(steps_per_epoch=max(1, effective_steps))
            else:
                self.lr_scheduler = self.get_lr_scheduler(steps_per_epoch=len(train_dataloader))
        save_model_parameters(self.results_dir, self.run_index, self.current_epoch, self.net)

        for e in range(self.current_epoch, self.num_epochs):
            self._print(f"Epoch: {e + 1}")

            epoch_start = time.perf_counter()
            if self.model_family == "vit":
                self._train_vit(train_dataloader)
            else:
                self._train_rmt(train_dataloader)
            epoch_end = time.perf_counter()

            self._store_test_summaries(test_dataloader, val_dataloader, epoch_number=e, epoch_runtime=epoch_end - epoch_start)
            self.current_epoch += 1

            self.extend_classes(training_data, test_data, val_data, train_dataloader)

            if self.current_epoch % self.checkpoint_save_frequency == 0:
                self.save_experiment_checkpoint()

    def _train_vit(self, train_dataloader: DataLoader):
        for step_number, sample in enumerate(tqdm(train_dataloader)):
            image = sample["image"].to(self.device)
            label = sample["label"].to(self.device)

            for param in self.net.parameters():
                param.grad = None

            predictions = self.net.forward(image)[:, self.all_classes[:self.current_num_classes]]
            current_loss = self.loss(predictions, label)
            detached_loss = current_loss.detach().clone()

            current_loss.backward()
            self.optim.step()
            if self.use_lr_schedule:
                self.lr_scheduler.step()
                if self.lr_scheduler.get_last_lr()[0] > 0.0 and not self.rescaled_wd:
                    self.optim.param_groups[0]["weight_decay"] = self.weight_decay / self.lr_scheduler.get_last_lr()[0]
            if self.swr_optim is not None:
                self.swr_optim.step()
                self.store_num_replaced()
            if self.use_parameter_noise:
                perturb_weights(self.net, self.parameter_noise_var)

            current_accuracy = compute_accuracy_from_batch(predictions, label)
            self.running_loss += detached_loss
            self.running_accuracy += current_accuracy.detach()
            if (step_number + 1) % self.running_avg_window == 0:
                self._store_training_summaries()

            self.current_minibatch += 1

    def _train_rmt(self, train_dataloader: DataLoader):
        if self.rmt_memory is None:
            self.rmt_memory = self._initialize_rmt_memory()

        active_classes = self.all_classes[:self.current_num_classes]
        self.optim.zero_grad(set_to_none=True)
        for step_number, sample in enumerate(tqdm(train_dataloader)):
            image = sample["image"].to(self.device)
            label = sample["label"].to(self.device)

            if self.rmt_variant == "baseline":
                predictions, encoded_memory = self.net.forward(image, self.rmt_memory, return_encoded_memory=True)
                predictions = predictions[:, active_classes]
                current_loss = self.loss(predictions, label)
                detached_loss = current_loss.detach().clone()
                current_loss.backward()

                with torch.no_grad():
                    next_memory = encoded_memory.detach().mean(dim=0)
                    memory_update_norm = (next_memory - self.rmt_memory.detach()).norm()
                    self.rmt_memory = next_memory
            else:
                # Inner (fast) update: compute memory gradient only.
                if not self.rmt_memory.requires_grad:
                    self.rmt_memory = self.rmt_memory.detach().requires_grad_(True)

                # Detach the loss to prevent gradients from flowing back through the memory update into the model parameters during the
                # fast update step. This ensures that the inner update optimizes the memory tokens for the current model parameters 
                # without affecting the model parameters themselves, which are only updated during the outer loop.
                inner_predictions = self.net.forward(image, self.rmt_memory)[:, active_classes]
                inner_loss = self.loss(inner_predictions, label)
                if self.rmt_inner_memory_l2 > 0.0:
                    inner_loss = inner_loss + 0.5 * self.rmt_inner_memory_l2 * torch.mean(self.rmt_memory ** 2)
                memory_loss = inner_loss.detach().clone()
                memory_grad = torch.autograd.grad(
                    inner_loss, self.rmt_memory, retain_graph=False, create_graph=False, allow_unused=True
                )[0]

                memory_update_norm = torch.tensor(0.0, device=self.device, dtype=torch.float32)
                with torch.no_grad():
                    if memory_grad is not None:
                        if self.rmt_clip_memory_grad is not None:
                            grad_norm = memory_grad.norm()
                            if grad_norm > self.rmt_clip_memory_grad:
                                scaling = self.rmt_clip_memory_grad / (grad_norm + 1e-12)
                                memory_grad = memory_grad * scaling
                        memory_step = self.rmt_fast_lr * memory_grad
                        self.rmt_memory = (self.rmt_memory - memory_step).detach()
                        memory_update_norm = memory_step.norm()
                    else:
                        self.rmt_memory = self.rmt_memory.detach()

                # Outer (slow) update: compute model gradients using adapted memory.
                predictions = self.net.forward(image, self.rmt_memory)[:, active_classes]
                current_loss = self.loss(predictions, label)
                detached_loss = current_loss.detach().clone()
                post_memory_loss = detached_loss
                if self.rmt_inner_memory_l2 > 0.0:
                    post_memory_loss = post_memory_loss + 0.5 * self.rmt_inner_memory_l2 * torch.mean(
                        self.rmt_memory.detach() ** 2
                    )
                self._log_rmt_minibatch_losses(memory_loss=memory_loss,
                                               memory_loss_improvement=memory_loss - post_memory_loss)
                current_loss.backward()

            # Slow updates must be scheduled in epoch-local coordinates so the
            # optimizer/scheduler step counts match steps_per_epoch exactly.
            do_slow_step = ((step_number + 1) % self.rmt_slow_update_freq) == 0
            if do_slow_step:
                self.optim.step()
                self.rmt_optimizer_step_count += 1
                if self.use_lr_schedule:
                    self.lr_scheduler.step()
                    if self.lr_scheduler.get_last_lr()[0] > 0.0 and not self.rescaled_wd:
                        self.optim.param_groups[0]["weight_decay"] = self.weight_decay / self.lr_scheduler.get_last_lr()[0]
                if self.use_parameter_noise:
                    perturb_weights(self.net, self.parameter_noise_var)
                self.optim.zero_grad(set_to_none=True)

            current_accuracy = compute_accuracy_from_batch(predictions, label)
            self.running_loss += detached_loss
            self.running_accuracy += current_accuracy.detach()
            self._store_rmt_extended_summaries(memory_norm=self.rmt_memory.detach().norm(), memory_update_norm=memory_update_norm)
            if (step_number + 1) % self.running_avg_window == 0:
                self._store_training_summaries()

            self.current_minibatch += 1
            self.rmt_global_step += 1

        # Flush trailing accumulated gradients when the number of batches is not divisible by the slow update frequency.
        has_pending_grads = any(p.grad is not None for p in self.net.parameters())
        if has_pending_grads:
            self.optim.step()
            self.rmt_optimizer_step_count += 1
            if self.use_lr_schedule:
                self.lr_scheduler.step()
                if self.lr_scheduler.get_last_lr()[0] > 0.0 and not self.rescaled_wd:
                    self.optim.param_groups[0]["weight_decay"] = self.weight_decay / self.lr_scheduler.get_last_lr()[0]
            if self.use_parameter_noise:
                perturb_weights(self.net, self.parameter_noise_var)
            self.optim.zero_grad(set_to_none=True)

    def get_lr_scheduler(self, steps_per_epoch: int):
        scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optim, max_lr=self.stepsize, anneal_strategy="linear",
                                                        epochs=self.class_increase_frequency,
                                                        steps_per_epoch=steps_per_epoch)
        if not self.rescaled_wd:
            self.optim.param_groups[0]["weight_decay"] = self.weight_decay / scheduler.get_last_lr()[0]
        return scheduler

    def extend_classes(self, training_data: CifarDataSet, test_data: CifarDataSet, val_data: CifarDataSet,
                       train_dataloader: DataLoader):
        """
        Adds new classes to the data set with the configured frequency.
        """
        if (self.current_epoch % self.class_increase_frequency) == 0 and (not self.fixed_classes):
            self._print("Best accuracy in the task: {0:.4f}".format(self.best_accuracy))
            if self.use_best_network:
                self.net.load_state_dict(self.best_model_parameters)
            self.best_accuracy = torch.zeros_like(self.best_accuracy)
            self.best_loss = torch.ones_like(self.best_accuracy) * torch.inf
            self.best_model_parameters = {}
            save_model_parameters(self.results_dir, self.run_index, self.current_epoch, self.net)

            if self.current_num_classes == self.num_classes:
                return

            self.current_num_classes += self.class_increase
            training_data.select_new_partition(self.all_classes[:self.current_num_classes])
            test_data.select_new_partition(self.all_classes[:self.current_num_classes])
            val_data.select_new_partition(self.all_classes[:self.current_num_classes])

            self._print("\tNew class added...")
            if self.reset_head:
                if self.model_family == "vit":
                    initialize_vit_heads(self.net.heads)
                else:
                    torch.nn.init.zeros_(self.net.head.weight)
                    torch.nn.init.zeros_(self.net.head.bias)
            if self.reset_network:
                if self.model_family == "vit":
                    initialize_vit(self.net)
                else:
                    initialize_memory_vit(self.net)
                    if self.rmt_memory_reset_at_task_boundary:
                        self.rmt_memory = self._initialize_rmt_memory()
                self.optim = self._get_optimizer()
            if self.reset_layer_norm:
                self.net.apply(initialize_layer_norm_module)
            if self.reset_attention_layers:
                self.net.apply(initialize_multihead_self_attention_module)
            if self.reset_mlp_blocks:
                self.net.apply(initialize_mlp_block)
            if self.reset_momentum:
                self.optim = self._get_optimizer()
            if self.use_lr_schedule:
                if self.model_family == "rmt":
                    effective_steps = int(np.ceil(len(train_dataloader) / self.rmt_slow_update_freq))
                    self.lr_scheduler = self.get_lr_scheduler(steps_per_epoch=max(1, effective_steps))
                else:
                    self.lr_scheduler = self.get_lr_scheduler(steps_per_epoch=len(train_dataloader))
            if self.model_family == "rmt" and self.rmt_memory_reset_at_task_boundary:
                self.rmt_memory = self._initialize_rmt_memory()
            return True
        return False


def main():
    """
    Function for running the experiment from command line given a path to a json config file
    """
    from mlproj_manager.file_management.file_and_directory_management import read_json_file
    terminal_arguments = parse_terminal_arguments()
    experiment_parameters = read_json_file(terminal_arguments.config_file)
    file_path = os.path.dirname(os.path.abspath(__file__))
    if terminal_arguments.tensorboard:
        experiment_parameters["use_tensorboard"] = True
    if terminal_arguments.tensorboard_log_dir is not None:
        experiment_parameters["tensorboard_log_dir"] = terminal_arguments.tensorboard_log_dir

    experiment_parameters["data_path"] = os.path.join(file_path, "data")
    print(experiment_parameters)
    relevant_parameters = experiment_parameters["relevant_parameters"]
    results_dir_name = "{0}-{1}".format(relevant_parameters[0], experiment_parameters[relevant_parameters[0]])
    for relevant_param in relevant_parameters[1:]:
        results_dir_name += "_" + relevant_param + "-" + str(experiment_parameters[relevant_param])

    results_path = os.path.join(file_path, "results") if "results_path" not in experiment_parameters.keys() else experiment_parameters["results_path"]

    initial_time = time.perf_counter()
    exp = IncrementalCIFARExperiment(experiment_parameters,
                                     results_dir=os.path.join(results_path, results_dir_name),
                                     run_index=terminal_arguments.run_index,
                                     verbose=terminal_arguments.verbose,
                                     gpu_index=terminal_arguments.gpu_index)
    try:
        exp.run()
        exp.store_results()
    finally:
        if getattr(exp, "tb_logger", None) is not None:
            exp.tb_logger.close()
    final_time = time.perf_counter()
    print("The running time in minutes is: {0:.2f}".format((final_time - initial_time) / 60))


if __name__ == "__main__":
    main()
