# built-in libraries
import json
import time
import os

# third party libraries
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# from ml project manager
from mlproj_manager.problems import MnistDataSet
from mlproj_manager.util import access_dict, Permute, turn_off_debugging_processes

# from src
from src.networks import MemoryTransformer, ThreeHiddenLayerNetwork, init_three_hidden_layer_network_weights, perturb_weights
from src.utils.experiment_utils import parse_terminal_arguments
from src.utils.evaluation_functions import compute_average_gradient_magnitude, set_random_seed
from src.utils.permuted_mnist_experiment_utils import PermutedMNISTExperimentBase
from src.swr_functions import SelectiveWeightReinitialization, get_network_init_parameters
from src.optimizers import SGDW
from src.utils.tensorboard_logging import TensorBoardLogger


class PermutedMNISTExperiment(PermutedMNISTExperimentBase):

    def __init__(self, exp_params: dict, results_dir: str, run_index: int, verbose=False, gpu_index: int = 0):
        super().__init__(exp_params, results_dir, run_index, verbose=verbose)

        # set debugging options for pytorch
        debug = access_dict(exp_params, key="debug", default=True, val_type=bool)
        turn_off_debugging_processes(debug)

        # define torch device
        gpu_index = access_dict(exp_params, "gpu_index", default=gpu_index, val_type=int)
        self.device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")

        """ For reproducibility """
        set_random_seed(self.run_index)

        """ Experiment parameters """
        self.extended_summaries = access_dict(exp_params, "extended_summaries", default=False, val_type=bool)
        self.use_tensorboard = access_dict(exp_params, "use_tensorboard", default=False, val_type=bool)
        self.tensorboard_log_dir = access_dict(exp_params,
                                               "tensorboard_log_dir",
                                               default=os.path.join(results_dir, "tensorboard"),
                                               val_type=str)
        self.tensorboard_flush_secs = access_dict(exp_params, "tensorboard_flush_secs", default=30, val_type=int)
        self.log_rmt_minibatch_losses = access_dict(exp_params,
                                                    "log_rmt_minibatch_losses",
                                                    default=False,
                                                    val_type=bool)
        if self.tensorboard_flush_secs <= 0:
            raise ValueError("tensorboard_flush_secs must be >= 1.")
        self.model_family = access_dict(exp_params, "model_family", default="mlp", val_type=str, choices=["mlp", "rmt"])
        # learning parameters
        self.stepsize = exp_params["stepsize"]
        self.l2_factor = access_dict(exp_params, "l2_factor", default=0.0, val_type=float)
        self.use_adamw = access_dict(exp_params, "use_adamw", default=False, val_type=bool)
        self.momentum = access_dict(exp_params, "momentum", default=0.0, val_type=float)
        self.beta2 = access_dict(exp_params, "beta2", default=None, val_type=float)

        """ Architecture parameters """
        self.num_hidden = exp_params["num_hidden"] if self.model_family == "mlp" else access_dict(exp_params, "num_hidden", default=100, val_type=int)
        self.activation_function = access_dict(exp_params, "activation_function", default="relu", val_type=str,
                                               choices=["relu", "sigmoid", "tanh", "leaky_relu", "gelu", "silu"])
        self.use_crelu = access_dict(exp_params, "use_crelu", default=False, val_type=bool)
        if self.use_crelu and self.activation_function != "relu":
            raise ValueError("The use_crelu parameter is only valid when the activation function is relu.")
        # layer norm parameters
        self.use_ln = access_dict(exp_params, "use_ln", default=False, val_type=bool)
        self.preactivation_ln = access_dict(exp_params, "preactivation_ln", default=False, val_type=bool)
        # residual network parameters
        self.use_skip_connections = access_dict(exp_params, "use_skip_connections", default=False, val_type=bool)
        self.preactivation_skip_connections = access_dict(exp_params, "preactivation_skip_connections", default=False, val_type=bool)

        # problem parameters
        self.num_permutations = exp_params["num_permutations"]      # 1 permutation = 1 epoch
        self.batch_size = access_dict(exp_params, "batch_size", default=30, val_type=int)
        self.steps_per_task = 60000
        self.current_experiment_step = 0

        # RMT parameters
        self.rmt_variant = access_dict(exp_params, "rmt_variant", default="fast_memory", val_type=str,
                                       choices=["baseline", "fast_memory"])
        self.rmt_patch_size = access_dict(exp_params, "rmt_patch_size", default=4, val_type=int)
        self.rmt_d_model = access_dict(exp_params, "rmt_d_model", default=64, val_type=int)
        self.rmt_n_layers = access_dict(exp_params, "rmt_n_layers", default=2, val_type=int)
        self.rmt_n_heads = access_dict(exp_params, "rmt_n_heads", default=4, val_type=int)
        self.rmt_mlp_ratio = access_dict(exp_params, "rmt_mlp_ratio", default=2.0, val_type=float)
        self.rmt_n_mem = access_dict(exp_params, "rmt_n_mem", default=2, val_type=int)
        self.rmt_fast_lr = access_dict(exp_params, "rmt_fast_lr", default=0.1, val_type=float)
        self.rmt_slow_update_freq = access_dict(exp_params, "rmt_slow_update_freq", default=10, val_type=int)
        self.rmt_memory_reset_at_task_boundary = access_dict(exp_params, "rmt_memory_reset_at_task_boundary", default=True, val_type=bool)
        self.rmt_clip_memory_grad = access_dict(exp_params, "rmt_clip_memory_grad", default=None, val_type=float)
        if self.rmt_slow_update_freq <= 0:
            raise ValueError("rmt_slow_update_freq must be >= 1.")
        self.rmt_memory = None
        self.rmt_global_step = 0
        self.rmt_optimizer_step_count = 0
        self.rmt_running_memory_norm = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        self.rmt_running_memory_update_norm = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        """ Reinitialization parameters """
        # SWR parameters
        self.reinit_freq = access_dict(exp_params, "reinit_freq", default=0, val_type=int)
        self.reinit_factor = access_dict(exp_params, "reinit_factor", default=0.0, val_type=float)
        self.utility_function = access_dict(exp_params, "utility_function", default="none", val_type=str, choices=["none", "magnitude", "gradient"])
        self.pruning_method = access_dict(exp_params, "pruning_method", default="none", val_type=str, choices=["none", "proportional", "threshold"])
        self.reinit_strat = access_dict(exp_params, "reinit_strat", default="none", val_type=str, choices=["none", "resample", "mean", "random"])
        self.use_swr = (self.pruning_method != "none") and (self.reinit_strat != "none") and (self.reinit_freq > 0) and (self.reinit_factor > 0.0)
        # cbp parameters
        feature_utility_names = ["none", "contribution", "magnitude", "gradient", "activation"]
        self.maturity_threshold = access_dict(exp_params, "maturity_threshold", default=None, val_type=int)
        self.replacement_rate = access_dict(exp_params, "replacement_rate", default=None, val_type=float)
        self.cbp_utility = access_dict(exp_params, "cbp_utility", default=None, val_type=str, choices=feature_utility_names)
        self.use_cbp = (self.maturity_threshold is not None) and (self.replacement_rate is not None) and (self.cbp_utility is not None)
        # redo parameters
        self.redo_reinit_freq = access_dict(exp_params, "redo_reinit_freq", default=None, val_type=int)
        self.redo_reinit_threshold = access_dict(exp_params, "redo_reinit_threshold", default=None, val_type=float)
        self.redo_utility = access_dict(exp_params, "redo_utility", default=None, val_type=str, choices=feature_utility_names)
        self.use_redo = (self.redo_reinit_freq is not None) and (self.redo_reinit_threshold is not None) and (self.redo_utility is not None)
        # for both cbp and redo
        self.reinit_after_ln = access_dict(exp_params, "reinit_after_ln", default=False, val_type=bool)
        if self.model_family == "rmt":
            # Explicitly disable CBP/ReDo/SWR for RMT runs.
            self.use_swr = False
            self.use_cbp = False
            self.use_redo = False
        # Shrink and Perturb parameters
        self.parameter_noise_var = access_dict(exp_params, "parameter_noise_var", default=0.0, val_type=float)
        self.use_parameter_noise = self.parameter_noise_var > 0.0
        # paths for loading and storing data
        self.use_reinit = self.use_swr or self.use_cbp or self.use_redo
        self.data_path = exp_params["data_path"]
        self.results_dir = results_dir

        """ Training constants """
        self.num_classes = 10
        self.num_inputs = 784

        """ Network set up """
        self.net = self._build_mlp_model() if self.model_family == "mlp" else self._build_rmt_model()
        self.net.to(self.device)
        # initialize selective weight reinitialization
        self.swr_optim = None
        if self.use_swr:
            means, std, normal_reinit = get_network_init_parameters(self.net, self.reinit_strat, reparam_ln=False)
            self.swr_optim = SelectiveWeightReinitialization(self.net.parameters(),
                                                             utility_function=self.utility_function,
                                                             pruning_method=self.pruning_method,
                                                             param_means=means,
                                                             param_stds=std,
                                                             normal_reinit=normal_reinit,
                                                             reinit_freq=self.reinit_freq,
                                                             reinit_factor=self.reinit_factor,
                                                             decay_rate=0.0)
        # initialize optimizer
        self.optim = self.get_optimizer()
        # define loss function
        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")

        """ Experiment Summaries """
        self.running_avg_window = 10
        self.store_next_loss = False        # indicates whether to store the loss computed on the next batch
        self.current_running_avg_step, self.running_loss, self.running_accuracy, self.current_permutation = (0, 0.0, 0.0, 0)
        self.running_avg_grad_magnitude = 0.0
        self.previous_activations = []
        self.results_dict = self.initialize_results_dict()
        self._initialize_rmt_summaries_if_needed()
        self.tb_logger = TensorBoardLogger(enabled=self.use_tensorboard,
                                           log_dir=self.tensorboard_log_dir,
                                           run_index=self.run_index,
                                           flush_secs=self.tensorboard_flush_secs)
        self.tb_logger.log_text("run/config", json.dumps(exp_params, indent=2, sort_keys=True, default=str), step=0)

    def _build_mlp_model(self):
        net = ThreeHiddenLayerNetwork(hidden_dim=self.num_hidden,
                                      activation_function=self.activation_function,
                                      use_skip_connections=self.use_skip_connections,
                                      preactivation_skip_connection=self.preactivation_skip_connections,
                                      use_cbp=self.use_cbp,
                                      maturity_threshold=self.maturity_threshold,
                                      replacement_rate=self.replacement_rate,
                                      cbp_utility=self.cbp_utility,
                                      use_redo=self.use_redo,
                                      reinit_frequency=self.redo_reinit_freq,
                                      reinit_threshold=self.redo_reinit_threshold,
                                      redo_utility=self.redo_utility,
                                      use_layer_norm=self.use_ln,
                                      preactivation_layer_norm=self.preactivation_ln,
                                      reinit_after_ln=self.reinit_after_ln,
                                      use_crelu=self.use_crelu)
        net.apply(lambda z: init_three_hidden_layer_network_weights(z, nonlinearity=self.activation_function))
        return net

    def _build_rmt_model(self):
        return MemoryTransformer(
            patch_size=self.rmt_patch_size,
            d_model=self.rmt_d_model,
            n_layers=self.rmt_n_layers,
            n_heads=self.rmt_n_heads,
            mlp_ratio=self.rmt_mlp_ratio,
            n_mem=self.rmt_n_mem,
            num_classes=self.num_classes,
            img_size=28,
        )

    def _initialize_rmt_summaries_if_needed(self):
        if self.model_family != "rmt" or not self.extended_summaries:
            return
        defaults = {"device": self.device, "dtype": torch.float32}
        total_ckpts = self.results_dict["train_loss_per_checkpoint"].shape[0]
        self.results_dict["rmt_memory_norm_per_checkpoint"] = torch.zeros(total_ckpts, **defaults)
        self.results_dict["rmt_memory_update_norm_per_checkpoint"] = torch.zeros(total_ckpts, **defaults)

    def _initialize_rmt_memory(self):
        memory = torch.zeros(self.rmt_n_mem, self.rmt_d_model, device=self.device, dtype=torch.float32)
        if self.rmt_variant == "fast_memory":
            memory.requires_grad_(True)
        return memory

    def _prepare_rmt_images(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(self.device)
        if image.ndim == 2:
            if image.shape[1] != self.num_inputs:
                raise ValueError(f"Expected image shape (B, {self.num_inputs}), got {tuple(image.shape)}")
            image = image.reshape(image.shape[0], 1, 28, 28)
        elif image.ndim == 3:
            if image.shape[1:] != (28, 28):
                raise ValueError(f"Expected image shape (B, 28, 28), got {tuple(image.shape)}")
            image = image.unsqueeze(1)
        elif image.ndim == 4:
            if image.shape[1:] == (1, 28, 28):
                pass
            elif image.shape[1:] == (28, 28, 1):
                image = image.permute(0, 3, 1, 2)
            else:
                raise ValueError(f"Expected image shape (B, 1, 28, 28) or (B, 28, 28, 1), got {tuple(image.shape)}")
        else:
            raise ValueError(f"Unsupported image shape: {tuple(image.shape)}")
        return image.to(torch.float32)

    def _prepare_rmt_labels(self, label: torch.Tensor) -> torch.Tensor:
        label = label.to(self.device)
        if label.ndim == 1:
            return label.to(torch.long)
        if label.ndim == 2:
            return label.argmax(dim=1).to(torch.long)
        raise ValueError(f"Unsupported label shape: {tuple(label.shape)}")

    def _store_rmt_extended_summaries(self, memory_norm: torch.Tensor, memory_update_norm: torch.Tensor):
        if not self.extended_summaries:
            return
        self.running_avg_grad_magnitude += compute_average_gradient_magnitude(self.net)
        self.rmt_running_memory_norm += memory_norm.detach()
        self.rmt_running_memory_update_norm += memory_update_norm.detach()

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
        if self.model_family == "rmt" and self.extended_summaries:
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

    def _move_results_to_cpu(self):
        """
        Ensure all torch tensors in results are CPU tensors so numpy conversion/storage is safe.
        """
        for key, value in self.results_dict.items():
            if isinstance(value, torch.Tensor):
                self.results_dict[key] = value.detach().cpu()
            elif isinstance(value, list):
                contains_tensor = any(isinstance(v, torch.Tensor) for v in value)
                if contains_tensor:
                    self.results_dict[key] = [v.detach().cpu() if isinstance(v, torch.Tensor) else v for v in value]

    def get_optimizer(self):
        if self.use_adamw:
            betas = (self.momentum, self.momentum) if self.beta2 is None else (self.momentum, self.beta2)
            return torch.optim.AdamW(self.net.parameters(), lr=self.stepsize, weight_decay=self.l2_factor/self.stepsize, betas=betas)
        else:
            return SGDW(self.net.parameters(), lr=self.stepsize, weight_decay=self.l2_factor/self.stepsize, momentum=self.momentum)

    # --------------------------- For running the experiment --------------------------- #
    def run(self):
        # load data
        mnist_train_data = MnistDataSet(root_dir=self.data_path, train=True, device=self.device,
                                        image_normalization="max", label_preprocessing="one-hot", use_torch=True)
        mnist_data_loader = DataLoader(mnist_train_data, batch_size=self.batch_size, shuffle=True)

        # train network
        self.train(mnist_data_loader=mnist_data_loader, training_data=mnist_train_data)
        self.post_process_extended_results()
        self._move_results_to_cpu()
        print(float(self.results_dict["train_accuracy_per_checkpoint"].mean()))

    def train(self, mnist_data_loader: DataLoader, training_data: MnistDataSet):
        if self.model_family == "rmt":
            self._train_rmt(mnist_data_loader=mnist_data_loader, training_data=training_data)
        else:
            self._train_mlp(mnist_data_loader=mnist_data_loader, training_data=training_data)

    def _train_mlp(self, mnist_data_loader: DataLoader, training_data: MnistDataSet):
        for _ in tqdm(range(self.num_permutations), disable=not self.verbose):
            if self.current_permutation == self.num_permutations:
                break
            training_data.set_transformation(Permute(np.random.permutation(self.num_inputs)))  # apply new permutation

            # compute percent of dead units, stable rank, and average weight magnitude
            self.compute_network_extended_summaries(mnist_data_loader)

            # train for one task
            for i, sample in enumerate(mnist_data_loader):
                self.current_experiment_step += 1

                # sample observation and target
                image = sample["image"].reshape(self.batch_size, self.num_inputs)
                label = sample["label"]

                # reset gradients
                for param in self.net.parameters():
                    param.grad = None  # apparently faster than optim.zero_grad()

                # compute prediction and loss
                current_activations = [] if (self.extended_summaries and self.use_ln and (self.use_cbp or self.use_swr or self.use_redo)) else None
                predictions = self.net.forward(image, current_activations)
                current_loss = self.loss(predictions, label)
                detached_loss = current_loss.detach().clone()

                # backpropagate, update weights, use swr, and perturb weights
                current_loss.backward()
                self.optim.step()
                if self.swr_optim is not None:
                    self.swr_optim.step()
                if self.use_parameter_noise:
                    perturb_weights(self.net, self.parameter_noise_var)

                # store extended summaries
                if self.extended_summaries:
                    self.running_avg_grad_magnitude += compute_average_gradient_magnitude(self.net)
                self.store_extended_summaries(detached_loss, current_activations)

                # store summaries
                current_accuracy = torch.mean((predictions.argmax(axis=1) == label.argmax(axis=1)).to(torch.float32))
                self.running_loss += detached_loss
                self.running_accuracy += current_accuracy.detach()
                if (i + 1) % self.running_avg_window == 0:
                    self._store_training_summaries()

            self.current_permutation += 1

    def _train_rmt(self, mnist_data_loader: DataLoader, training_data: MnistDataSet):
        self.rmt_memory = self._initialize_rmt_memory()
        num_batches = len(mnist_data_loader)
        # Average accumulated outer gradients over the true chunk size, including the final short chunk.
        tail_batches = num_batches % self.rmt_slow_update_freq

        for _ in tqdm(range(self.num_permutations), disable=not self.verbose):
            if self.current_permutation == self.num_permutations:
                break
            training_data.set_transformation(Permute(np.random.permutation(self.num_inputs)))

            if self.rmt_memory_reset_at_task_boundary:
                self.rmt_memory = self._initialize_rmt_memory()
            elif self.rmt_variant == "fast_memory" and not self.rmt_memory.requires_grad:
                self.rmt_memory = self.rmt_memory.detach().requires_grad_(True)

            self.optim.zero_grad(set_to_none=True)
            for i, sample in enumerate(mnist_data_loader):
                self.current_experiment_step += 1

                image = self._prepare_rmt_images(sample["image"])
                label = self._prepare_rmt_labels(sample["label"])
                is_tail_accumulation = tail_batches != 0 and i >= (num_batches - tail_batches)
                accumulation_divisor = tail_batches if is_tail_accumulation else self.rmt_slow_update_freq

                if self.rmt_variant == "baseline":
                    predictions, encoded_memory = self.net.forward(image, self.rmt_memory, return_encoded_memory=True)
                    current_loss = self.loss(predictions, label)
                    detached_loss = current_loss.detach().clone()
                    (current_loss / accumulation_divisor).backward()

                    with torch.no_grad():
                        next_memory = encoded_memory.detach().mean(dim=0)
                        memory_update_norm = (next_memory - self.rmt_memory.detach()).norm()
                        self.rmt_memory = next_memory

                else:   # fast_memory
                    if not self.rmt_memory.requires_grad:
                        self.rmt_memory = self.rmt_memory.detach().requires_grad_(True)

                    predictions = self.net.forward(image, self.rmt_memory)
                    current_loss = self.loss(predictions, label)
                    detached_loss = current_loss.detach().clone()
                    memory_loss = detached_loss
                    (current_loss / accumulation_divisor).backward()

                    memory_update_norm = torch.tensor(0.0, device=self.device, dtype=torch.float32)
                    with torch.no_grad():
                        if self.rmt_memory.grad is not None:
                            # Undo the outer-loss normalization so the per-minibatch memory update stays unchanged.
                            memory_grad = self.rmt_memory.grad * accumulation_divisor
                            if self.rmt_clip_memory_grad is not None:
                                grad_norm = memory_grad.norm()
                                if grad_norm > self.rmt_clip_memory_grad:
                                    scaling = self.rmt_clip_memory_grad / (grad_norm + 1e-12)
                                    memory_grad = memory_grad * scaling
                            memory_step = self.rmt_fast_lr * memory_grad
                            self.rmt_memory -= memory_step
                            memory_update_norm = memory_step.norm()
                        updated_memory = self.rmt_memory.detach()
                        if self.log_rmt_minibatch_losses and self.tb_logger is not None:
                            post_memory_predictions = self.net.forward(image, updated_memory)
                            post_memory_loss = self.loss(post_memory_predictions, label).detach()
                            self._log_rmt_minibatch_losses(memory_loss=memory_loss,
                                   memory_loss_improvement=memory_loss - post_memory_loss)
                    self.rmt_memory = self.rmt_memory.detach().requires_grad_(True)

                if ((i + 1) % self.rmt_slow_update_freq) == 0 or (i + 1) == num_batches:
                    self.optim.step()
                    self.rmt_optimizer_step_count += 1
                    if self.use_parameter_noise:
                        perturb_weights(self.net, self.parameter_noise_var)
                    self.optim.zero_grad(set_to_none=True)

                current_accuracy = torch.mean((predictions.argmax(axis=1) == label).to(torch.float32))
                self.running_loss += detached_loss
                self.running_accuracy += current_accuracy.detach()
                self._store_rmt_extended_summaries(memory_norm=self.rmt_memory.detach().norm(), memory_update_norm=memory_update_norm)

                if (i + 1) % self.running_avg_window == 0:
                    self._store_training_summaries()

                self.rmt_global_step += 1

            self.current_permutation += 1


def main():
    """
    This is a quick demonstration of how to run the experiments. For a more systematic run, use the mlproj_manager
    scheduler.
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
    relevant_parameters = experiment_parameters["relevant_parameters"]

    results_dir_name = "{0}-{1}".format(relevant_parameters[0], experiment_parameters[relevant_parameters[0]])
    for relevant_param in relevant_parameters[1:]:
        results_dir_name += "_" + relevant_param + "-" + str(experiment_parameters[relevant_param])

    results_path = os.path.join(file_path, "results") if "results_path" not in experiment_parameters.keys() else experiment_parameters["results_path"]

    initial_time = time.perf_counter()
    exp = PermutedMNISTExperiment(experiment_parameters,
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
