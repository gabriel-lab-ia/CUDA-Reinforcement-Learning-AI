from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import time
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn import functional as F
from torch.optim import Adam

AlgorithmName = Literal["reinforce", "dqn", "a2c"]


@dataclass(slots=True)
class ExperimentConfig:
    algorithm: AlgorithmName = "dqn"
    environment_id: str = "CartPole-v1"
    seed: int = 42
    total_episodes: int = 500
    max_steps_per_episode: int = 1_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    hidden_sizes: tuple[int, ...] = (256, 256)
    device: str = "auto"
    deterministic_torch: bool = False
    gradient_clip_norm: float = 10.0
    reward_window: int = 100
    solve_score: float = 475.0
    checkpoint_every: int = 50
    evaluation_every: int = 25
    evaluation_episodes: int = 10
    output_directory: str = "reports/rl"
    render_evaluation: bool = False
    normalize_observations: bool = False
    normalize_returns: bool = True
    entropy_coefficient: float = 0.01
    value_loss_coefficient: float = 0.5
    n_steps: int = 5
    batch_size: int = 128
    replay_capacity: int = 100_000
    learning_starts: int = 1_000
    train_frequency: int = 1
    gradient_steps: int = 1
    target_update_interval: int = 500
    tau: float = 1.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    double_dqn: bool = True


@dataclass(slots=True)
class EpisodeMetrics:
    episode: int
    reward: float
    length: int
    loss: float | None
    epsilon: float | None
    elapsed_seconds: float
    global_step: int
    moving_average_reward: float


@dataclass(slots=True)
class EvaluationMetrics:
    episode: int
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    mean_length: float


class Transition(NamedTuple):
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool


@dataclass(slots=True)
class RolloutStep:
    observation: np.ndarray
    action: int
    reward: float
    terminated: bool
    truncated: bool
    log_probability: Tensor
    value: Tensor
    entropy: Tensor


def parse_hidden_sizes(raw_value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in raw_value.split(",")]
    sizes = tuple(int(part) for part in parts if part)
    if not sizes:
        raise argparse.ArgumentTypeError("At least one hidden size is required.")
    if any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("Hidden sizes must be positive.")
    return sizes


def parse_arguments() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Deep reinforcement learning laboratory for CartPole-style tasks."
    )
    parser.add_argument(
        "--algorithm",
        choices=("reinforce", "dqn", "a2c"),
        default="dqn",
    )
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--hidden-sizes", type=parse_hidden_sizes, default=(256, 256))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--deterministic-torch", action="store_true")
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--reward-window", type=int, default=100)
    parser.add_argument("--solve-score", type=float, default=475.0)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--evaluation-every", type=int, default=25)
    parser.add_argument("--evaluation-episodes", type=int, default=10)
    parser.add_argument("--output-directory", default="reports/rl")
    parser.add_argument("--render-evaluation", action="store_true")
    parser.add_argument("--normalize-observations", action="store_true")
    parser.add_argument(
        "--disable-return-normalization",
        action="store_true",
    )
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--value-loss-coefficient", type=float, default=0.5)
    parser.add_argument("--n-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--train-frequency", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=500)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=50_000)
    parser.add_argument("--disable-double-dqn", action="store_true")
    args = parser.parse_args()

    return ExperimentConfig(
        algorithm=args.algorithm,
        environment_id=args.env,
        seed=args.seed,
        total_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
        deterministic_torch=args.deterministic_torch,
        gradient_clip_norm=args.gradient_clip,
        reward_window=args.reward_window,
        solve_score=args.solve_score,
        checkpoint_every=args.checkpoint_every,
        evaluation_every=args.evaluation_every,
        evaluation_episodes=args.evaluation_episodes,
        output_directory=args.output_directory,
        render_evaluation=args.render_evaluation,
        normalize_observations=args.normalize_observations,
        normalize_returns=not args.disable_return_normalization,
        entropy_coefficient=args.entropy_coefficient,
        value_loss_coefficient=args.value_loss_coefficient,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        learning_starts=args.learning_starts,
        train_frequency=args.train_frequency,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        tau=args.tau,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        double_dqn=not args.disable_double_dqn,
    )


def validate_config(config: ExperimentConfig) -> None:
    if config.total_episodes <= 0:
        raise ValueError("total_episodes must be positive.")
    if config.max_steps_per_episode <= 0:
        raise ValueError("max_steps_per_episode must be positive.")
    if not 0.0 < config.gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1].")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if config.gradient_clip_norm <= 0.0:
        raise ValueError("gradient_clip_norm must be positive.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.replay_capacity < config.batch_size:
        raise ValueError("replay_capacity must be at least batch_size.")
    if config.learning_starts < 0:
        raise ValueError("learning_starts cannot be negative.")
    if config.target_update_interval <= 0:
        raise ValueError("target_update_interval must be positive.")
    if not 0.0 < config.tau <= 1.0:
        raise ValueError("tau must be in (0, 1].")
    if config.epsilon_decay_steps <= 0:
        raise ValueError("epsilon_decay_steps must be positive.")
    if not 0.0 <= config.epsilon_end <= config.epsilon_start <= 1.0:
        raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1.")


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_reproducibility(seed: int, deterministic_torch: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def seed_environment(environment: gym.Env[Any, Any], seed: int) -> np.ndarray:
    observation, _ = environment.reset(seed=seed)
    environment.action_space.seed(seed)
    environment.observation_space.seed(seed)
    return np.asarray(observation, dtype=np.float32)


def make_environment(
    environment_id: str,
    *,
    render_mode: str | None = None,
) -> gym.Env[Any, Any]:
    environment = gym.make(environment_id, render_mode=render_mode)
    if not isinstance(environment.action_space, gym.spaces.Discrete):
        raise TypeError(
            "This laboratory currently supports discrete action spaces only."
        )
    if not isinstance(environment.observation_space, gym.spaces.Box):
        raise TypeError("This laboratory currently supports Box observations only.")
    if len(environment.observation_space.shape) != 1:
        raise ValueError("Only flat vector observations are currently supported.")
    return environment


class RunningMeanStd:
    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.variance = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, batch: np.ndarray) -> None:
        values = np.asarray(batch, dtype=np.float64)
        if values.ndim == self.mean.ndim:
            values = np.expand_dims(values, axis=0)
        batch_mean = values.mean(axis=0)
        batch_variance = values.var(axis=0)
        batch_count = values.shape[0]
        self._merge(batch_mean, batch_variance, batch_count)

    def _merge(
        self,
        batch_mean: np.ndarray,
        batch_variance: np.ndarray,
        batch_count: int,
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        current_m2 = self.variance * self.count
        batch_m2 = batch_variance * batch_count
        correction = np.square(delta) * self.count * batch_count / total_count
        new_variance = (current_m2 + batch_m2 + correction) / total_count
        self.mean = new_mean
        self.variance = new_variance
        self.count = total_count

    def normalize(
        self,
        value: np.ndarray,
        clip: float = 10.0,
    ) -> np.ndarray:
        normalized = (value - self.mean) / np.sqrt(self.variance + 1e-8)
        return np.clip(normalized, -clip, clip).astype(np.float32)


class ObservationProcessor:
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        enabled: bool,
    ) -> None:
        self.enabled = enabled
        self.statistics = RunningMeanStd(observation_shape)

    def process(self, observation: np.ndarray, *, update: bool) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32)
        if not self.enabled:
            return value
        if update:
            self.statistics.update(value)
        return self.statistics.normalize(value)


def initialize_linear_layer(layer: nn.Linear, gain: float = math.sqrt(2.0)) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


def build_mlp(
    input_dim: int,
    hidden_sizes: Sequence[int],
    output_dim: int,
    *,
    output_gain: float = 1.0,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_sizes:
        linear = nn.Linear(previous_dim, hidden_dim)
        initialize_linear_layer(linear)
        layers.extend((linear, nn.Tanh()))
        previous_dim = hidden_dim
    output_layer = nn.Linear(previous_dim, output_dim)
    initialize_linear_layer(output_layer, gain=output_gain)
    layers.append(output_layer)
    return nn.Sequential(*layers)


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int],
    ) -> None:
        super().__init__()
        self.network = build_mlp(
            observation_dim,
            hidden_sizes,
            action_dim,
            output_gain=0.01,
        )

    def forward(self, observations: Tensor) -> Tensor:
        return cast(Tensor, self.network(observations))

    def distribution(self, observations: Tensor) -> Categorical:
        logits = self.forward(observations)
        return Categorical(logits=logits)

    def act(
        self,
        observations: Tensor,
        *,
        deterministic: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        distribution = self.distribution(observations)
        if deterministic:
            actions = torch.argmax(distribution.logits, dim=-1)
        else:
            actions = distribution.sample()  # type: ignore[no-untyped-call]
        log_probabilities = distribution.log_prob(actions)  # type: ignore[no-untyped-call]
        entropies = distribution.entropy()  # type: ignore[no-untyped-call]
        return actions, log_probabilities, entropies


class ValueNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        hidden_sizes: Sequence[int],
    ) -> None:
        super().__init__()
        self.network = build_mlp(
            observation_dim,
            hidden_sizes,
            1,
            output_gain=1.0,
        )

    def forward(self, observations: Tensor) -> Tensor:
        return cast(Tensor, self.network(observations)).squeeze(-1)


class ActorCriticNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int],
    ) -> None:
        super().__init__()
        self.policy = PolicyNetwork(
            observation_dim,
            action_dim,
            hidden_sizes,
        )
        self.value = ValueNetwork(
            observation_dim,
            hidden_sizes,
        )

    def act(
        self,
        observations: Tensor,
        *,
        deterministic: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        actions, log_probabilities, entropies = self.policy.act(
            observations,
            deterministic=deterministic,
        )
        values = self.value(observations)
        return actions, log_probabilities, entropies, values


class QNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int],
    ) -> None:
        super().__init__()
        self.network = build_mlp(
            observation_dim,
            hidden_sizes,
            action_dim,
            output_gain=1.0,
        )

    def forward(self, observations: Tensor) -> Tensor:
        return cast(Tensor, self.network(observations))


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = capacity
        self.storage: deque[Transition] = deque(maxlen=capacity)
        self.random = random.Random(seed)

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, transition: Transition) -> None:
        self.storage.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size > len(self.storage):
            raise ValueError("Cannot sample more transitions than currently stored.")
        return self.random.sample(list(self.storage), batch_size)


def transitions_to_tensors(
    transitions: Sequence[Transition],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    observations = torch.as_tensor(
        np.stack([item.observation for item in transitions]),
        dtype=torch.float32,
        device=device,
    )
    actions = torch.as_tensor(
        [item.action for item in transitions],
        dtype=torch.int64,
        device=device,
    )
    rewards = torch.as_tensor(
        [item.reward for item in transitions],
        dtype=torch.float32,
        device=device,
    )
    next_observations = torch.as_tensor(
        np.stack([item.next_observation for item in transitions]),
        dtype=torch.float32,
        device=device,
    )
    dones = torch.as_tensor(
        [item.terminated for item in transitions],
        dtype=torch.float32,
        device=device,
    )
    return observations, actions, rewards, next_observations, dones


def discounted_returns(
    rewards: Sequence[float],
    gamma: float,
    *,
    normalize: bool,
    device: torch.device,
) -> Tensor:
    returns: list[float] = []
    running_return = 0.0
    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.append(running_return)
    returns.reverse()
    tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)
    if normalize and tensor.numel() > 1:
        tensor = (tensor - tensor.mean()) / (tensor.std(unbiased=False) + 1e-8)
    return tensor


def compute_n_step_returns(
    rollout: Sequence[RolloutStep],
    bootstrap_value: Tensor,
    gamma: float,
    device: torch.device,
) -> Tensor:
    returns: list[Tensor] = []
    running_return = bootstrap_value.detach()
    for step in reversed(rollout):
        terminal_mask = 0.0 if step.terminated else 1.0
        reward_tensor = torch.tensor(step.reward, dtype=torch.float32, device=device)
        running_return = reward_tensor + gamma * terminal_mask * running_return
        returns.append(running_return)
    returns.reverse()
    return torch.stack(returns)


def clip_gradients(
    parameters: Iterable[nn.Parameter],
    max_norm: float,
) -> float:
    norm = nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)
    return float(norm.item())


def soft_update(
    target_network: nn.Module,
    source_network: nn.Module,
    tau: float,
) -> None:
    with torch.no_grad():
        for target_parameter, source_parameter in zip(
            target_network.parameters(),
            source_network.parameters(),
            strict=True,
        ):
            target_parameter.mul_(1.0 - tau)
            target_parameter.add_(source_parameter, alpha=tau)


class MetricsWriter:
    def __init__(self, output_directory: Path) -> None:
        self.output_directory = output_directory
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.episode_path = output_directory / "episodes.csv"
        self.evaluation_path = output_directory / "evaluations.csv"
        self.summary_path = output_directory / "summary.json"
        self._episode_header_written = self.episode_path.exists()
        self._evaluation_header_written = self.evaluation_path.exists()

    def write_episode(self, metrics: EpisodeMetrics) -> None:
        with self.episode_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(asdict(metrics).keys()))
            if not self._episode_header_written:
                writer.writeheader()
                self._episode_header_written = True
            writer.writerow(asdict(metrics))

    def write_evaluation(self, metrics: EvaluationMetrics) -> None:
        with self.evaluation_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(asdict(metrics).keys()))
            if not self._evaluation_header_written:
                writer.writeheader()
                self._evaluation_header_written = True
            writer.writerow(asdict(metrics))

    def write_summary(self, summary: dict[str, Any]) -> None:
        with self.summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, sort_keys=True)


class CheckpointManager:
    def __init__(self, output_directory: Path) -> None:
        self.directory = output_directory / "checkpoints"
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        name: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: ExperimentConfig,
        episode: int,
        global_step: int,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        path = self.directory / f"{name}_episode_{episode:05d}.pt"
        payload: dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "episode": episode,
            "global_step": global_step,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        return path


class BaseAgent:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: ExperimentConfig,
        device: torch.device,
    ) -> None:
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.config = config
        self.device = device
        self.global_step = 0

    def select_action(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool,
    ) -> int:
        raise NotImplementedError

    def train_episode(
        self,
        environment: gym.Env[Any, Any],
        observation_processor: ObservationProcessor,
        episode_seed: int,
    ) -> tuple[float, int, float | None, float | None]:
        raise NotImplementedError

    def model_for_checkpoint(self) -> nn.Module:
        raise NotImplementedError

    def optimizer_for_checkpoint(self) -> torch.optim.Optimizer:
        raise NotImplementedError


class ReinforceAgent(BaseAgent):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: ExperimentConfig,
        device: torch.device,
    ) -> None:
        super().__init__(observation_dim, action_dim, config, device)
        self.policy = PolicyNetwork(
            observation_dim,
            action_dim,
            config.hidden_sizes,
        ).to(device)
        self.optimizer = Adam(
            self.policy.parameters(),
            lr=config.learning_rate,
        )

    def select_action(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool,
    ) -> int:
        tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = self.policy.act(
                tensor,
                deterministic=deterministic,
            )
        return int(action.item())

    def train_episode(
        self,
        environment: gym.Env[Any, Any],
        observation_processor: ObservationProcessor,
        episode_seed: int,
    ) -> tuple[float, int, float | None, float | None]:
        observation, _ = environment.reset(seed=episode_seed)
        observation = observation_processor.process(
            np.asarray(observation, dtype=np.float32),
            update=True,
        )
        rewards: list[float] = []
        log_probabilities: list[Tensor] = []
        entropies: list[Tensor] = []

        for _step in range(self.config.max_steps_per_episode):
            observation_tensor = torch.as_tensor(
                observation,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            action, log_probability, entropy = self.policy.act(
                observation_tensor,
                deterministic=False,
            )
            next_observation, reward, terminated, truncated, _ = environment.step(
                int(action.item())
            )
            rewards.append(float(reward))
            log_probabilities.append(log_probability.squeeze(0))
            entropies.append(entropy.squeeze(0))
            self.global_step += 1
            observation = observation_processor.process(
                np.asarray(next_observation, dtype=np.float32),
                update=True,
            )
            if terminated or truncated:
                break

        returns = discounted_returns(
            rewards,
            self.config.gamma,
            normalize=self.config.normalize_returns,
            device=self.device,
        )
        policy_loss = -(torch.stack(log_probabilities) * returns).mean()
        entropy_bonus = torch.stack(entropies).mean()
        loss = policy_loss - self.config.entropy_coefficient * entropy_bonus

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        clip_gradients(
            self.policy.parameters(),
            self.config.gradient_clip_norm,
        )
        self.optimizer.step()

        return sum(rewards), len(rewards), float(loss.item()), None

    def model_for_checkpoint(self) -> nn.Module:
        return self.policy

    def optimizer_for_checkpoint(self) -> torch.optim.Optimizer:
        return self.optimizer


class DQNAgent(BaseAgent):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: ExperimentConfig,
        device: torch.device,
    ) -> None:
        super().__init__(observation_dim, action_dim, config, device)
        self.online_network = QNetwork(
            observation_dim,
            action_dim,
            config.hidden_sizes,
        ).to(device)
        self.target_network = QNetwork(
            observation_dim,
            action_dim,
            config.hidden_sizes,
        ).to(device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
        self.optimizer = Adam(
            self.online_network.parameters(),
            lr=config.learning_rate,
        )
        self.replay_buffer = ReplayBuffer(
            capacity=config.replay_capacity,
            seed=config.seed,
        )
        self.random = random.Random(config.seed)

    def epsilon(self) -> float:
        fraction = min(1.0, self.global_step / self.config.epsilon_decay_steps)
        return self.config.epsilon_start + fraction * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def select_action(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool,
    ) -> int:
        epsilon = 0.0 if deterministic else self.epsilon()
        if self.random.random() < epsilon:
            return self.random.randrange(self.action_dim)
        tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_network(tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def optimize(self) -> float:
        transitions = self.replay_buffer.sample(self.config.batch_size)
        observations, actions, rewards, next_observations, dones = (
            transitions_to_tensors(transitions, self.device)
        )
        current_q_values = self.online_network(observations)
        selected_q_values = current_q_values.gather(
            dim=1,
            index=actions.unsqueeze(1),
        ).squeeze(1)

        with torch.no_grad():
            if self.config.double_dqn:
                next_actions = self.online_network(next_observations).argmax(
                    dim=1,
                    keepdim=True,
                )
                next_q_values = (
                    self.target_network(next_observations)
                    .gather(
                        dim=1,
                        index=next_actions,
                    )
                    .squeeze(1)
                )
            else:
                next_q_values = self.target_network(next_observations).max(dim=1).values
            targets = rewards + self.config.gamma * (1.0 - dones) * next_q_values

        loss = F.smooth_l1_loss(selected_q_values, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        clip_gradients(
            self.online_network.parameters(),
            self.config.gradient_clip_norm,
        )
        self.optimizer.step()
        return float(loss.item())

    def train_episode(
        self,
        environment: gym.Env[Any, Any],
        observation_processor: ObservationProcessor,
        episode_seed: int,
    ) -> tuple[float, int, float | None, float | None]:
        observation, _ = environment.reset(seed=episode_seed)
        observation = observation_processor.process(
            np.asarray(observation, dtype=np.float32),
            update=True,
        )
        episode_reward = 0.0
        losses: list[float] = []

        for _step in range(self.config.max_steps_per_episode):
            action = self.select_action(
                observation,
                deterministic=False,
            )
            next_observation, reward, terminated, truncated, _ = environment.step(
                action
            )
            processed_next_observation = observation_processor.process(
                np.asarray(next_observation, dtype=np.float32),
                update=True,
            )
            self.replay_buffer.add(
                Transition(
                    observation=observation.copy(),
                    action=action,
                    reward=float(reward),
                    next_observation=processed_next_observation.copy(),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
            )
            observation = processed_next_observation
            episode_reward += float(reward)
            self.global_step += 1

            should_train = (
                self.global_step >= self.config.learning_starts
                and self.global_step % self.config.train_frequency == 0
                and len(self.replay_buffer) >= self.config.batch_size
            )
            if should_train:
                for _ in range(self.config.gradient_steps):
                    losses.append(self.optimize())

            if self.global_step % self.config.target_update_interval == 0:
                soft_update(
                    self.target_network,
                    self.online_network,
                    self.config.tau,
                )

            if terminated or truncated:
                break

        mean_loss = statistics.fmean(losses) if losses else None
        return episode_reward, _step + 1, mean_loss, self.epsilon()

    def model_for_checkpoint(self) -> nn.Module:
        return self.online_network

    def optimizer_for_checkpoint(self) -> torch.optim.Optimizer:
        return self.optimizer


class A2CAgent(BaseAgent):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: ExperimentConfig,
        device: torch.device,
    ) -> None:
        super().__init__(observation_dim, action_dim, config, device)
        self.network = ActorCriticNetwork(
            observation_dim,
            action_dim,
            config.hidden_sizes,
        ).to(device)
        self.optimizer = Adam(
            self.network.parameters(),
            lr=config.learning_rate,
        )

    def select_action(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool,
    ) -> int:
        tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            action, _, _, _ = self.network.act(
                tensor,
                deterministic=deterministic,
            )
        return int(action.item())

    def update_rollout(
        self,
        rollout: Sequence[RolloutStep],
        bootstrap_value: Tensor,
    ) -> float:
        returns = compute_n_step_returns(
            rollout,
            bootstrap_value,
            self.config.gamma,
            self.device,
        )
        values = torch.stack([step.value for step in rollout])
        log_probabilities = torch.stack([step.log_probability for step in rollout])
        entropies = torch.stack([step.entropy for step in rollout])
        advantages = returns - values

        policy_loss = -(log_probabilities * advantages.detach()).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_bonus = entropies.mean()
        loss = (
            policy_loss
            + self.config.value_loss_coefficient * value_loss
            - self.config.entropy_coefficient * entropy_bonus
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        clip_gradients(
            self.network.parameters(),
            self.config.gradient_clip_norm,
        )
        self.optimizer.step()
        return float(loss.item())

    def train_episode(
        self,
        environment: gym.Env[Any, Any],
        observation_processor: ObservationProcessor,
        episode_seed: int,
    ) -> tuple[float, int, float | None, float | None]:
        observation, _ = environment.reset(seed=episode_seed)
        observation = observation_processor.process(
            np.asarray(observation, dtype=np.float32),
            update=True,
        )
        episode_reward = 0.0
        losses: list[float] = []
        rollout: list[RolloutStep] = []

        for _step in range(self.config.max_steps_per_episode):
            observation_tensor = torch.as_tensor(
                observation,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            action, log_probability, entropy, value = self.network.act(
                observation_tensor,
                deterministic=False,
            )
            next_observation, reward, terminated, truncated, _ = environment.step(
                int(action.item())
            )
            rollout.append(
                RolloutStep(
                    observation=observation.copy(),
                    action=int(action.item()),
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    log_probability=log_probability.squeeze(0),
                    value=value.squeeze(0),
                    entropy=entropy.squeeze(0),
                )
            )
            episode_reward += float(reward)
            self.global_step += 1
            observation = observation_processor.process(
                np.asarray(next_observation, dtype=np.float32),
                update=True,
            )

            rollout_finished = (
                len(rollout) >= self.config.n_steps or terminated or truncated
            )
            if rollout_finished:
                if terminated:
                    bootstrap_value = torch.zeros(
                        (),
                        dtype=torch.float32,
                        device=self.device,
                    )
                else:
                    next_tensor = torch.as_tensor(
                        observation,
                        dtype=torch.float32,
                        device=self.device,
                    ).unsqueeze(0)
                    with torch.no_grad():
                        bootstrap_value = self.network.value(next_tensor).squeeze(0)
                losses.append(
                    self.update_rollout(
                        rollout,
                        bootstrap_value,
                    )
                )
                rollout.clear()

            if terminated or truncated:
                break

        mean_loss = statistics.fmean(losses) if losses else None
        return episode_reward, _step + 1, mean_loss, None

    def model_for_checkpoint(self) -> nn.Module:
        return self.network

    def optimizer_for_checkpoint(self) -> torch.optim.Optimizer:
        return self.optimizer


def create_agent(
    config: ExperimentConfig,
    observation_dim: int,
    action_dim: int,
    device: torch.device,
) -> BaseAgent:
    if config.algorithm == "reinforce":
        return ReinforceAgent(
            observation_dim,
            action_dim,
            config,
            device,
        )
    if config.algorithm == "dqn":
        return DQNAgent(
            observation_dim,
            action_dim,
            config,
            device,
        )
    if config.algorithm == "a2c":
        return A2CAgent(
            observation_dim,
            action_dim,
            config,
            device,
        )
    raise ValueError(f"Unsupported algorithm: {config.algorithm}")


def evaluate_agent(
    agent: BaseAgent,
    config: ExperimentConfig,
    observation_processor: ObservationProcessor,
    episode: int,
) -> EvaluationMetrics:
    render_mode = "human" if config.render_evaluation else None
    environment = make_environment(
        config.environment_id,
        render_mode=render_mode,
    )
    rewards: list[float] = []
    lengths: list[int] = []

    try:
        for evaluation_index in range(config.evaluation_episodes):
            observation, _ = environment.reset(
                seed=config.seed + 100_000 + evaluation_index
            )
            observation = observation_processor.process(
                np.asarray(observation, dtype=np.float32),
                update=False,
            )
            episode_reward = 0.0

            for _step in range(config.max_steps_per_episode):
                action = agent.select_action(
                    observation,
                    deterministic=True,
                )
                observation, reward, terminated, truncated, _ = environment.step(action)
                observation = observation_processor.process(
                    np.asarray(observation, dtype=np.float32),
                    update=False,
                )
                episode_reward += float(reward)
                if terminated or truncated:
                    break

            rewards.append(episode_reward)
            lengths.append(_step + 1)
    finally:
        environment.close()

    return EvaluationMetrics(
        episode=episode,
        mean_reward=statistics.fmean(rewards),
        std_reward=statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
        min_reward=min(rewards),
        max_reward=max(rewards),
        mean_length=statistics.fmean(lengths),
    )


def describe_hardware(device: torch.device) -> dict[str, Any]:
    description: dict[str, Any] = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        description.update(
            {
                "gpu_name": properties.name,
                "compute_capability": (f"{properties.major}.{properties.minor}"),
                "vram_gib": properties.total_memory / (1024**3),
                "multiprocessors": properties.multi_processor_count,
            }
        )
    return description


def print_experiment_header(
    config: ExperimentConfig,
    device: torch.device,
    observation_dim: int,
    action_dim: int,
) -> None:
    hardware = describe_hardware(device)
    print("=" * 80)
    print("CUDA Reinforcement Learning AI")
    print("=" * 80)
    print(f"Algorithm:          {config.algorithm}")
    print(f"Environment:        {config.environment_id}")
    print(f"Device:             {hardware['device']}")
    print(f"Observation dim:    {observation_dim}")
    print(f"Action dim:         {action_dim}")
    print(f"Hidden sizes:       {config.hidden_sizes}")
    print(f"Learning rate:      {config.learning_rate}")
    print(f"Gamma:              {config.gamma}")
    if device.type == "cuda":
        print(f"GPU:                {hardware['gpu_name']}")
        print(f"Compute capability: {hardware['compute_capability']}")
        print(f"VRAM:               {hardware['vram_gib']:.2f} GiB")
        print(f"CUDA build:         {hardware['cuda_build']}")
    print("=" * 80)


def train(config: ExperimentConfig) -> dict[str, Any]:
    validate_config(config)
    configure_reproducibility(
        config.seed,
        config.deterministic_torch,
    )
    device = select_device(config.device)
    environment = make_environment(config.environment_id)
    initial_observation = seed_environment(environment, config.seed)
    observation_dim = int(initial_observation.shape[0])
    if not isinstance(environment.action_space, gym.spaces.Discrete):
        raise TypeError("This module currently supports only discrete action spaces.")

    action_dim = int(environment.action_space.n)

    observation_processor = ObservationProcessor(
        (observation_dim,),
        enabled=config.normalize_observations,
    )
    agent = create_agent(
        config,
        observation_dim,
        action_dim,
        device,
    )
    run_name = (
        f"{config.algorithm}_{config.environment_id}_seed_{config.seed}_"
        f"{time.strftime('%Y%m%d_%H%M%S')}"
    )
    output_directory = Path(config.output_directory) / run_name
    writer = MetricsWriter(output_directory)
    checkpoint_manager = CheckpointManager(output_directory)

    print_experiment_header(
        config,
        device,
        observation_dim,
        action_dim,
    )

    reward_history: deque[float] = deque(maxlen=config.reward_window)
    all_rewards: list[float] = []
    evaluation_history: list[EvaluationMetrics] = []
    started_at = time.perf_counter()
    best_evaluation_reward = -math.inf
    solved_episode: int | None = None

    try:
        for episode in range(1, config.total_episodes + 1):
            episode_started_at = time.perf_counter()
            reward, length, loss, epsilon = agent.train_episode(
                environment,
                observation_processor,
                config.seed + episode,
            )
            reward_history.append(reward)
            all_rewards.append(reward)
            moving_average = statistics.fmean(reward_history)
            elapsed = time.perf_counter() - episode_started_at
            episode_metrics = EpisodeMetrics(
                episode=episode,
                reward=reward,
                length=length,
                loss=loss,
                epsilon=epsilon,
                elapsed_seconds=elapsed,
                global_step=agent.global_step,
                moving_average_reward=moving_average,
            )
            writer.write_episode(episode_metrics)

            loss_text = "n/a" if loss is None else f"{loss:.5f}"
            epsilon_text = "n/a" if epsilon is None else f"{epsilon:.4f}"
            print(
                f"Episode {episode:04d} | "
                f"reward={reward:7.2f} | "
                f"avg={moving_average:7.2f} | "
                f"length={length:4d} | "
                f"loss={loss_text:>10} | "
                f"epsilon={epsilon_text:>7} | "
                f"steps={agent.global_step:7d}"
            )

            should_evaluate = (
                episode == 1
                or episode % config.evaluation_every == 0
                or episode == config.total_episodes
            )
            if should_evaluate:
                evaluation = evaluate_agent(
                    agent,
                    config,
                    observation_processor,
                    episode,
                )
                evaluation_history.append(evaluation)
                writer.write_evaluation(evaluation)
                print(
                    f"Evaluation @ {episode:04d} | "
                    f"mean={evaluation.mean_reward:7.2f} | "
                    f"std={evaluation.std_reward:7.2f} | "
                    f"range=[{evaluation.min_reward:.2f}, "
                    f"{evaluation.max_reward:.2f}]"
                )
                if evaluation.mean_reward > best_evaluation_reward:
                    best_evaluation_reward = evaluation.mean_reward
                    checkpoint_manager.save(
                        "best",
                        agent.model_for_checkpoint(),
                        agent.optimizer_for_checkpoint(),
                        config,
                        episode,
                        agent.global_step,
                        extra={"evaluation": asdict(evaluation)},
                    )

            should_checkpoint = (
                episode % config.checkpoint_every == 0
                or episode == config.total_episodes
            )
            if should_checkpoint:
                checkpoint_manager.save(
                    "checkpoint",
                    agent.model_for_checkpoint(),
                    agent.optimizer_for_checkpoint(),
                    config,
                    episode,
                    agent.global_step,
                )

            if (
                len(reward_history) == config.reward_window
                and moving_average >= config.solve_score
            ):
                solved_episode = episode
                print(
                    f"Environment solved at episode {episode} "
                    f"with moving average {moving_average:.2f}."
                )
                break
    finally:
        environment.close()

    total_elapsed = time.perf_counter() - started_at
    final_evaluation = evaluate_agent(
        agent,
        config,
        observation_processor,
        solved_episode or len(all_rewards),
    )
    writer.write_evaluation(final_evaluation)

    summary: dict[str, Any] = {
        "config": asdict(config),
        "hardware": describe_hardware(device),
        "episodes_completed": len(all_rewards),
        "global_steps": agent.global_step,
        "total_elapsed_seconds": total_elapsed,
        "mean_training_reward": statistics.fmean(all_rewards),
        "max_training_reward": max(all_rewards),
        "final_moving_average_reward": (
            statistics.fmean(reward_history) if reward_history else 0.0
        ),
        "best_evaluation_reward": best_evaluation_reward,
        "final_evaluation": asdict(final_evaluation),
        "solved_episode": solved_episode,
        "output_directory": str(output_directory),
    }
    writer.write_summary(summary)
    from cuda_rl.storage import JsonlDocumentStore

    document = JsonlDocumentStore(output_directory / "nosql").insert(
        "experiment_summaries",
        {
            "algorithm": config.algorithm,
            "environment_id": config.environment_id,
            "seed": config.seed,
            "episodes_completed": len(all_rewards),
            "global_steps": agent.global_step,
            "best_evaluation_reward": best_evaluation_reward,
            "final_moving_average_reward": summary["final_moving_average_reward"],
            "summary_path": str(writer.summary_path),
        },
    )
    summary["document_store"] = {
        "backend": "jsonl",
        "collection": document.collection,
        "document_id": document.id,
        "path": str(output_directory / "nosql" / "experiment_summaries.jsonl"),
    }
    writer.write_summary(summary)
    return summary


def main() -> None:
    config = parse_arguments()
    summary = train(config)
    print("=" * 80)
    print("Training complete")
    print("=" * 80)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
