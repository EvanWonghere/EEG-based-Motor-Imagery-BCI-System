"""Replay stream: iterate through offline replay data trial by trial."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


class ReplayStream:
    """Iterate over saved replay data (``replay_data.npz``) one trial at a time.

    Parameters
    ----------
    replay_data_path : str or Path
        Path to ``.npz`` file containing ``X`` (n_trials, n_ch, n_times) and
        ``y`` (n_trials,).
    shuffle : bool
        Shuffle trial order.
    loop : bool
        If True, restart from the beginning after all trials are consumed.
    seed : int
        Random seed for shuffling.
    """

    def __init__(
        self,
        replay_data_path: str | Path,
        shuffle: bool = True,
        loop: bool = True,
        seed: int = 42,
    ):
        data = np.load(replay_data_path)
        self.X: np.ndarray = data["X"]  # (n_trials, n_ch, n_times)
        self.y: np.ndarray = data["y"]  # (n_trials,)
        self.shuffle = shuffle
        self.loop = loop
        self.rng = np.random.default_rng(seed)

        self.n_trials = len(self.y)
        logger.info(
            "Replay data loaded: %d trials, shape %s",
            self.n_trials, self.X.shape,
        )

    def __len__(self) -> int:
        return self.n_trials

    def __iter__(self):
        """Yield ``(epoch, true_label)`` tuples.

        ``epoch`` has shape ``(1, n_channels, n_times)`` — ready for
        :meth:`OnlineClassifier.predict_trial`.
        """
        while True:
            indices = np.arange(self.n_trials)
            if self.shuffle:
                self.rng.shuffle(indices)

            for idx in indices:
                epoch = self.X[idx : idx + 1]  # keep batch dim
                label = int(self.y[idx])
                yield epoch, label

            if not self.loop:
                break

    @staticmethod
    def infer_path(model_path: str | Path) -> Path | None:
        """Guess replay data path from model path.

        ``{name}_sub{id}.pkl`` → ``replay_data_sub{id}.npz``
        """
        model_path = Path(model_path)
        # Extract subject id from filename like "fbcsp_svm_2a_sub1.pkl"
        stem = model_path.stem
        if "_sub" in stem:
            sub_part = stem.split("_sub")[-1]  # "1"
            replay_path = model_path.parent / f"replay_data_sub{sub_part}.npz"
            if replay_path.exists():
                return replay_path
        return None
