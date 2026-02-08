"""
LSL replay script: load trained model and stream classification markers.
Simulates real-time by feeding data (from file or random trials) and pushing
predictions to LSL for Unity to consume.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time
import random
import numpy as np
from pylsl import StreamInfo, StreamOutlet

from python_backend.training import load_model
from python_backend.utils import get_project_root
from python_backend.utils import get_models_dir


# LSL stream name Unity will resolve
LSL_STREAM_NAME = "BCI_Control"
LSL_STREAM_TYPE = "Markers"


def create_outlet() -> StreamOutlet:
    """Create LSL outlet for string markers."""
    info = StreamInfo(
        name=LSL_STREAM_NAME,
        type=LSL_STREAM_TYPE,
        channel_count=1,
        nominal_srate=0,
        channel_format="string",
        source_id="bci_python_sim_1",
    )
    return StreamOutlet(info)


def run_simulation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    command_map: dict,
    model_name: str = "csp_lda",
    interval_sec: float = 2.0,
) -> None:
    """
    Load model, create outlet, and loop: sample trial -> predict -> push marker.
    """
    clf = load_model(name=model_name)
    outlet = create_outlet()
    print(f"LSL stream '{LSL_STREAM_NAME}' active. Send predictions every {interval_sec}s.")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            idx = random.randint(0, len(X_train) - 1)
            trial = X_train[idx]
            trial_2d = trial[np.newaxis, :, :]
            pred = clf.predict(trial_2d)[0]
            command = command_map.get(pred, "Unknown")
            outlet.push_sample([command])
            print(f"Sent: [{command}] (pred={pred}, true={y_train[idx]})")
            time.sleep(interval_sec)
    except KeyboardInterrupt:
        print("Stopped.")


# 预测标签 -> 命令字符串。支持 PhysioNet (2/3)、BCI 2a/2b (769/770)、2b 部分为 (1/2)
DEFAULT_COMMAND_MAP = {3: "Left", 2: "Right", 769: "Left", 770: "Right", 1: "Left", 2: "Right"}


def main() -> None:
    """Entry point: 从 models/replay_data.npz 加载数据，可选 --model 指定模型名（如 best_model_sub1）。"""
    import argparse
    parser = argparse.ArgumentParser(description="LSL 回放：加载模型与 replay 数据，发送分类结果")
    parser.add_argument("--model", "-m", default="best_model_sub1",
                        help="模型名称（不含扩展名），如 best_model_sub1、csp_lda_sub1（默认: best_model_sub1）")
    args = parser.parse_args()

    root = get_project_root()
    try:
        data_path = root / "models" / "replay_data.npz"
        if data_path.exists():
            z = np.load(data_path)
            X_train, y_train = z["X"], z["y"]
        else:
            print("No replay_data.npz found. Run train_model.py first (it can save a replay set).")
            print("Exiting. Create models/replay_data.npz with keys 'X', 'y' to use replay.")
            return
    except Exception as e:
        print(f"Cannot load replay data: {e}")
        return

    models_dir = get_models_dir()
    model_name = args.model
    if not (models_dir / f"{model_name}.pkl").exists() and not (models_dir / f"{model_name}.joblib").exists():
        if model_name == "best_model_sub1":
            if (models_dir / "csp_lda.joblib").exists() or (models_dir / "csp_lda.pkl").exists():
                model_name = "csp_lda"
                print("Using fallback model: csp_lda (best_model_sub1 not found)")
    print(f"Using model: {model_name}")
    run_simulation(X_train, y_train, DEFAULT_COMMAND_MAP, model_name=model_name)


if __name__ == "__main__":
    main()
