"""
训练入口：支持 PhysioNet EEGBCI、BCI Competition IV 2a、IV 2b。
多模型对比（LDA / SVM / RF），遍历 9 名受试者，10 折交叉验证，保存最佳模型供 Unity 演示。
"""
from pathlib import Path
import sys

# 优先加载 .env，保证 MNE_DATA 等生效
ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

# NumPy 2: fromstring(binary) removed; MNE GDF/EDF uses fromstring(etmode, UINT8)
if hasattr(np, "fromstring"):
    _np_fromstring = np.fromstring
    def _fromstring_compat(s, dtype=float, count=-1, sep=""):
        if sep == "":
            try:
                return np.frombuffer(s, dtype=dtype, count=count)
            except (TypeError, ValueError):
                if isinstance(s, str):
                    return np.frombuffer(s.encode("latin-1"), dtype=dtype, count=count)
                raise
        return _np_fromstring(s, dtype=dtype, count=count, sep=sep)
    np.fromstring = _fromstring_compat

import mne
from mne.io import read_raw_gdf, RawArray
from scipy.io import loadmat

from python_backend.utils import (
    BAND_LOW_HZ,
    BAND_HIGH_HZ,
    EPOCH_TMIN,
    EPOCH_TMAX,
    EPOCH_TRAIN_TMIN,
    EPOCH_TRAIN_TMAX,
    get_models_dir,
    get_data_dir,
    DIR_BCI_2A,
    DIR_BCI_2B,
)
from python_backend.datasets import get_mne_data_path, _set_mne_data_path
from python_backend.preprocessing import bandpass_filter, get_epochs, epochs_to_arrays
from python_backend.training import (
    MODEL_TYPES,
    run_training,
    save_model,
    get_best_model_name_for_subject,
)


# BCI IV 2a 四类中的左右手二分类（用于 2a 数据）
EVENT_ID_LEFT_RIGHT_2A = {"Left Hand": 769, "Right Hand": 770}


# BCI IV 2a/2b .mat 采样率（常用 250 Hz）
BCI_2A_SFREQ = 250
BCI_2B_SFREQ = 250


def _find_file_in_dir(patterns: list, subject_id: int, base_dir: Path, mne_path: Path) -> Path:
    """
    在 base_dir 下查找文件，patterns 为文件名列表模板，如 ["A0{:d}E.mat"]（一位数字：A01E 而非 A001E）。
    """
    if base_dir.exists():
        for pat in patterns:
            fname = pat.format(subject_id)
            p = base_dir / fname
            if p.exists():
                return p
            for p in base_dir.rglob(fname):
                return p
    if mne_path.exists():
        for pat in patterns:
            fname = pat.format(subject_id)
            for p in mne_path.rglob(fname):
                return p
    return None


def _mat_to_raw_events(mat_path: Path, sfreq: float, keep_classes: list, event_id_map: dict):
    """
    从 BCI IV .mat 文件加载数据并构造 MNE Raw 与 events。
    keep_classes: 要保留的类别标签，如 [1, 2] 表示左右手。
    event_id_map: 标签 -> 事件码，如 {1: 769, 2: 770}。
    返回 (raw, events, event_id)。
    """
    mat = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    # 常见 key: data/Data/X, label/labels/y/class；或仅 data 为 struct 含 x/y 等字段
    data = None
    for k in ("data", "Data", "X", "x"):
        if k in mat and isinstance(mat[k], np.ndarray):
            data = np.asarray(mat[k])
            break
    if data is None:
        raise ValueError(f".mat 中未找到 data/Data/X，keys: {[k for k in mat if not k.startswith('_')]}")
    labels = None
    for k in ("label", "labels", "y", "class", "Y"):
        if k in mat and isinstance(mat[k], np.ndarray):
            labels = np.asarray(mat[k]).ravel()
            break
    # 若顶层无 label：尝试从 data struct 中取（部分 .mat 仅含 data，标签在 data 的字段里）
    if labels is None and data is not None:
        data_flat = data.ravel()
        if len(data_flat) > 0:
            first = data_flat[0]
            # data 可能是 (1,1) 的 struct，唯一元素有 .x/.y 等属性
            for attr in ("y", "label", "labels", "class", "t", "trial_type", "type"):
                if hasattr(first, attr):
                    arr = getattr(first, attr)
                    if isinstance(arr, np.ndarray):
                        labels = np.asarray(arr).ravel()
                        break
            if labels is None and hasattr(first, "_fieldnames"):
                for fn in getattr(first, "_fieldnames", []):
                    if fn.lower() in ("y", "label", "class", "t"):
                        arr = getattr(first, fn, None)
                        if isinstance(arr, np.ndarray):
                            labels = np.asarray(arr).ravel()
                            break
        # 若 data 本身是 (n_trials,) 的 struct，每项有 .y
        if labels is None and data.size > 1 and hasattr(data_flat[0], "y"):
            labels = np.array([getattr(t, "y", None) for t in data_flat])
            if labels.dtype == object:
                labels = np.array([int(l) if l is not None else -1 for l in labels])
        # (1,1) struct：唯一元素即整个 dataset，其 .y 为标签
        if labels is None and data.size == 1 and hasattr(first, "y"):
            labels = np.asarray(getattr(first, "y")).ravel()
    if labels is None:
        raise ValueError(f".mat 中未找到 label/labels/y（含 data 内字段），keys: {[k for k in mat if not k.startswith('_')]}")
    # 统一为 (n_trials, n_channels, n_times)：data 可能是 struct 需先取出 X
    data_flat = data.ravel()
    first = data_flat[0] if len(data_flat) > 0 else None
    if first is not None and (data.ndim == 1 or (data.ndim == 2 and data.size > 1)):
        # struct array: 每行为一 trial，取 .x 或 .X
        attr = "x" if hasattr(first, "x") else "X" if hasattr(first, "X") else None
        if attr is not None:
            parts = [np.asarray(getattr(t, attr), dtype=float) for t in data_flat]
            if all(p.ndim == 3 for p in parts):
                data = np.concatenate(parts, axis=0)
                if hasattr(first, "y"):
                    label_parts = []
                    for t in data_flat:
                        if hasattr(t, "y"):
                            yv = getattr(t, "y")
                            if yv is not None:
                                label_parts.append(np.asarray(yv).ravel())
                    if label_parts:
                        labels = np.concatenate(label_parts)
            elif len(parts) == 1:
                data = parts[0]
    # (1,1) struct：唯一元素 .x 或 .X 为 (n_trials, n_ch, n_times)
    if first is not None and data.ndim != 3 and data.size == 1:
        if hasattr(first, "x"):
            data = np.asarray(getattr(first, "x"), dtype=float)
        elif hasattr(first, "X"):
            data = np.asarray(getattr(first, "X"), dtype=float)
    if data.ndim == 3:
        a, b, c = data.shape
        if a == labels.shape[0]:
            X = np.asarray(data, dtype=float)  # (trials, ch, times)
        elif c == labels.shape[0]:
            X = np.transpose(data, (2, 0, 1))  # (ch, times, trials) -> (trials, ch, times)
        elif b == labels.shape[0]:
            X = np.transpose(data, (1, 0, 2))  # (ch, trials, times) -> (trials, ch, times)
        else:
            raise ValueError(f"data shape {data.shape} does not match labels len {labels.shape[0]}")
        assert X.shape[0] == labels.shape[0], f"data trials {X.shape[0]} vs labels {labels.shape[0]}"
    else:
        raise ValueError(f"data ndim must be 3, got {data.ndim}")
    n_trials, n_ch, n_times = X.shape
    # 只保留指定类别
    keep = np.isin(labels, keep_classes)
    X = X[keep]
    labels = labels[keep]
    if len(labels) == 0:
        raise ValueError(f"保留类别 {keep_classes} 后无 trial")
    # 构造连续数据与事件：按 trial 拼接，事件在每段起始
    event_id = {f"class_{v}": event_id_map[v] for v in keep_classes if v in event_id_map}
    # 用 event_id 的值作为事件码
    samples_per_trial = n_times
    data_concat = X.reshape(n_ch, -1)  # (n_ch, n_trials * n_times)
    events_list = []
    for i, lab in enumerate(labels):
        code = event_id_map.get(int(lab), int(lab))
        events_list.append([i * samples_per_trial, 0, code])
    events = np.array(events_list, dtype=int)
    info = mne.create_info(
        ch_names=[f"EEG{i+1}" for i in range(n_ch)],
        sfreq=sfreq,
        ch_types="eeg",
    )
    raw = RawArray(data_concat, info, verbose=False)
    # event_id 为 MNE 格式：名称 -> 事件码（与 events 中的 code 一致）
    codes_in_use = np.unique(events[:, 2]).tolist()
    names = ["Left Hand", "Right Hand"] if len(codes_in_use) == 2 else [f"class_{c}" for c in codes_in_use]
    event_id = dict(zip(names, codes_in_use))
    return raw, events, event_id


def load_bci_iv_2a(subject_id: int):
    """
    加载 BCI Competition IV 2a 指定受试者数据。优先 GDF（如 A01E.gdf、A01T.gdf），其次 .mat。
    数据目录：data/MNE-bnci-data/database/data-sets/001-2014。仅保留左右手二类（769/770）。
    返回 (raw, events, event_id)。
    """
    data_dir = get_data_dir()
    dir_2a = data_dir / DIR_BCI_2A  # e.g. data/MNE-bnci-data/database/data-sets/001-2014
    mne_path = get_mne_data_path()
    _set_mne_data_path(mne_path)
    # 优先 GDF；训练用 T（trial 多），评估用 E 也可
    path = _find_file_in_dir(["A0{:d}T.gdf", "A0{:d}E.gdf"], subject_id, dir_2a, mne_path)
    if path is not None:
        raw = read_raw_gdf(str(path), preload=True)
        try:
            events = mne.find_events(raw, shortest_event=1)
            event_id = {k: v for k, v in EVENT_ID_LEFT_RIGHT_2A.items()}
        except (ValueError, RuntimeError):
            events, event_id_ann = mne.events_from_annotations(raw)
            # GDF annotations may use 768/783 etc.; keep two class codes for left/right
            codes = np.unique(events[:, 2])
            skip = {276, 277, 1023, 1072, 32766}  # non-class markers
            class_codes = [c for c in codes if c not in skip][:2]
            if len(class_codes) >= 2:
                event_id = {"Left Hand": int(class_codes[0]), "Right Hand": int(class_codes[1])}
            else:
                event_id = {k: v for k, v in EVENT_ID_LEFT_RIGHT_2A.items()}
        events = events[np.isin(events[:, 2], list(event_id.values()))]
        return raw, events, event_id
    # 回退到 .mat
    path = _find_file_in_dir(
        ["A0{:d}E.mat", "A0{:d}T.mat", "A0{:d}e.mat", "A0{:d}t.mat"],
        subject_id, dir_2a, mne_path,
    )
    if path is not None and path.suffix.lower() == ".mat":
        return _mat_to_raw_events(
            path, BCI_2A_SFREQ,
            keep_classes=[1, 2],
            event_id_map={1: 769, 2: 770},
        )
    raise FileNotFoundError(
        f"BCI IV 2a subject {subject_id}: 未找到 A0{subject_id}E.gdf / A0{subject_id}T.gdf 或 .mat，"
        f"请放在 data/{DIR_BCI_2A}/ 下（如 data/MNE-bnci-data/database/data-sets/001-2014/）"
    )


def load_bci_iv_2b(subject_id: int):
    """
    加载 BCI Competition IV 2b 指定受试者数据。优先 GDF（如 B01E.gdf、B01T.gdf），其次 .mat。
    数据目录：data/MNE-bnci-data/database/data-sets/004-2014。两分类（769/770 或 1/2）。
    返回 (raw, events, event_id)。
    """
    data_dir = get_data_dir()
    dir_2b = data_dir / DIR_BCI_2B
    mne_path = get_mne_data_path()
    _set_mne_data_path(mne_path)
    # 优先 GDF（2b 命名为 B0104E.gdf、B0105E.gdf、B0101T.gdf 等，即 B{subj:02d}{run:02d}E/T.gdf）
    path = _find_file_in_dir(
        ["B{:02d}04E.gdf", "B{:02d}05E.gdf", "B{:02d}01T.gdf", "B{:02d}02T.gdf", "B{:02d}03T.gdf"],
        subject_id, dir_2b, mne_path,
    )
    if path is not None:
        raw = read_raw_gdf(str(path), preload=True)
        try:
            events = mne.find_events(raw, shortest_event=1)
        except (ValueError, RuntimeError):
            events, _ = mne.events_from_annotations(raw)
        uniq = np.unique(events[:, 2])
        if 769 in uniq and 770 in uniq:
            event_id = {"Left Hand": 769, "Right Hand": 770}
        elif 1 in uniq and 2 in uniq:
            event_id = {"Left Hand": 1, "Right Hand": 2}
        else:
            from collections import Counter
            c = Counter(events[:, 2])
            two = [x for x, _ in c.most_common(2)]
            event_id = {"Left Hand": int(two[0]), "Right Hand": int(two[1])}
        events = events[np.isin(events[:, 2], list(event_id.values()))]
        return raw, events, event_id
    # 回退到 .mat
    path = _find_file_in_dir(
        ["B0{:d}E.mat", "B0{:d}T.mat", "B0{:d}e.mat", "B0{:d}t.mat"],
        subject_id, dir_2b, mne_path,
    )
    if path is not None and path.suffix.lower() == ".mat":
        mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
        labels = None
        for k in ("label", "labels", "y", "class", "Y"):
            if k in mat and isinstance(mat[k], np.ndarray):
                labels = np.asarray(mat[k]).ravel()
                break
        if labels is not None:
            uniq = np.unique(labels)
            if len(uniq) >= 2:
                two = uniq[:2]
                event_id_map = {int(two[0]): 769, int(two[1]): 770}
                return _mat_to_raw_events(path, BCI_2B_SFREQ, keep_classes=two.tolist(), event_id_map=event_id_map)
        return _mat_to_raw_events(path, BCI_2B_SFREQ, keep_classes=[1, 2], event_id_map={1: 769, 2: 770})
    raise FileNotFoundError(
        f"BCI IV 2b subject {subject_id}: 未找到 B0{subject_id}E.gdf / B0{subject_id}T.gdf 或 .mat，"
        f"请放在 data/{DIR_BCI_2B}/ 下（如 data/MNE-bnci-data/database/data-sets/004-2014/）"
    )


def load_physionet_eegbci(subject: int = 1, runs: list = None):
    """加载 PhysioNet EEGBCI（手 vs 脚）。返回 (raw, events, event_id)。"""
    from mne.datasets import eegbci
    from mne.io import concatenate_raws, read_raw_edf

    if runs is None:
        runs = [6, 10, 14]
    raw_fnames = eegbci.load_data(subject, runs)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage)
    raw.annotations.rename(dict(T1="hands", T2="feet"))
    events, event_id = mne.events_from_annotations(raw)
    event_id = {k: v for k, v in event_id.items() if k in ("hands", "feet")}
    return raw, events, event_id


def get_epochs_for_eval(dataset: str, subject_id: int):
    """
    加载指定数据集+受试者并做与训练一致的预处理，得到 (X, y)。
    用于分析阶段在多数据集上评估泛化能力。若数据不存在或出错则返回 None。
    返回 (X, y) 或 None；X shape (n_epochs, n_channels, n_times)。
    """
    try:
        if dataset == "2a":
            raw, events, event_id = load_bci_iv_2a(subject_id)
        elif dataset == "2b":
            raw, events, event_id = load_bci_iv_2b(subject_id)
        elif dataset == "eegbci":
            raw, events, event_id = load_physionet_eegbci(subject_id)
        else:
            return None
        bandpass_filter(raw, l_freq=BAND_LOW_HZ, h_freq=BAND_HIGH_HZ)
        epochs = get_epochs(raw, events, event_id, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX)
        X, y = epochs_to_arrays(epochs, tmin=EPOCH_TRAIN_TMIN, tmax=EPOCH_TRAIN_TMAX)
        if len(np.unique(y)) < 2:
            return None
        return X, y
    except Exception:
        return None


def train_one_subject(
    subject_id: int,
    dataset: str,
    n_splits: int = 10,
    dataset_suffix: str | None = None,
) -> dict:
    """
    对一名受试者加载数据、预处理，训练 LDA/SVM/RF，返回各模型 CV 平均准确率及拟合的 pipeline。
    返回: { "lda": (mean_acc, pipeline), "svm": ..., "rf": ... }
    """
    # 数据加载
    if dataset == "2a":
        raw, events, event_id = load_bci_iv_2a(subject_id)
    elif dataset == "2b":
        raw, events, event_id = load_bci_iv_2b(subject_id)
    else:
        raw, events, event_id = load_physionet_eegbci(subject_id)
    # 滤波 8–30 Hz
    bandpass_filter(raw, l_freq=BAND_LOW_HZ, h_freq=BAND_HIGH_HZ)
    epochs = get_epochs(raw, events, event_id, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX)
    X, y = epochs_to_arrays(epochs, tmin=EPOCH_TRAIN_TMIN, tmax=EPOCH_TRAIN_TMAX)
    if len(np.unique(y)) < 2:
        raise ValueError(f"Subject {subject_id}: 至少需要两类，当前 y 唯一值: {np.unique(y).tolist()}")

    results = {}
    name_suffix = f"_sub{subject_id}" + (f"_{dataset_suffix}" if dataset_suffix else "")
    for model_type in MODEL_TYPES:
        model_name = f"csp_{model_type}{name_suffix}"
        clf, scores = run_training(X, y, model_type=model_type, model_name=model_name)
        mean_acc = float(np.mean(scores))
        results[model_type] = (mean_acc, clf)
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="多模型训练：9 名受试者 × LDA/SVM/RF，保存最佳模型")
    parser.add_argument("--dataset", choices=["2a", "2b", "eegbci", "all"], default="2a",
                        help="数据集：2a, 2b, eegbci；all=依次训练 2a(1-9), 2b(1-9), eegbci(1)，模型名带 _2a/_2b/_eegbci 后缀")
    parser.add_argument("--subjects", type=str, default="1-9",
                        help="受试者 ID 范围，如 1-9 或 1,2,3（--dataset all 时忽略，使用各数据集默认）")
    args = parser.parse_args()

    def run_one_dataset(dataset: str, subject_ids: list, dataset_suffix: str | None = None):
        """训练指定数据集下的所有受试者；dataset_suffix 用于保存名（all 模式下为 2a/2b/eegbci）。"""
        all_scores = []
        for sid in subject_ids:
            try:
                results = train_one_subject(sid, dataset, n_splits=10, dataset_suffix=dataset_suffix)
            except FileNotFoundError as e:
                print(f"Subject {sid}: 跳过 — {e}")
                all_scores.append((sid, None, None, None, None))
                continue
            except ValueError as e:
                print(f"Subject {sid}: 跳过 — {e}")
                all_scores.append((sid, None, None, None, None))
                continue
            best_type = max(results, key=lambda k: results[k][0])
            best_acc = results[best_type][0]
            best_clf = results[best_type][1]
            best_name = get_best_model_name_for_subject(sid, dataset_suffix)
            save_model(best_clf, best_name)
            print(f"  -> 最佳模型: {best_type.upper()} ({best_acc:.2%})，已保存为 {best_name}.pkl")
            row = (sid, results["lda"][0], results["svm"][0], results["rf"][0], best_type)
            all_scores.append(row)
        return all_scores, subject_ids, dataset

    # 解析受试者列表（单数据集时用）
    if "-" in args.subjects:
        a, b = args.subjects.strip().split("-")
        subject_ids_single = list(range(int(a), int(b) + 1))
    else:
        subject_ids_single = [int(x) for x in args.subjects.split(",")]

    models_dir = get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "all":
        # 2a: 1-9, 2b: 1-9, eegbci: 1
        runs = [
            ("2a", list(range(1, 10)), "2a"),
            ("2b", list(range(1, 10)), "2b"),
            ("eegbci", [1], "eegbci"),
        ]
        for dataset, subject_ids, suffix in runs:
            print(f"\n数据集: {dataset}, 受试者: {subject_ids} (模型名带 _{suffix})")
            print("=" * 60)
            all_scores, _, _ = run_one_dataset(dataset, subject_ids, dataset_suffix=suffix)
            print()
            print("受试者 | LDA    | SVM    | RF     | Best")
            print("-" * 50)
            for row in all_scores:
                if row[1] is None:
                    print(f"  {row[0]}     | (skip)")
                else:
                    sid, lda, svm, rf, best = row
                    print(f"  {sid}     | {lda:.2%} | {svm:.2%} | {rf:.2%} | {best.upper()}")
        # Replay: 用 2a 受试者 1（与单数据集行为一致时可改为最后一组）
        sid0 = 1
        try:
            raw, events, event_id = load_bci_iv_2a(sid0)
            bandpass_filter(raw, l_freq=BAND_LOW_HZ, h_freq=BAND_HIGH_HZ)
            epochs = get_epochs(raw, events, event_id, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX)
            X, y = epochs_to_arrays(epochs, tmin=EPOCH_TRAIN_TMIN, tmax=EPOCH_TRAIN_TMAX)
            replay_path = get_models_dir() / "replay_data.npz"
            np.savez_compressed(replay_path, X=X, y=y)
            print(f"\nReplay 数据已保存: {replay_path}（2a 受试者 {sid0}）")
        except Exception as e:
            print(f"\n未保存 replay_data.npz: {e}")
    else:
        dataset = args.dataset
        if dataset == "eegbci":
            subject_ids_single = subject_ids_single[:1] if subject_ids_single else [1]
            print("PhysioNet EEGBCI 仅使用 subject 1 进行演示。")
        print(f"数据集: {dataset}, 受试者: {subject_ids_single}")
        print("=" * 60)
        all_scores, _, _ = run_one_dataset(dataset, subject_ids_single, dataset_suffix=None)
        print()
        print("受试者 | LDA    | SVM    | RF     | Best")
        print("-" * 50)
        for row in all_scores:
            if row[1] is None:
                print(f"  {row[0]}     | (skip)")
            else:
                sid, lda, svm, rf, best = row
                print(f"  {sid}     | {lda:.2%} | {svm:.2%} | {rf:.2%} | {best.upper()}")
        if all_scores and all_scores[0][1] is not None:
            sid0 = subject_ids_single[0]
            try:
                if dataset == "2a":
                    raw, events, event_id = load_bci_iv_2a(sid0)
                elif dataset == "2b":
                    raw, events, event_id = load_bci_iv_2b(sid0)
                else:
                    raw, events, event_id = load_physionet_eegbci(sid0)
                bandpass_filter(raw, l_freq=BAND_LOW_HZ, h_freq=BAND_HIGH_HZ)
                epochs = get_epochs(raw, events, event_id, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX)
                X, y = epochs_to_arrays(epochs, tmin=EPOCH_TRAIN_TMIN, tmax=EPOCH_TRAIN_TMAX)
                replay_path = get_models_dir() / "replay_data.npz"
                np.savez_compressed(replay_path, X=X, y=y)
                print(f"\nReplay 数据已保存: {replay_path}（受试者 {sid0}）")
            except Exception as e:
                print(f"\n未保存 replay_data.npz: {e}")

    print("\n完成。可用 /analyze-model 分析模型（含图像与 models/analysis/report.txt），/run-replay 进行 LSL 演示。")


if __name__ == "__main__":
    main()
