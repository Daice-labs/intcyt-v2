"""Modernized entry point for running the legacy IntCyt training loop.

The original script relied on implicit ``sys.argv`` access, binary/text mixes for gzip
files, and global side effects.  This rewrite preserves the documented CLI
(``python main.py DATASET [OPTIONS]``) while making the data loading and I/O explicit so
the module can be imported or unit-tested from the new codebase.
"""

from __future__ import annotations

import argparse
import gzip
import math
import random
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio

from intcyt import Cell, Operad, SuperCell, debug_time, intcyt as run_intcyt, usf

DATASET_CONFIG = {
    "SVHN": {"image_size": (32, 32, 3), "ary": 20},
    "MNIST": {"image_size": (28, 28, 1), "ary": 20},
    "fashion-MNIST": {"image_size": (28, 28, 1), "ary": 20},
    "DREAM3": {"image_size": (64, 1, 1), "ary": 50},
}

FASHION_LABELS = [
    "t-shirt",
    "trousers",
    "pullover",
    "dress",
    "jacket",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle-boot",
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the legacy IntCyt training loop on a supported dataset.",
    )
    parser.add_argument("dataset", choices=DATASET_CONFIG.keys())
    parser.add_argument(
        "-iterations",
        type=int,
        default=30000,
        help="Number of learning iterations to perform (default: 30000).",
    )
    parser.add_argument(
        "-seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducibility.",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-load",
        dest="load_index",
        nargs="?",
        const=0,
        type=int,
        help="Load initial organelles from result-load/load_initial.gz (optional index).",
    )
    group.add_argument(
        "-load-selfsup-right",
        dest="load_selfsup_right",
        nargs="?",
        const=0,
        type=int,
        help="Load a right-side self-supervised challenge (optional index).",
    )
    group.add_argument(
        "-load-selfsup-left",
        dest="load_selfsup_left",
        nargs="?",
        const=0,
        type=int,
        help="Load a left-side self-supervised challenge (optional index).",
    )

    parser.add_argument(
        "--result-load",
        type=Path,
        default=Path("result-load") / "load_initial.gz",
        help="Path to the gzip file that stores serialized initial organelles.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory that contains the dataset files (train_32x32.mat, etc.).",
    )
    parser.add_argument(
        "--save-root",
        type=Path,
        default=Path("."),
        help="Directory where save_roo.gz/save_org.gz/save_tre.gz are written.",
    )
    parser.add_argument(
        "--dream3-training",
        type=Path,
        default=Path("data") / "dream3_training.gz",
        help="Location of the DREAM3 preprocessed training file.",
    )
    return parser.parse_args(argv)


def _load_svhn(data_dir: Path) -> Dict[str, np.ndarray]:
    mat_path = data_dir / "train_32x32.mat"
    dataset = sio.loadmat(mat_path)
    return {"images": dataset["X"], "labels": dataset["y"]}


def _open_mnist_files(
    data_dir: Path,
    image_file: str,
    label_file: str,
    stack: ExitStack,
) -> Dict[str, gzip.GzipFile]:
    images = stack.enter_context(gzip.open(data_dir / image_file, "rb"))
    labels = stack.enter_context(gzip.open(data_dir / label_file, "rb"))
    labels.read(8)
    images.read(16)
    return {"images": images, "labels": labels}


def _load_dream3(training_file: Path) -> np.ndarray:
    with gzip.open(training_file, "rt", encoding="utf-8") as handle:
        return np.array(usf.get_memory(handle, 1, list(range(9285))))


def _resolve_load_mode(args: argparse.Namespace) -> Tuple[Optional[str], Optional[int]]:
    if args.load_index is not None:
        return "load", args.load_index
    if args.load_selfsup_right is not None:
        return "selfsup-right", args.load_selfsup_right
    if args.load_selfsup_left is not None:
        return "selfsup-left", args.load_selfsup_left
    return None, None


def _self_supervision_window(
    dataset: str,
    load_mode: Optional[str],
    image_size: Tuple[int, int, int],
    dim: int,
) -> List[List[int]]:
    if dataset == "DREAM3":
        return [[16, dim - 1, dim]]
    if load_mode == "selfsup-right":
        width = image_size[1]
        return [[width // 2, width - 1, width]]
    if load_mode == "selfsup-left":
        width = image_size[1]
        return [[0, width // 2 - 1, width]]
    return []


def _load_initial_supercell(
    dim: int,
    ary: int,
    path: Path,
    index: int,
) -> SuperCell:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = usf.get_memory(handle, ary, [index])
    if not payload:
        raise ValueError(f"no initialization found at index {index} in {path}")
    cell = Cell(dim, 0, [0] * dim, payload[0])
    return SuperCell(cell)


def _prepare_output_files(save_root: Path, stack: ExitStack) -> Tuple[gzip.GzipFile, ...]:
    save_root.mkdir(parents=True, exist_ok=True)
    roo = stack.enter_context(gzip.open(save_root / "save_roo.gz", "wt", encoding="utf-8"))
    org = stack.enter_context(gzip.open(save_root / "save_org.gz", "wt", encoding="utf-8"))
    tre = stack.enter_context(gzip.open(save_root / "save_tre.gz", "wt", encoding="utf-8"))
    return roo, org, tre


def _normalize_inputs(inputs: np.ndarray) -> np.ndarray:
    maximum = float(np.max(inputs))
    if maximum == 0:
        return inputs.copy()
    return (0.0001 / maximum) * inputs


def _dataset_label_and_input(
    dataset: str,
    resources: Dict[str, object],
    iteration: int,
    dim: int,
) -> Tuple[str, np.ndarray]:
    if dataset == "SVHN":
        idx = iteration % resources["images"].shape[3]
        label = ", ".join(map(str, resources["labels"][idx][0]))
        inputs = resources["images"][:, :, :, idx].reshape(dim)
        return label, inputs.astype(np.float32)

    if dataset == "MNIST":
        buf_lab = resources["labels"].read(1)
        buf_inp = resources["images"].read(dim)
        label = ", ".join(map(str, np.frombuffer(buf_lab, dtype=np.uint8).astype(np.int64)))
        inputs = np.frombuffer(buf_inp, dtype=np.uint8).astype(np.float32)
        return label, inputs

    if dataset == "fashion-MNIST":
        buf_lab = resources["labels"].read(1)
        buf_inp = resources["images"].read(dim)
        indices = np.frombuffer(buf_lab, dtype=np.uint8).astype(np.int64)
        label = ", ".join(FASHION_LABELS[idx] for idx in indices)
        inputs = np.frombuffer(buf_inp, dtype=np.uint8).astype(np.float32)
        return label, inputs

    training_images = resources["listim"].shape[0]
    label = resources["categories"][iteration % training_images]
    inputs = resources["listim"][iteration % training_images][0]
    return label, inputs.astype(np.float32)


def configure_resources(
    args: argparse.Namespace,
    stack: ExitStack,
) -> Dict[str, object]:
    dataset = args.dataset
    config = DATASET_CONFIG[dataset]
    resources: Dict[str, object] = {"ary": config["ary"], "image_size": config["image_size"]}

    if dataset == "SVHN":
        payload = _load_svhn(args.data_dir)
        resources.update({"images": payload["images"], "labels": payload["labels"]})
    elif dataset == "MNIST":
        payload = _open_mnist_files(
            args.data_dir,
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            stack,
        )
        resources.update(payload)
    elif dataset == "fashion-MNIST":
        payload = _open_mnist_files(
            args.data_dir,
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
            stack,
        )
        resources.update(payload)
    else:  # DREAM3
        listim = _load_dream3(args.dream3_training)
        training_images = listim.shape[0]
        resources.update(
            {
                "listim": listim,
                "categories": list(map(str, range(training_images))),
            }
        )
    return resources


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    load_mode, load_index = _resolve_load_mode(args)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    with ExitStack() as stack:
        resources = configure_resources(args, stack)
        dim = math.prod(resources["image_size"])
        operad = Operad(dim)

        if load_mode:
            sc = _load_initial_supercell(dim, resources["ary"], args.result_load, load_index or 0)
        else:
            sc = operad.generate(levels=0, arity=resources["ary"], key=lambda: random.randint(0, 30))

        save_files = _prepare_output_files(args.save_root, stack)
        fsave_roo, fsave_org, fsave_tre = save_files

        usf.print_root(sc, resources["image_size"], fsave_roo, option="save")
        usf.print_organelles(sc, resources["image_size"], fsave_org, option="save")
        usf.print_tree(sc, resources["image_size"], fsave_tre, option="save")

        selfsup = _self_supervision_window(
            args.dataset,
            load_mode,
            resources["image_size"],
            dim,
        )

        iterations = args.iterations
        c_info: List[List[object]] = []
        start_time = time.time()

        for i in range(iterations):
            label, inputs = _dataset_label_and_input(args.dataset, resources, i, dim)
            inputs = _normalize_inputs(inputs)
            vector = inputs.tolist()

            debug_time.set(f"Learning data labeled as {label}")
            step = i / float(iterations)
            interval = lambda s, a, b: a * (1 - s) + b * s if s < 1 else b  # noqa: E731

            start = 4
            epoch = 4
            fission_events = [0]
            fusion_events = [2]
            compose_events = [1, 3]
            events = [start, epoch, fission_events, fusion_events, compose_events]
            filtering = [1.5, 1.5, interval(step, 10, 2)]

            for k in range(epoch):
                debug_time.set("TREE")
                sc.stdout(vector)

                brightness, profiles, scores, E, F = determine_contrast_profiles(
                    args.dataset,
                    i,
                    resources,
                )
                gamma_parameter = usf.gamma(E, F, brightness, profiles, scores, *selfsup)
                run_intcyt(operad, sc, epoch * i + k, events, vector, gamma_parameter, filtering)

                usf.print_root(sc, resources["image_size"], fsave_roo, option="save")
                usf.print_organelles(sc, resources["image_size"], fsave_org, option="save")
                usf.print_tree(sc, resources["image_size"], fsave_tre, option="save")

                cycle = 4 * i + k
                if cycle % 1000 == 0:
                    c_info.append([cycle, filtering, step, scores, E])
                current = c_info + [[cycle, filtering, step, scores, E]]
                with open(args.save_root / "c_info.txt", "w", encoding="utf-8") as info_file:
                    for record in current:
                        info_file.write(f"cycle number = {record[0]}\n")
                        info_file.write(f"filtering = {record[1]}\n")
                        info_file.write(f"step = {record[2]}\n")
                        info_file.write(f"score = {record[3]}\n")
                        info_file.write(f"E = {record[4]}\n")

        print("time =", time.time() - start_time)
    return 0


def determine_contrast_profiles(
    dataset: str,
    iteration: int,
    resources: Dict[str, object],
) -> Tuple[List[float], List[List[Tuple[float, float]]], List[Tuple[float, float]], float, float]:
    if dataset == "SVHN":
        brightness = [0.1, 0.25, 0.5, 0.75, 0.9]
        profiles = [[(0, 0.6), (0, 0.375), (0, 0.15), (0, 0.05), (0, 0.005)]]
        scores = [(0.82, 1)]
        return brightness, profiles, scores, 13, 25

    if dataset == "MNIST":
        brightness = [0.1, 0.25, 0.5, 0.75, 0.9]
        profiles = [[(0, 0.45), (0, 0.25), (0, 0.1), (0, 0.05), (0, 0.01)], [(0, 0.3), (0, 0.3), (0, 0.1), (0, 0.05), (0, 0.01)]]
        scores = [(0.7, 1), (0.8, 1)]
        return brightness, profiles, scores, 14, 25

    if dataset == "fashion-MNIST":
        brightness = [0.1, 0.25, 0.5, 0.75, 0.9]
        profiles = [[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1)], [(0, 0.7), (0, 0.6), (0, 0.5), (0, 0.4), (0, 0.3)]]
        scores = [(0.85, 1), (0.95, 1)]
        return brightness, profiles, scores, 13.5, 22.5

    training_images = resources["listim"].shape[0]
    brightness = [0.98]
    if iteration < training_images:
        profiles = [[(0.85, 1)], [(0, 0.85)]]
        scores = [(0.9, 1), (0.98, 1)]
        return brightness, profiles, scores, 11.0, 20
    if iteration < 2 * training_images - 1:
        profiles = [[(0.85, 1)], [(0, 0.85)]]
        scores = [(0.85, 1), (0.98, 1)]
        return brightness, profiles, scores, 11.25, 20
    profiles = [[(0.85, 1)], [(0, 0.85)]]
    scores = [(0.8, 1), (0.98, 1)]
    return brightness, profiles, scores, 11.5, 20


if __name__ == "__main__":
    raise SystemExit(main())
