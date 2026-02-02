import gzip
import random
import sys
from typing import Callable, Generator, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio

from intcyt import debug_time, usf

SAMPLE_CAP = 1000
NOISY_FLAGS = {"-right-noisy", "-left-noisy"}
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


def _image_size(dataset: str) -> Optional[List[int]]:
  if dataset == "SVHN":
    return [32, 32, 3]
  if dataset in {"MNIST", "fashion-MNIST"}:
    return [28, 28, 1]
  return None


def _iter_svhn(dim: int) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
  data = sio.loadmat("data/train_32x32.mat")
  images = data["X"]
  labels = data["y"]
  for idx in range(min(SAMPLE_CAP, images.shape[3])):
    yield labels[idx], images[:, :, :, idx].reshape(dim)


def _iter_idx_dataset(
  image_path: str,
  label_path: str,
  dim: int,
  label_transform: Optional[Callable[[np.ndarray], Sequence]] = None,
) -> Generator[Tuple[Sequence, np.ndarray], None, None]:
  with gzip.open(image_path, "rb") as image_file, gzip.open(label_path, "rb") as label_file:
    label_file.read(8)
    image_file.read(16)
    for _ in range(SAMPLE_CAP):
      buf_lab = label_file.read(1)
      buf_inp = image_file.read(dim)
      if not buf_lab or len(buf_inp) < dim:
        break
      raw_label = np.frombuffer(buf_lab, dtype=np.uint8).astype(np.int64)
      label = label_transform(raw_label) if label_transform else raw_label
      inputs = np.frombuffer(buf_inp, dtype=np.uint8).astype(np.float32)
      yield label, inputs


def _fashion_label_transform(raw_label: np.ndarray) -> List[str]:
  return [FASHION_LABELS[int(idx)] for idx in raw_label.flatten().tolist()]


def _format_label(label: Sequence) -> str:
  if isinstance(label, np.ndarray):
    iterable = label.flatten().tolist()
  elif isinstance(label, (list, tuple)):
    iterable = label
  else:
    iterable = [label]
  return ", ".join(map(str, iterable))


def _region_bounds(flag: Optional[str], flatten_width: int) -> Optional[Tuple[int, int]]:
  if not flag:
    return None
  if flag.startswith("-right"):
    return (0, max(0, flatten_width // 2 - 1))
  if flag.startswith("-left"):
    return (flatten_width // 2, flatten_width - 1)
  return None


def _apply_randomization(
  pixels: List[float],
  flatten_width: int,
  region_bounds: Optional[Tuple[int, int]],
  flag: Optional[str],
  noise_mode: Optional[str],
  noise_params: Optional[Tuple[int, int]],
) -> None:
  for idx, value in enumerate(pixels):
    column = idx % flatten_width
    if region_bounds and region_bounds[0] <= column <= region_bounds[1]:
      pixels[idx] = random.randint(0, 250)
    elif flag in NOISY_FLAGS and value == 0:
      if noise_mode == "zeros":
        pixels[idx] = random.randint(0, 250)
      elif noise_mode == "probability" and noise_params:
        hits, upper = noise_params
        if 1 <= random.randint(0, upper) <= hits:
          pixels[idx] = random.randint(0, 250)


def _record_stream(dataset: str, dim: int):
  if dataset == "SVHN":
    return _iter_svhn(dim)
  if dataset == "MNIST":
    return _iter_idx_dataset(
      "data/train-images-idx3-ubyte.gz",
      "data/train-labels-idx1-ubyte.gz",
      dim,
    )
  if dataset == "fashion-MNIST":
    return _iter_idx_dataset(
      "data/t10k-images-idx3-ubyte.gz",
      "data/t10k-labels-idx1-ubyte.gz",
      dim,
      label_transform=_fashion_label_transform,
    )
  return None


def main(argv: Optional[Sequence[str]] = None) -> int:
  argv = sys.argv[1:] if argv is None else list(argv)
  if len(argv) < 2:
    print("Usage: challenge.py DATASET ARY [REGION] [NOISE_LOW NOISE_HIGH]")
    return 1

  dataset = argv[0]
  try:
    ary = int(argv[1])
  except ValueError:
    print("Error: challenge.py: parameter [ary] must be an integer value")
    return 1

  if not 0 < ary <= SAMPLE_CAP:
    print(f"Error: challenge.py: parameter [ary] must be between 1 and {SAMPLE_CAP}")
    return 1

  region_flag = argv[2] if len(argv) > 2 else None
  extra_args = argv[3:] if len(argv) > 3 else []

  image_size = _image_size(dataset)
  if image_size is None:
    print(f"Error: challenge.py: unsupported dataset '{dataset}'")
    return 1

  dim = image_size[0] * image_size[1] * image_size[2]
  flatten_width = image_size[1] * image_size[2]

  region_bounds = _region_bounds(region_flag, flatten_width)
  if region_flag and region_bounds is None:
    print(f"Error: challenge.py: unsupported region flag '{region_flag}'")
    return 1

  noise_mode = None
  noise_params = None
  if region_flag in NOISY_FLAGS:
    if not extra_args:
      noise_mode = "zeros"
    elif len(extra_args) >= 2:
      try:
        hits = int(extra_args[0])
        upper = int(extra_args[1])
      except ValueError:
        print("Error: challenge.py: noise parameters must be integers")
        return 1
      if hits <= 0 or upper <= 0:
        print("Error: challenge.py: noise parameters must be positive")
        return 1
      noise_mode = "probability"
      noise_params = (hits, upper)
    else:
      print("Error: challenge.py: noisy flags require zero or two numeric parameters")
      return 1

  init_indices = sorted(random.sample(range(SAMPLE_CAP), ary))
  pending = set(init_indices)

  records = _record_stream(dataset, dim)
  if records is None:
    print(f"Error: challenge.py: unable to prepare stream for dataset '{dataset}'")
    return 1

  with gzip.open("result-load/load_initial.gz", "wt", encoding="utf-8") as fself:
    for idx, (label, inputs) in enumerate(records):
      if idx not in pending:
        continue

      debug_time.set("Data picked: " + _format_label(label))
      pixels = inputs.tolist()
      _apply_randomization(pixels, flatten_width, region_bounds, region_flag, noise_mode, noise_params)

      usf.print_data(pixels, image_size, sys.stdout, option="display")
      usf.print_data(pixels, image_size, fself, option="save")
      fself.write("\n")

      pending.remove(idx)
      if not pending:
        break

  if pending:
    missing = len(pending)
    print(f"Warning: challenge.py: only {ary - missing} samples written (dataset exhausted)")

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
