from __future__ import annotations

import json
import random
import shutil
from pathlib import Path


CONFIG_JSON = """
{
	"input_hr_dir": "./hr",
	"input_lr_dir": "./lr",
	"output_train_dir": "E:/FAST/SUPARCO/KAIR/trainsets/airbus",
	"output_test_dir": "E:/FAST/SUPARCO/KAIR/testsets/airbus",
  "test_size": 0.2,
  "seed": 42
}
"""


def load_config() -> dict:
	cfg = json.loads(CONFIG_JSON)

	required = [
		"input_hr_dir",
		"input_lr_dir",
		"output_train_dir",
		"output_test_dir",
		"test_size",
		"seed",
	]
	missing = [k for k in required if k not in cfg]
	if missing:
		raise ValueError(f"Missing config keys: {', '.join(missing)}")

	test_size = float(cfg["test_size"])
	if not (0.0 < test_size < 1.0):
		raise ValueError("test_size must be between 0 and 1 (exclusive).")

	cfg["test_size"] = test_size
	cfg["seed"] = int(cfg["seed"])
	return cfg


def ensure_clean_dir(path: Path) -> None:
	if path.exists():
		shutil.rmtree(path)
	path.mkdir(parents=True, exist_ok=True)


def copy_pairs(filenames: list[str], hr_src: Path, lr_src: Path, hr_dst: Path, lr_dst: Path) -> None:
	for name in filenames:
		shutil.copy2(hr_src / name, hr_dst / name)
		shutil.copy2(lr_src / name, lr_dst / name)


def main() -> None:
	cfg = load_config()

	base_dir = Path(__file__).resolve().parent
	hr_dir = (base_dir / cfg["input_hr_dir"]).resolve()
	lr_dir = (base_dir / cfg["input_lr_dir"]).resolve()
	train_root = (base_dir / cfg["output_train_dir"]).resolve()
	test_root = (base_dir / cfg["output_test_dir"]).resolve()

	if not hr_dir.exists() or not lr_dir.exists():
		raise FileNotFoundError("Configured input directories do not exist.")

	hr_names = {p.name for p in hr_dir.glob("*.png")}
	lr_names = {p.name for p in lr_dir.glob("*.png")}

	common = sorted(hr_names & lr_names)
	if not common:
		raise RuntimeError("No matching PNG filename pairs found between HR and LR folders.")

	missing_hr = sorted(lr_names - hr_names)
	missing_lr = sorted(hr_names - lr_names)
	if missing_hr:
		print(f"Warning: {len(missing_hr)} files exist in LR but missing in HR. Ignoring those.")
	if missing_lr:
		print(f"Warning: {len(missing_lr)} files exist in HR but missing in LR. Ignoring those.")

	rng = random.Random(cfg["seed"])
	rng.shuffle(common)

	n_total = len(common)
	n_test = max(1, int(round(n_total * cfg["test_size"])))
	n_test = min(n_test, n_total - 1) if n_total > 1 else 1

	test_files = common[:n_test]
	train_files = common[n_test:]

	train_hr = train_root / "hr"
	train_lr = train_root / "lr"
	test_hr = test_root / "hr"
	test_lr = test_root / "lr"

	for path in [train_hr, train_lr, test_hr, test_lr]:
		ensure_clean_dir(path)

	copy_pairs(train_files, hr_dir, lr_dir, train_hr, train_lr)
	copy_pairs(test_files, hr_dir, lr_dir, test_hr, test_lr)

	manifest = {
		"config": cfg,
		"total_pairs": n_total,
		"train_count": len(train_files),
		"test_count": len(test_files),
		"train_files": train_files,
		"test_files": test_files,
	}
	(base_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

	print("Split complete")
	print(f"Input HR: {hr_dir}")
	print(f"Input LR: {lr_dir}")
	print(f"Train out: {train_root}")
	print(f"Test out:  {test_root}")
	print(f"Total:    {n_total}")
	print(f"Train:    {len(train_files)}")
	print(f"Test:     {len(test_files)}")


if __name__ == "__main__":
	main()
