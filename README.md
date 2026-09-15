# Class-Anchored Attribute Residual Learning

This repository provides the CAAR extension for the official
[TOPECL](https://github.com/Thirtory/TOPECL) implementation. It does not
redistribute the TOPECL or CL_Pytorch source tree. Users obtain TOPECL from its
official repository and apply the CAAR overlay locally.

CAAR adds a lightweight residual branch guided by frozen class descriptions.
It selects four relevant and diverse attributes per class, removes the
class-name component, and pools the remaining directions for each image. A
zero-initialized bottleneck MLP learns a bounded differential score while the
native TOPECL classifier remains intact.

## Installation

Clone the pinned TOPECL revision and this repository side by side:

```bash
git clone https://github.com/Thirtory/TOPECL.git
git -C TOPECL checkout 0c655b136970a14e4cdd47c3bb2ea7ecb150ede6
git clone https://github.com/xianzhe678/CAAR.git

pip install -r TOPECL/requirement.txt
pip install -r CAAR/requirements-caar.txt
bash CAAR/scripts/install.sh TOPECL
```

The installer applies the minimal TOPECL integration patch and copies the
CAAR-only modules, configurations, tests, and description resources into the
local TOPECL checkout.

## Repository layout

- `overlay/`: CAAR-only files copied into a user-provided TOPECL checkout.
- `patches/topecl-caar.patch`: minimal integration changes for the pinned
  TOPECL revision.
- `scripts/install.sh`: local installer for the patch and overlay.
- `requirements-caar.txt`: dependencies not guaranteed by TOPECL.
- `THIRD_PARTY.md`: upstream projects and data provenance.

## Description banks

The CIFAR-100 class-level attribute bank is included in the overlay and does
not contain training or test images. ImageNet-R uses descriptions from CuPL;
those third-party descriptions are not redistributed here. Build the bank from
a local CuPL checkout:

```bash
git clone https://github.com/sarahpratt/CuPL.git
curl -L https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json \
  -o imagenet_class_index.json

cd TOPECL
python tools/build_imagenet_r_cupl_bank.py \
  --cupl-prompts ../CuPL/imagenet_prompts/CuPL_image_prompts.json \
  --cupl-classes ../CuPL/imagenet_classnames/imagenet_classes.py \
  --imagenet-index ../imagenet_class_index.json \
  --output descriptions/imagenet_r_cupl_attributes_v1.json \
  --mapping-output descriptions/imagenet_r_classes_v1.json

python tools/install_imagenet_r_metadata.py \
  --mapping descriptions/imagenet_r_classes_v1.json \
  --dataset-root "$DATA/imagenet_r"
```

## Evaluation

Run the paired four-group evaluation from the patched TOPECL directory:

```bash
cd TOPECL
export DATA=/path/to/datasets
python main.py --config \
  options/multi_steps/formal_topecl_attribute_ablation/cifar100_seed42_official.yaml

python main.py --config \
  options/multi_steps/formal_topecl_attribute_ablation/imagenet_r_seed42_official.yaml
```

The configurations use ten class-incremental tasks, seed 42, 20 training
epochs and five replay-only epochs per task, batch size 32, and CLIP ViT-B/16.
Datasets, logs, checkpoints, predictions, and result artifacts are excluded
from this repository.

## License and acknowledgement

The original CAAR additions are released under the MIT License. TOPECL,
CL_Pytorch, CuPL, and other dependencies remain subject to their respective
authors' terms. See [THIRD_PARTY.md](THIRD_PARTY.md) before use.
