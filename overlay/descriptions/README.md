# Frozen description banks

- `cifar100_visual_attributes_v1.json` provides eight class-level visual
  descriptions for each CIFAR-100 class and is distributed with CAAR.
- ImageNet-R uses descriptions from the public
  [CuPL](https://github.com/sarahpratt/CuPL) project. The CuPL descriptions are
  not redistributed here. Run `tools/build_imagenet_r_cupl_bank.py` with a
  local CuPL checkout to create `imagenet_r_cupl_attributes_v1.json` and
  `imagenet_r_classes_v1.json`.

The descriptions are fixed before continual training and are shared by all
tasks and ablation variants. They provide class-level auxiliary knowledge and
do not use individual training or test images. Validate the files with:

```bash
python tools/validate_semantic_descriptions.py
```
