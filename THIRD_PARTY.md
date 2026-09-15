# Third-party components

CAAR is an extension for the following projects. Their source code is not
included in this repository.

- [TOPECL](https://github.com/Thirtory/TOPECL) provides the continual-learning
  baseline. Users clone it separately. `patches/topecl-caar.patch` targets
  revision `0c655b136970a14e4cdd47c3bb2ea7ecb150ede6` and contains only the
  integration changes required by CAAR.
- [CL_Pytorch](https://github.com/GiantJun/CL_Pytorch) is an upstream framework
  used by TOPECL. CAAR does not redistribute it directly.
- [CuPL](https://github.com/sarahpratt/CuPL) provides the ImageNet descriptions.
  CAAR does not redistribute the CuPL prompt bank; the included builder reads a
  user-provided CuPL checkout.
- [OpenAI CLIP](https://github.com/openai/CLIP) is installed as a dependency and
  remains under its own license.

The MIT License in this repository applies to the original CAAR additions. It
does not grant rights to third-party projects or third-party data.
