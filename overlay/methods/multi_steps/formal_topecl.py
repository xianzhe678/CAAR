"""Public TOPECL training with formal three-head evaluation only."""

from argparse import ArgumentParser

from methods.multi_steps.formal_eval import (
    TOPECLFormalEvalMixin,
    add_formal_eval_args,
)
from methods.multi_steps.topecl import TOPECL, add_special_args as add_topecl_args


def add_special_args(parser: ArgumentParser) -> ArgumentParser:
    parser = add_topecl_args(parser)
    return add_formal_eval_args(parser)


class Formal_TOPECL(TOPECLFormalEvalMixin, TOPECL):
    """Keep TOPECL training unchanged and add reproducible evaluation output."""

    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._init_formal_evaluation()
