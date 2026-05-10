# from __future__ import absolute_import, division, print_function
from trainer import Trainer
from options import MonodepthOptions
import sys
import argparse

options = MonodepthOptions()

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)

if __name__ == "__main__":
    options.parser.convert_arg_line_to_args = convert_arg_line_to_args
    if sys.argv.__len__() >= 2 and not sys.argv[1].startswith("-"):
        arg_filename_with_prefix = '@' + sys.argv[1]
        cli_overrides = sys.argv[2:]
        opts = options.parser.parse_args([arg_filename_with_prefix] + cli_overrides)
    else:
        opts = options.parser.parse_args()
    trainer = Trainer(opts)
    trainer.train()
