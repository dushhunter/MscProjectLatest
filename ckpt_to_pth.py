import sys
import io
import os
import argparse
import torch
import torch.nn as nn
# import networks
#from options import MonodepthOptions
from collections import OrderedDict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from SQLdepth import SQLdepth, MonodepthOptions

def convert(opt, checkpoint_path, save_folder):
    model = SQLdepth(opt)
    print("loading checkpoint from {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model'] 

    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0) 

    model.load_state_dict(torch.load(buffer, map_location='cpu'))

    buffer.close()

    encoder_state_dict = model.encoder.state_dict()
    encoder_state_dict['height'] = opt.height
    encoder_state_dict['width'] = opt.width
    encoder_state_dict['use_stereo'] = opt.use_stereo
    decoder_state_dict = model.depth_decoder.state_dict()
    os.makedirs(save_folder, exist_ok=True) # mkdir if not exits

    encoder_save_path = os.path.join(save_folder, "encoder.pth")
    decoder_save_path = os.path.join(save_folder, "depth.pth")
    torch.save(encoder_state_dict, encoder_save_path)
    torch.save(decoder_state_dict, decoder_save_path)

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a fine-tune .pt checkpoint into encoder.pth/depth.pth files')
    parser.add_argument('sql_config', help='Path to the SQLdepth config file, e.g. ./conf/cvnXt.txt')
    parser.add_argument('checkpoint_path', help='Path to the fine-tuned .pt checkpoint')
    parser.add_argument('save_folder', help='Output folder where encoder.pth and depth.pth will be written')
    cli_args = parser.parse_args()

    SQLdepth_options = MonodepthOptions()
    SQLdepth_options.parser.convert_arg_line_to_args = convert_arg_line_to_args
    SQLdepth_opt_filename = '@' + cli_args.sql_config
    opt = SQLdepth_options.parser.parse_args([SQLdepth_opt_filename])
    opt.load_pretrained_model = False

    print("converting weights...")
    convert(opt, cli_args.checkpoint_path, cli_args.save_folder)
    print("done.")


