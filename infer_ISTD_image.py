import os
import sys
import time
from argparse import ArgumentParser

sys.path.append(os.getcwd())

from src.utils.data_loaders.ISTD_dataloader import ISTDDataLoaderVal

from easytorch import Runner, launch_runner
from easytorch.utils.misc import scan_dir

def parse_args():
    parser = ArgumentParser(description='Noise estimate')
    parser.add_argument('-c', '--cfg', help='net config', required=True)
    parser.add_argument('-i', '--input-path', help='input path', type=str, required=True)
    parser.add_argument('-o', '--output-dir', help='output dir', type=str, required=True)
    parser.add_argument('--max-frame-num', help='max frame num', type=int)
    parser.add_argument('--save-input', help='whether save gt and ldr', type=bool, default=False)
    parser.add_argument('--ckpt', help='ckpt path', type=str)
    parser.add_argument('--gpus', help='visible gpus', type=str)
    return parser.parse_args()

def main(_, runner: Runner, args):
    infer_set = ISTDDataLoaderVal(rgb_dir=args.input_path, img_options={'patch_size': 256})
    runner.init_logger(logger_name='easytorch-inference', log_file_name='infer_hisi_v2_result')
    runner.load_model(ckpt_path=args.ckpt)
    runner.model.eval()
    total_start_time = time.time()
    runner.infer_istd(infer_set, args.output_dir, args.max_frame_num)
    total_end_time = time.time()
    total_infer_time = total_end_time - total_start_time
    print(f"Total inference time: {total_infer_time:.4f} seconds for {len(infer_set)} images")
    if os.path.isdir(args.input_path):
        infer_set = list(scan_dir(args.input_path, '.raw', full_path=True))
        infer_set.sort()
    else:
        infer_set = (args.input_path,)
    runner.infer_istd(infer_set, args.output_dir, args.max_frame_num)
if __name__ == '__main__':
    args_ = parse_args()
    launch_runner(args_.cfg, main, (args_,), args_.gpus)
