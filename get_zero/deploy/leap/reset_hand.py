from get_zero.deploy.leap.util.get_zero_py_leap_client import MultiEmbodimentLeapHand
from get_zero.deploy.leap.util.get_zero_leap_asset_utils import get_leap_embodiment_properties
import time
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--embodiment_name', type=str, required=True)
    parser.add_argument('--duration', type=int, default=2)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    embodiment_properties = get_leap_embodiment_properties(args.embodiment_name)
    leap = MultiEmbodimentLeapHand(embodiment_properties) # this will command to initial joint position in initializer
    if args.duration == -1:
        while True:
            time.sleep(1)
    else:
        time.sleep(args.duration)
