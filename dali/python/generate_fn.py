import sys
import argparse



parser = argparse.ArgumentParser(description='Generate API stubs for DALI operators.')
parser.add_argument(
    '-p', '--build_path', required=True, help="Path to build directory with the wheel to be packed,"
    " for example DALI/build/dali/python")

args = parser.parse_args()

# Add the build dir path to the sys.path list, so the yet-to-package DALI can be imported
# and it will be the first one to do so.
sys.path.insert(0, args.build_path)

print(sys.path)
import nvidia.dali.fn._api_utils as api_utils

# import nvidia.dali

api_utils._generate_prototypes(args.build_path + "/nvidia/dali/")
