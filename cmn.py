from nvidia.dali import pipeline_def, types, fn
import numpy as np
from timeit import default_timer as timer

import argparse

parser = argparse.ArgumentParser(description='Bench.')

parser.add_argument('-i', '--iters', type=int,
                    help='How many iters', default=3)
args = parser.parse_args()


num_iters = args.iters
shape = [1000, 1000, 3]
batch_size = 64

@pipeline_def(num_threads=4, device_id=0)
def pipe1(option=1):
    images = types.Constant(np.zeros(shape, dtype=np.uint8), device='gpu')
    images = fn.reshape(images, layout="HWC")
    if option == 0:
        print("CMN-current")
        out = fn.crop_mirror_normalize(
            images,
            output_layout="CHW",
            run_old=True,
            mean=[128., 128., 128.], std=[1., 1., 1.]
        )
    if option == 1:
        print("CMN-experiment")
        out = fn.crop_mirror_normalize(
            images,
            output_layout="CHW",
            run_old=False,
            mean=[128., 128., 128.], std=[1., 1., 1.]
        )
    elif option == 2:
        print("Transpose + Normalize")
        out = fn.normalize(
            fn.transpose(images, perm=[2,0,1]),
            axes=[1,2],
            mean=np.array([[[128.]], [[128.]], [[128.]]], dtype=np.float32),
            stddev=np.array([[[1.]], [[1.]], [[1.]]], dtype=np.float32)
        )
    elif option == 3:
        print("Transpose")
        out = fn.transpose(images, perm=[2,0,1])
    elif option == 4:
        print("Normalize")
        out = fn.normalize(images,
            axes=[0, 1],
            mean=np.array([[[128., 128., 128.]]], dtype=np.float32),
            stddev=np.array([[[1., 1., 1.]]], dtype=np.float32)
        )
    elif option == 5:
        print("Copy")
        out = fn.copy(images)
    elif option == 6:
        print("Add")
        out = images + 100
    # elif option == 7:
    #     print("New CMN")
    #     out = fn.crop_mirror_normalize(
    #         images,
    #         output_layout="CHW",
    #         mean=[128., 128., 128.], std=[1., 1., 1.]
    #     )
    elif option == 7:
        print("No-op (returning constants)")
        out = images
    elif option == 8:
        print("Cast to float")
        out = fn.cast(images, dtype=types.FLOAT)
    return out

for o in [7, 7, 0, 1, 2, 3, 4, 5, 6, 8]:
    p = pipe1(option=o, batch_size=batch_size)
    p.build()

    start_time = timer()
    for _ in range(num_iters):
        out, = p.run()
    total_time = timer() - start_time

    in_size = 1000 * 1000 * 3 * 1
    if o == 3 or o == 5 or o == 6 or o == 8:
        out_size = 1000 * 1000 * 3 * 1
    else:
        out_size = 1000 * 1000 * 3 * 4

    throughput = float(batch_size * num_iters * (out_size + in_size)) / total_time
    print("{} GB/s".format(throughput / 1e9))