import os
import argparse

from SensorData import SensorData


parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--dataset_path', required=True, help='path to dataset folder')
parser.add_argument('--output_path', default=None, help='path to output folder, defaults to dataset_path if not given')

opt = parser.parse_args()
print(opt)


def save_poses(filename, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # load the data
    print('loading %s...' % filename, flush=True)
    sd = SensorData(filename)
    print('loaded!', flush=True)
    sd.export_poses(output_path)


if __name__ == '__main__':
    dataset_path = opt.dataset_path
    output_path = opt.output_path if opt.output_path is not None else dataset_path

    for split in ['scans', 'scans_test']:
        base = os.path.join(dataset_path, split)
        output_path_tmp = os.path.join(output_path, split)
        scenes = sorted(os.listdir(base))

        for scene in scenes:
            if not scene.startswith('scene'):
                continue
            print('processing', scene, flush=True)
            save_poses(os.path.join(base, scene, scene + '.sens'), 
                       os.path.join(output_path_tmp, scene, 'pose'))