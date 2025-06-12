# pylint: disable=too-many-branches, too-many-statements, import-outside-toplevel

import argparse
import sys, os
from openpifpaf import decoder, network, visualizer, show, logger

# Get the directory of the current Python file
current_dir = os.path.dirname(os.path.abspath(__file__))
monoloco_dir = os.path.dirname(os.path.dirname(current_dir))

def cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparsers = parser.add_subparsers(help='Different parsers for main actions', dest='command')
    predict_parser = subparsers.add_parser("predict")

    predict_parser.add_argument('--image', type=str, default="", help='Path to input image')
    predict_parser.add_argument('--model', type=str, default=os.path.join(monoloco_dir,'models/monoloco_pp-201207-1350.pkl'), help='Path to .pkl MonoLoco++ model')
    predict_parser.add_argument('--checkpoint', type=str, default=os.path.join(monoloco_dir,'models/shufflenetv2k30-201104-224654-cocokp-d75ed641.pkl'), help='Path to .pkl Pifpaf model')

    parser.add_argument('--conf', type=str, default=os.path.join(monoloco_dir,'_monoloco/config/camera_intrinsic.yaml'), help='Path to camera intrinsics YAML file')
    parser.add_argument('--robot', type=str, default='scand_jackal', help='Robot name key in YAML file')

    parser.add_argument('--min_conf', type=float, default=0., help='Minimum confident value for keyponts detection')

    # Predict (2D pose and/or 3D location from images)
    predict_parser.add_argument('--output_types', nargs='+', default= [],
                                help='MonoLoco - what to output: json bird front or multi')
    predict_parser.add_argument('--json-output', default=None, nargs='?', const=True,
                                help='OpenpifPaf - whether to output a json file,'
                                     'with the option to specify the output path or directory')
    predict_parser.add_argument('--no_save', help='to show images', action='store_true')
    predict_parser.add_argument('--hide_distance', help='to not show the absolute distance of people from the camera',
                                default=False, action='store_true')
    predict_parser.add_argument('--dpi', help='image resolution', type=int, default=100)
    predict_parser.add_argument('--long-edge', default=None, type=int,
                                help='rescale the long side of the image (aspect ratio maintained)')
    predict_parser.add_argument('--white-overlay',
                                nargs='?', default=False, const=0.8, type=float,
                                help='increase contrast to annotations by making image whiter')
    predict_parser.add_argument('--font-size', default=0, type=int, help='annotation font size')
    predict_parser.add_argument('--monocolor-connections', default=False, action='store_true',
                                help='use a single color per instance')
    predict_parser.add_argument('--instance-threshold', type=float, default=None, help='threshold for entire instance')
    predict_parser.add_argument('--seed-threshold', type=float, default=0.5, help='threshold for single seed')
    predict_parser.add_argument('--disable-cuda', action='store_true', help='disable CUDA')
    predict_parser.add_argument('--precise-rescaling', dest='fast_rescaling', default=True, action='store_false',
                                help='use more exact image rescaling (requires scipy)')
    predict_parser.add_argument('--decoder-workers', default=None, type=int,
                                help='number of workers for pose decoding, 0 for windows')

    decoder.cli(parser)
    logger.cli(parser)
    network.Factory.cli(parser)
    show.cli(parser)
    visualizer.cli(parser)

    # Monoloco
    predict_parser.add_argument('--activities', nargs='+', choices=['raise_hand', 'social_distance'],
                                help='Choose activities to show: social_distance, raise_hand', default=[])
    predict_parser.add_argument('--mode', help='keypoints, mono, stereo', default='mono')
    # parser.add_argument('--model', help='path of MonoLoco/MonStereo model to load')
    predict_parser.add_argument('--net', help='only to select older MonoLoco model, otherwise use --mode')
    predict_parser.add_argument('--path_gt', help='path of json file with gt 3d localization')
                                #default='data/arrays/names-kitti-200615-1022.json')
    predict_parser.add_argument('--z_max', type=int, help='maximum meters distance for predictions', default=100)
    predict_parser.add_argument('--n_dropout', type=int, help='Epistemic uncertainty evaluation', default=0)
    predict_parser.add_argument('--dropout', type=float, help='dropout parameter', default=0.2)
    predict_parser.add_argument('--show_all', help='only predict ground-truth matches or all', action='store_true')
    predict_parser.add_argument('--webcam', help='monstereo streaming', action='store_true')
    predict_parser.add_argument('--camera', help='device to use for webcam streaming', type=int, default=0)
    predict_parser.add_argument('--calibration', help='type of calibration camera, either custom, nuscenes, or kitti',
                                type=str, default='custom')
    predict_parser.add_argument('--focal_length',
                                help='foe a custom camera: focal length in mm for a sensor of 7.2x5.4 mm. (nuScenes)',
                                type=float, default=4.7)

    predict_parser.add_argument('--scale',
                                help='foe a custom camera: focal length in mm for a sensor of 7.2x5.4 mm. (nuScenes)',
                                type=float, default=1.5)

    if len(sys.argv) == 1 or sys.argv[1] not in ['predict']:
        sys.argv.insert(1, 'predict')

    args = parser.parse_args()
    return args

