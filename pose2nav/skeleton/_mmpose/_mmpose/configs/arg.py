from argparse import ArgumentParser
import os

# Get the directory of the current Python file
current_dir = os.path.dirname(os.path.abspath(__file__))



# body_3d_keypoint/image-pose-lift_tcn_8xb64-200e_h36m.py
# https://download.openmmlab.com/mmpose/body3d/simple_baseline/simple3Dbaseline_h36m-f0ad73a4_20210419.pth

# body_3d_keypoint/video-pose-lift_tcn-243frm-supv-cpn-ft_8xb128-200e_h36m.py
# https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth

# body_3d_keypoint/motionbert_dstformer-243frm_8xb32-240e_h36m-original.py
# https://download.openmmlab.com/mmpose/v1/body_3d_keypoint/pose_lift/h36m/motionbert_h36m-f554954f_20230531.pth

def get_mmpose_parser():
    parser = ArgumentParser()
    parser.add_argument(
        '--det_config', 
        type=str,
        default=os.path.join(current_dir,'mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py'),
        help='Config file for detection')
    parser.add_argument(
        '--det_checkpoint',
        type=str, 
        default='https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth',
        help='Checkpoint file for detection')
    parser.add_argument(
        '--pose_estimator_config',
        type=str,
        default=os.path.join(current_dir,'body_2d_keypoint/rtmpose-m_8xb256-420e_body8-256x192.py'),
        help='Config file for the 1st stage 2D pose estimator')
    parser.add_argument(
        '--pose_estimator_checkpoint',
        type=str,
        default='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth',
        help='Checkpoint file for the 1st stage 2D pose estimator')
    parser.add_argument(
        '--pose_lifter_config',
        type=str,
        default=os.path.join(current_dir,'body_3d_keypoint/motionbert_dstformer-243frm_8xb32-240e_h36m-original.py'),
        help='Config file for the 2nd stage pose lifter model')
    parser.add_argument(
        '--pose_lifter_checkpoint',
        type=str,
        default='https://download.openmmlab.com/mmpose/v1/body_3d_keypoint/pose_lift/h36m/motionbert_h36m-f554954f_20230531.pth',
        help='Checkpoint file for the 2nd stage pose lifter model')
    parser.add_argument('--input', type=str, default='', help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='Whether to show visualizations')
    parser.add_argument(
        '--disable-rebase-keypoint',
        action='store_true',
        default=False,
        help='Whether to disable rebasing the predicted 3D pose so its '
        'lowest keypoint has a height of 0 (landing on the ground). Rebase '
        'is useful for visualization when the model do not predict the '
        'global position of the 3D pose.')
    parser.add_argument(
        '--disable-norm-pose-2d',
        action='store_true',
        default=False,
        help='Whether to scale the bbox (along with the 2D pose) to the '
        'average bbox scale of the dataset, and move the bbox (along with the '
        '2D pose) to the average bbox center of the dataset. This is useful '
        'when bbox is small, especially in multi-person scenarios.')
    parser.add_argument(
        '--num-instances',
        type=int,
        default=1,
        help='The number of 3D poses to be visualized in every frame. If '
        'less than 0, it will be set to the number of pose results in the '
        'first frame.')
    parser.add_argument(
        '--output-root',
        type=str,
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=False,
        help='Whether to save predicted results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument('--kpt-thr', type=float, default=0.3)
    parser.add_argument(
        '--use-oks-tracking', action='store_true', help='Using OKS tracking')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--online',
        action='store_true',
        default=False,
        help='Inference mode. If set to True, can not use future frame'
        'information when using multi frames for inference in the 2D pose'
        'detection stage. Default: False.')

    # args = parser.parse_args()
    # return args

    return parser

def parse_args():
    parser = get_mmpose_parser()
    args = parser.parse_args()
    return args