# Data Parsing

## Downloading Datasets
To download SCAND please follow [this](https://dataverse.tdl.org/dataset.xhtml?persistentId=doi:10.18738/T8/0PRYRH) link and download the bag files. And download MuSoHu bag files from [here](https://dataverse.orc.gmu.edu/dataset.xhtml?persistentId=doi:10.13021/orc2020/HZI4LJ).

## Installing Dependencies
The parser requires `Python>=3.9` for type annotations. `pyntcloud` package works best with `pandas==2.0.1`.

Run in the root folder:
```
conda env create -f environment_parser.yaml
conda activate parser
export PYTHONPATH=$(pwd)/pose2nav:$PYTHONPATH
export PYTHONPATH=$(pwd)/pose2nav/skeleton/_monoloco:$PYTHONPATH
export PYTHONPATH=$(pwd)/pose2nav/skeleton/_mmpose:$PYTHONPATH
```

## Parsing the data

It is recommended to place all the bag files inside the [data](../social_nav/data) directory as depicted in the project structure below. Otherwise, you need to change `bags_dir` variable in the [parser config file](../social_nav/conf/parser.yaml). You can also change other parameters inside the config files.

Project structure:
```
pose2nav/
├── DATA.md
├── README.md
├── requirements.txt
├── env.yaml
├── data/
│   ├── musohu/
│   │   └── 03112023_mn_dc_night_1_casual.bag
│   ├── scand/
│   │   └── A_Jackal_AHG_Library_Thu_Nov_4_16.bag
│   └── processed/
│       ├── A_Jackal_AHG_Library_Thu_Nov_4_16_0/
│       │   ├── point_cloud/
│       │   ├── rgb/
│       │   ├── keypoints/
│       │   │   └── keypoints.pkl
│       │   └── traj_data.pkl
│       ├── mn_dc_night_1_casual_0/
│       │   ├── depth/
│       │   ├── point_cloud/
│       │   ├── rgb/
│       │   ├── keypoints/
│       │   │   └── keypoints.pkl
│       │   └── traj_data.pkl
└── pose2nav/
    ├── __init__.py
    ├── config/
    │   └── parser.yaml
    ├── data_parser/
    │   ├── __init__.py
    │   ├── parser.py
    │   ├── musohu_parser.py
    │   └── scand_parser.py
    └── utils/
        ├── __init__.py
        ├── helpers.py
        └── parser_utils.py

```

To run the MuSoHu parser, from the root directory of the project run:
```bash
python pose2nav/data_parser/parser.py --name musohu --conf pose2nav/config/parser
```

And to run SCAND parser, change the parser argument to `scand` like:
```bash
python pose2nav/data_parser/parser.py --name scand --conf pose2nav/config/parser
```
We only store the front facing camera for the Spot in SCAND, so both MuSoHu and SCAND have the *same* interface. The only difference is that SCAND does not contain depth data.


### Keypoints Data

```bash
{
  "labels": {
    "0.jpg": [
      {
        "label_id": "0",
        "keypoints_3d": [[x, y, z], ..., [x, y, z]],
        "root_3d": [x, y, z],
        "keypoints_2d": [[x, y], ..., [x, y]]
      },
      ...
    ],
    ...
  }
}
```

`-kp` for parsing the keypoints, run only after parsing data


```bash
python pose2nav/data_parser/parser.py --conf pose2nav/config/parser -kp
```

## Creating Samples

```bash
 sample = {
	"past_positions":      torch.Size([T_obs, 2]), dtype=torch.float32
	"future_positions":    torch.Size([T_pred, 2]), dtype=torch.float32
	"past_yaw":            torch.Size([T_obs]),    dtype=torch.float32
	"future_yaw":          torch.Size([T_pred]),   dtype=torch.float32
	
	"past_vw":             torch.Size([T_obs, 2]), dtype=torch.float32   # (v, w)
	"future_vw":           torch.Size([T_pred, 2]),dtype=torch.float32
	
	"past_frames":         torch.Size([T_obs, 3, H, W]), dtype=torch.float32, range [0,1]
	                       # list of transformed tensors
	
	"original_frame":      torch.Size([H, W, 3]), dtype=torch.float32, range [0,1]
	                       # numpy array, last past frame in original format
	
	"future_frame":        torch.Size([3, H, W]), dtype=torch.float32, range [0,1]
	                       # transformed tensor of target frame
	
	"past_kp_3d":          torch.Size([T_obs, N_h, 17, 3]), dtype=torch.float32
	                       # (x, y, z) per keypoint
	
	"future_kp_3d":        torch.Size([T_pred, N_h, 17, 3]), dtype=torch.float32
	
	"past_kp_2d":          torch.Size([T_obs, N_h, 17, 2]), dtype=torch.float32
	                       # (x, y) pixel coordinates
	
	"future_kp_2d":        torch.Size([T_pred, N_h, 17, 2]), dtype=torch.float32
	
	"past_root_3d":        torch.Size([T_obs, N_h, 3]), dtype=torch.float32
	                       # root joint position in 3D
	
	"future_root_3d":      torch.Size([T_pred, 3]), dtype=torch.float32
	
	"goal_direction":      torch.Size([2]), dtype=torch.float32
	                       # normalized polar coordinates
	
	"dt":                  torch.Size([1]), dtype=torch.float32
}
```

```bash
python pose2nav/data_parser/parser.py --conf pose2nav/config/parser -cs
```

Run only after parsing dadta (including keypoints)