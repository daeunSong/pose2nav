# Data Parsing

## Downloading Datasets
To download SCAND please follow [this](https://dataverse.tdl.org/dataset.xhtml?persistentId=doi:10.18738/T8/0PRYRH) link and download the bag files. And download MuSoHu bag files from [here](https://dataverse.orc.gmu.edu/dataset.xhtml?persistentId=doi:10.13021/orc2020/HZI4LJ).

## Installing Dependencies
The parser requires `Python>=3.9` for type annotations. `pyntcloud` package works best with `pandas==2.0.1`.

```
conda env create -f env.yaml
conda activate pose2nav
pip install -r requirements.txt
```

## Parsing the data

It is recommended to place all the bag files inside the [data](../social_nav/data) directory as depicted in the project structure below. Otherwise, you need to change `bags_dir` variable in the [parser config file](../social_nav/conf/parser.yaml). You can also change other parameters inside the config files.

Project structure:
```
pose2nav/
├── data_parser.md
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
│       │   └── traj_data.pkl
│       ├── mn_dc_night_1_casual_0/
│       │   ├── depth/
│       │   ├── point_cloud/
│       │   ├── rgb/
│       │   └── traj_data.pkl
└── pose2nav/
    ├── __init__.py
    ├── config/
    │   └── parser.yaml
    ├── data_parser/
    │   ├── __init__.py
    │   ├── parser.py
    │   ├── parser_utils.py
    │   ├── musohu_parser.py
    │   └── scand_parser.py
    └── utils/
        ├── __init__.py
        ├── helpers.py
        └── parser_utils.py

```

To run the MuSoHu parser, from the root directory of the project run:
```bash
python -m pose2nav.data_parser.parser --name musohu --conf pose2nav/config/parser
```

And to run SCAND parser, change the parser argument to `scand` like:
```bash
python -m pose2nav.data_parser.parser --name scand --conf pose2nav/config/parser
```
We only store the front facing camera for the Spot in SCAND, so both MuSoHu and SCAND have the *same* interface. The only difference is that SCAND does not contain depth data.
