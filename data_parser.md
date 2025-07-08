

conda env create -f env.yaml
conda activate pose2nav
pip install -r requirements.txt

python -m pose2nav.data_parser.parser --name scand --conf pose2nav/config/parser
