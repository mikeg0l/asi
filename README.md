```
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh
cp .env.example .env
conda create -n asi python=3.10 -y
conda activate asi
pip install -r requirements.txt
```
