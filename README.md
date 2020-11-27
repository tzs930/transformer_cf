0. Setup environement
```
conda create -f environment.yml
```

1. Download MovieLens-20M dataset

```
wget http://files.grouplens.org/datasets/movielens/ml-20m.zip
unzip ml-20m.zip
```

2. Preprocess dataset
```
python preprocess_ml20m.py
```

3. Train