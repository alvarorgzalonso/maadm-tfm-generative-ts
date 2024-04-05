# maadm-tfm-generative-ts
TFM - Álvaro Rodríguez Alonso

## Usage

```bash
python -W ignore scripts/train_classification.py --model-config classificator0.json --dataset-config melbourne_pedestrian.json --out-dir classificator0
```


```bash
python -W ignore scripts/train_classification.py --model-config InceptionTime.json --dataset-config melbourne_pedestrian.json --out-dir inception_time0
```



```bash
python scripts/train_gan.py --generator-config generator_inceptionTime.json --discriminator-config InceptionTime.json --dataset-config melbourne_pedestrian.json --out-dir gan0
```