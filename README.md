# maadm-tfm-generative-ts
TFM - Álvaro Rodríguez Alonso

## Usage


```bash
python -W ignore scripts/train_classification.py --model-config InceptionTime.json --dataset-config melbourne_pedestrian.json --out-dir inception_time0
```

```bash
python -W ignore scripts/train_generation.py --model-config generator_inceptionTime.json --dataset-config melbourne_pedestrian.json --out-dir generator0
```

```bash
python scripts/train_gan.py --generator-config generator_inceptionTime.json --discriminator-config discriminator_InceptionTime.json --dataset-config melbourne_pedestrian.json --out-dir gan0
```