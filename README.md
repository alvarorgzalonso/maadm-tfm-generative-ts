# maadm-tfm-generative-ts
TFM - Álvaro Rodríguez Alonso

## Usage

```bash
python -W ignore scripts/train_classification.py --model-config classificator_custom0.json --dataset-config melbourne_pedestrian.json --ckpt-name classificator_custom0.ckpt
```


```bash
python -W ignore scripts/train_classification.py --model-config InceptionTime.json --dataset-config melbourne_pedestrian.json --ckpt-name inception_time0.ckpt
```



```bash
python scripts/train_gan.py --model-config GAN.json --dataset-config sine.json --ckpt-name gan0.ckpt
```