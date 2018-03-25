# ZeroShotLearning
Implementations of state-of-the-art zero-shot learning algorithms.

## Semantic Autoencoder for Zero-shot Learning (CVPR2017)
Python implementation.
```
Requirements:
numpy, scipy, sklearn
```
### Demo
Download awa data from [here](https://drive.google.com/open?id=0B3-qq6zHiDF3THJqeFhQX0hROVk) and put it into 'data_zsl' folder.
```
python semanticAE.py

Result:
training
evaluate
AWA dataset V >> S Acc : 0.8488673139158576

```
The result is comparable with the [author's matlab implementation](https://github.com/Elyorcv/SAE).


