# torch-es
### Project for bio-inspired methods using [PyTorch](https://pytorch.org/)
<p float="center">
  <img src="https://user-images.githubusercontent.com/16518993/123286330-ca1a1280-d548-11eb-8789-1b27edaee9a8.gif" width="300" />
  <img src="https://user-images.githubusercontent.com/16518993/123286575-fcc40b00-d548-11eb-9e73-1ec3b465d5ce.gif" width="300" /> 
</p>

## Algorithms
### learning strategies
- [x] vanilla evolution srtategy
- [x] vanilla genetic srtategy
- [x] [OpenAI ES](https://openai.com/blog/evolution-strategies/)
- [ ] CMA-ES
- [ ] MAPPO(Multi Agent RL)
- [ ] [WANN](https://arxiv.org/abs/1906.04358)
- [ ] [hebbian plasticity](https://arxiv.org/abs/2007.02686)

### networks
- [x] ANN(+ GRU)
- [ ] Indirect Encoding
- [ ] SNN

## Recurrent ANN with POMDP CartPole
Recurrent ANN(GRU) is also implemented by default. The use of the gru module can be set in the config file. For environment, LunarLander and CartPole support POMDP setting.
```python
network:
  gru: True
env:
  name: "CartPole-v1"
  pomdp: True
```
### POMDP CartPole benchmarks
GRU agent with simple-evolution strategy(green) got perfect score (500) in POMDP CartPole environment, whereas ANN agent(yellow) scores nearly 60, failed to learn POMDP CartPole environment. GRU agent with simple-genetic strategy(purple) also shows poor performance.
<img src=https://user-images.githubusercontent.com/16518993/125189883-4d3fa600-e275-11eb-9311-1a3cce3d5041.png width=600>


## Installation

```bash
# recommend python==3.8.10
git clone https://github.com/jinPrelude/torch-es.git
cd torch-es
pip install -r requirements.txt
```

## Train

```bash
# training LunarLander-v2
python run_es.py --cfg-path conf/lunarlander.yaml 

# training BiPedalWalker-v3
python run_es.py --cfg-path conf/bipedal.yaml --log
```

You need [wandb](https://wandb.ai/) account for logging. Wandb provides various useful logging features for free.

## Test saved model

```bash
# training LunarLander-v2
python test.py --cfg-path conf/lunarlander.yaml --ckpt-path <saved-model-dir> --save-gif
```


