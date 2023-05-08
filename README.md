# Multi-Agent Experiments with simple-es

## Note: This is a fork of [simple-es](https://github.com/jinPrelude/simple-es)

### Usage 

For installation and basic usage notes, please visit [simple-es](https://github.com/jinPrelude/simple-es).

If you want to check if everything works fine and all environment configurations can be run, execute:

```bash
./scripts/test.sh
```

### Changes to simple-es

- adds dockerization
- add package version info to `requirements.txt`
- adds multi-agent support for `gym` environments
- uses a new neural network model with a hidden dimension of 64 and the *reLU* activation function
- adds logs for mean return and mean steps per episode that are saved to file
- adds the following multi-agent environments:
  - [Level Based Foraging (lbforaging)](https://github.com/semitable/lb-foraging)
  - [Robotic Warehouse (rware)](https://github.com/semitable/robotic-warehouse)
  - Multi-Particle Environments (MPE):
    - [speaker-listener](https://pettingzoo.farama.org/environments/mpe/simple_speaker_listener/)
    - [adversary](https://pettingzoo.farama.org/environments/mpe/simple_adversary/#simple-adversary)
- bug fixes:
  - fix `NoneType` error when passing the argument `--save-gif` to `test.py`
  - the pettingzoo environments now use the `max_step` configuration defined in the environment config yaml-files. (This did not work properly before, it always run just 1 step per episode)
