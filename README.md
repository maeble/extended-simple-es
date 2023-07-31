# Multi-Agent Experiments with simple-es

## Note: This is a fork of [simple-es](https://github.com/jinPrelude/simple-es)

For installation and usage notes, please visit [simple-es](https://github.com/jinPrelude/simple-es).

### Changes to simple-es

- add version info to `requirements.txt`
- adds multi-agent support for `gym` environments
- uses a different neural network model
- adds logs for mean return and mean steps per episode that are saved to file
- adds the following environments:
  - [Level Based Foraging (lbforaging)](https://github.com/semitable/lb-foraging)
  - [Robotic Warehouse (rware)](https://github.com/semitable/robotic-warehouse)
  - Multi-Particle Environments:
    - [speaker-listener](https://pettingzoo.farama.org/environments/mpe/simple_speaker_listener/)
    - [adversary](https://pettingzoo.farama.org/environments/mpe/simple_adversary/#simple-adversary)
- bug fixes:
  - fix `NoneType` error when passing the argument `--save-gif` to `test.py`
  - the pettingzoo environments now use the `max_step` configuration defined in the environment config yaml-files. (This did not work properly before, it always run just 1 step per episode)
