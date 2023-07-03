import numpy as np
from pettingzoo.mpe import simple_spread_v3, simple_adversary_v3, simple_speaker_listener_v4
from pettingzoo.sisl import multiwalker_v6, waterworld_v3
from enum import Enum

PETTINGZOO_ENVS = [
    "simple_spread",
    "waterworld",
    "multiwalker",
    "simple_adversary",
    "simple_speaker_listener",
]


class PettingzooWrapper:
    def __init__(self, name, max_step=None):
        if name == "simple_spread":
            self.env = simple_spread_v3.env(N=2)
        elif name == "waterworld":
            self.env = waterworld_v3.env()
        elif name == "multiwalker":
            self.env = multiwalker_v6.env()
        elif name == "simple_adversary":
            # EPyMARL: https://github.com/semitable/multiagent-particle-envs/blob/master/mpe/scenarios/simple_adversary.py
            # pettingzoo: https://pettingzoo.farama.org/environments/mpe/simple_adversary/
            args = {
                "N": 3
            }
            self.env = simple_adversary_v3.env(**args)
        elif name == "simple_speaker_listener":
            args = {
                "max_cycles":25, 
                "continuous_actions": False
            }
            self.env = simple_speaker_listener_v4.env(**args)
        else:
            assert AssertionError, "wrong env name."
        self.max_step = max_step
        self.curr_step = 0
        self.name = name
        self.agents = self.env.possible_agents
        self.states = self.env.reset()

    def num_states(self, agent_id):
        return np.array(self.states[agent_id]["states"]).shape[1]

    def reset(self):
        self.curr_step = 0
        self.env.reset()
        return_list = {}
        for agent in self.env.agents:
            s = self.env.observe(agent)
            transition = {}
            transition["state"] = s
            return_list[agent] = transition
        return return_list

    def step(self, action):
        self.curr_step += 1
        return_list = {}
        for agent in self.env.agent_iter():
            act = action[agent]
            if self.name == "waterworld":
                act *= 0.001
            self.env.step(act)
            if agent == self.agents[-1]:
                break
        total_d = 0
        total_r = 0
        for agent in self.env.agents:
            transition = {}
            transition["state"] = self.env.observe(agent)
            transition["reward"] = self.env.rewards[agent]
            transition["done"] = agent in self.env.agents
            transition["info"] = self.env.infos[agent]
            total_d += int(not transition["done"])
            total_r += transition["reward"]
            return_list[agent] = transition
        done = total_d == 0
        if self.max_step != "None":
            if self.curr_step >= self.max_step or done:
                done = True
        return return_list, total_r, done, {}

    def get_agent_ids(self):
        return self.env.agents

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)
