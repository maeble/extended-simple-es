import gym
import pybullet_envs

gym.logger.set_level(40)


class GymWrapper:
    def __init__(self, name, max_step=None):
        self.env = gym.make(name)
        self.max_step = max_step
        self.curr_step = 0
        self.name = name

    def reset(self):
        self.curr_step = 0
        return_list = {}
        transition = {}
        s = self.env.reset()
        transition["state"] = s
        return_list["0"] = transition
        return return_list

    def step(self, action):
        self.curr_step += 1
        return_list = {}
        transition = {}
        s, r, d, info = self.env.step(action["0"])
        if self.max_step != "None":
            if self.curr_step >= self.max_step or d:
                d = True
        transition["state"] = s
        transition["reward"] = r
        transition["done"] = d
        transition["info"] = info
        return_list["0"] = transition
        return return_list, r, d, info

    def get_agent_ids(self):
        return ["0"]

    def render(self):
        return self.env.render(mode="rgb_array")

    def close(self):
        self.env.close()
