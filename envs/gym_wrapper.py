import gym
import pybullet_envs
import lbforaging
import rware

gym.logger.set_level(40)


class GymWrapper:
    def __init__(self, name, max_step=None, pomdp=False, kwargs=None):
        if name.startswith("lbf"):
            gym.envs.register(
                id=name,
                entry_point="lbforaging.foraging:ForagingEnv",
                kwargs={
                    "players": kwargs["p"],
                    "max_player_level": 3,
                    "field_size": (kwargs["x"], kwargs["x"]),
                    "max_food": kwargs["f"],
                    "sight": kwargs["s"],
                    "max_episode_steps": max_step,
                    "force_coop": kwargs["c"],
                }
            )
            self.env = gym.make(name)
        else:
            self.env = gym.make(name)
            if pomdp:
                if "LunarLander" in name:
                    print("POMDP LunarLander")
                    self.env = LunarLanderPOMDP(self.env)
                elif "CartPole" in name:
                    print("POMDP CartPole")
                    self.env = CartPolePOMDP(self.env)
                else:
                    raise AssertionError(f"{name} doesn't support POMDP.")
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
        if type(r)==list:
            r = sum(r)
        if type(d)==list and all(d):
            d = True 
        else:
            d = False
        if self.max_step != "None":
            if self.curr_step >= self.max_step:
                if type(d)==bool:
                    d = True
        transition["state"] = s
        transition["reward"] = r
        transition["done"] = d
        transition["info"] = info
        return_list["0"] = transition
        return return_list, r, d, info

    def get_agent_ids(self):
        return ["0"]

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        self.env.close()


class LunarLanderPOMDP(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        # modify obs
        obs[2] = 0
        obs[3] = 0
        obs[5] = 0
        return obs


class CartPolePOMDP(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        # modify obs
        obs[1] = 0
        obs[3] = 0
        return obs
