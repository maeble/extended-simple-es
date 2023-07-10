import os
import time
from datetime import datetime
from collections import deque

import numpy as np
import multiprocessing as mp
import torch
import torch.nn.functional as F

import wandb
from .abstracts import BaseESLoop


class ESLoop(BaseESLoop):
    def __init__(
        self,
        config,
        offspring_strategy,
        env,
        network,
        generation_num,
        process_num,
        eval_ep_num,
        log=False,
        save_model_period=10,
    ):
        super().__init__()
        self.env = env
        self.network = network
        self.process_num = process_num
        self.network.zero_init()
        self.offspring_strategy = offspring_strategy
        self.generation_num = generation_num
        self.eval_ep_num = eval_ep_num
        self.ep5_rewards = deque(maxlen=5)
        self.num_states = config["network"]["num_state"]
        if "has_collection_state_vector" in config["network"].keys():
            self.coll_state_vector = config["network"]["has_collection_state_vector"]
        else:
            self.coll_state_vector = False
        self.log = log
        self.save_model_period = save_model_period

        # create log directory
        now = datetime.now()
        curr_time = now.strftime("%Y%m%d%H%M%S")
        dir_lst = []
        self.save_dir = f"logs/{self.env.name}/{curr_time}"
        dir_lst.append(self.save_dir)
        dir_lst.append(self.save_dir + "/saved_models/")
        for _dir in dir_lst:
            os.makedirs(_dir)

        if self.log:
            wandb.init(project=self.env.name, config=config)

    def run(self):

        # init offsprings
        offsprings = self.offspring_strategy.init_offspring(
            self.network, self.env.get_agent_ids()
        )
        print("Agents:", self.env.get_agent_ids(), "\n\n")

        prev_reward = float("-inf")
        ep_num = 0
        for _ in range(self.generation_num):
            start_time = time.time()
            ep_num += 1

            # create an actor by the number of cores
            p = mp.Pool(self.process_num)
            arguments = [(self.env, off, self.eval_ep_num, self.num_states, self.coll_state_vector) for off in offsprings]

            # start ollout actors
            rollout_start_time = time.time()

            # rollout(https://stackoverflow.com/questions/41273960/python-3-does-pool-keep-the-original-order-of-data-passed-to-map)
            if self.process_num > 1:
                results = p.map(RolloutWorker, arguments)
            else:
                results = [RolloutWorker(arg) for arg in arguments]
            # concat output lists to single list
            p.close()
            rollout_consumed_time = time.time() - rollout_start_time

            eval_start_time = time.time()
            offsprings, best_reward, curr_sigma = self.offspring_strategy.evaluate(
                results
            )
            eval_consumed_time = time.time() - eval_start_time

            # print log
            consumed_time = time.time() - start_time
            print(
                f"episode: {ep_num}, Best reward: {best_reward:.2f}, sigma: {curr_sigma:.3f}, time: {consumed_time:.2f}, rollout_t: {rollout_consumed_time:.2f}, eval_t: {eval_consumed_time:.2f}"
            )

            prev_reward = best_reward
            if self.log:
                self.ep5_rewards.append(best_reward)
                ep5_mean_reward = sum(self.ep5_rewards) / len(self.ep5_rewards)
                wandb.log(
                    {"ep5_mean_reward": ep5_mean_reward, "curr_sigma": curr_sigma}
                )

            elite = self.offspring_strategy.get_elite_model()
            if ep_num % self.save_model_period == 0:
                save_pth = self.save_dir + "/saved_models" + f"/ep_{ep_num}.pt"
                torch.save(elite.state_dict(), save_pth)


# offspring_id, worker_id, eval_ep_num=10
def RolloutWorker(arguments, debug=False):
    env, offspring, eval_ep_num, num_states, coll_state_vector = arguments
    total_reward = 0
    for _ in range(eval_ep_num):
        states = env.reset()
        done = False
        for k, model in offspring.items():
            model.reset()
        while not done:
            actions = {}
            s = np.array(states[k]["state"])[np.newaxis, ...]
            if coll_state_vector:
                if debug:
                    print("split collection state vector...")
                for k, model in offspring.items():
                    team_actions = []
                    for ag_s in s[0]:
                        ag_s =np.array([ag_s])
                        model_ag_s = int(__RolloutWorkerModelIterate(model, ag_s, num_states, debug))
                        team_actions.append(model_ag_s)
                        actions[k] = team_actions
            else: 
                for k, model in offspring.items():
                    model_s = __RolloutWorkerModelIterate(model, s, num_states, debug)
                    actions[k] = model_s
            states, r, done, _ = env.step(actions)
            #env.render()
            if coll_state_vector:
                #print(r)
                # FIXME reward is always 0 for envs with collection_state_vector=True
                total_reward += sum(r)
            else:
                total_reward += r
    rewards = total_reward / eval_ep_num
    return rewards

def __RolloutWorkerModelIterate(model, s, num_states, debug):
    if debug:
        print("1", s.shape, s)
    if s.size < num_states: # add zero padding
        if debug:
            print("add zero padding...")
        additional_padding_shape = (num_states - s.size, 0)
        s = F.pad(torch.Tensor(s), additional_padding_shape, "constant", 0).numpy()
        if debug:
            print("2", s.shape, s)
    return model(s)
