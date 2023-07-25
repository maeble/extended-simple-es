import logging
import random
import yaml
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import builder
import os
from copy import deepcopy
from learning_strategies.evolution.loop import ESLoop
from moviepy.editor import ImageSequenceClip
import json 


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, default="conf/ant.yaml")
    parser.add_argument("--ckpt-path", type=str)
    parser.add_argument("--save-gif", action="store_true")
    args = parser.parse_args()

    with open(args.cfg_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

    env = builder.build_env(config["env"])
    agent_ids = env.get_agent_ids() 
    num_of_epochs = int(os.path.basename(args.ckpt_path).split("_")[1].split(".")[0])
    run_num = args.ckpt_path.split("/")[-3]

    print("number of epochs trained:", num_of_epochs)

    # create save dirs
    if args.save_gif:
        save_clips_dir = f"test_gif/{run_num}/"
        os.makedirs(save_clips_dir, exist_ok=True)
        
    network = builder.build_network(config["network"])
    network.load_state_dict(torch.load(args.ckpt_path))
    for i in range(100):
        models = {}
        for agent_id in agent_ids:
            models[agent_id] = deepcopy(network)
            models[agent_id].eval()
            models[agent_id].reset()
        obs = env.reset()

        done = False
        episode_reward = 0
        ep_step = 0
        ep_render_lst = []
        coll_state_vector = config["network"]["has_collection_state_vector"]
        num_states = config["network"]["num_state"]
        while not done:
            actions = {}
            if coll_state_vector:
                for k, model in models.items():
                    s = np.array(obs[k]["state"])[np.newaxis, ...]
                    team_actions = []
                    for ag_s in s[0]:
                        ag_s =np.array([ag_s])
                        model_ag_s = int(__RolloutWorkerModelIterate(model, ag_s, num_states))
                        team_actions.append(model_ag_s)
                        actions[k] = tuple(team_actions)
            else: 
                for k, model in models.items():
                    s = np.array(obs[k]["state"])[np.newaxis, ...]
                    model_s = __RolloutWorkerModelIterate(model, s, num_states)
                    actions[k] = model_s
            obs, r, done, _ = env.step(actions)

            rgb_array = env.render(mode="rgb_array")
            if args.save_gif:
                ep_render_lst.append(rgb_array)

            if coll_state_vector:
                episode_reward += sum(r)
            else:
                episode_reward += r             
            ep_step += 1
        print("reward: ", episode_reward, "ep_step: ", ep_step)
        if args.save_gif:
            clip = ImageSequenceClip(ep_render_lst, fps=30)
            save_clips_file=save_clips_dir + f"ep_{i}.gif"
            clip.write_gif(save_clips_file, fps=30) # overwrites existing files
        del ep_render_lst


def __RolloutWorkerModelIterate(model, s, num_states):
    if s.size < num_states: # add zero padding
        additional_padding_shape = (num_states - s.size, 0)
        s = F.pad(torch.Tensor(s), additional_padding_shape, "constant", 0).numpy()
    return model(s)


if __name__ == "__main__":
    main()
