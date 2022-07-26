import copy

import gym
import numpy as np
import ray


@ray.remote
def rollout(env, dist, args):
    if dist == "Bernoulli":
        actions = np.random.binomial(**args)
    else:
        raise ValueError("Unknown distribution")
    sampled_reward = 0
    for a in actions:
        obs, reward, done, info = env.step(a)
        sampled_reward += reward
        if done:
            break
    return actions, sampled_reward


class CEM:
    def __init__(
        self,
        env_name,
        optimizer,
        look_ahead,
        num_parallel,
        elite_frac,
        num_ep,
        opt_iters,
        dist,
        control,
    ):
        self.env = gym.make(env_name)
        self.optimizer = optimizer
        self.look_ahead = look_ahead
        self.num_parallel = num_parallel
        self.elite_frac = elite_frac
        self.num_ep = num_ep
        self.opt_iters = opt_iters
        self.dist = dist
        self.control = control
        self.reset()

    def reset(self):
        self.episode_rewards = []
        self.num_episodes = 0

    def cross_ent_optimizer(self):
        n_elites = int(np.ceil(self.num_parallel * self.elite_frac))
        if self.dist == "Bernoulli":
            p = [0.5] * self.look_ahead
            for i in range(self.opt_iters):
                features = []
                for j in range(self.num_parallel):
                    args = {"n": 1, "p": p, "size": self.look_ahead}
                    fid = rollout.remote(copy.deepcopy(self.env), self.dist, args)
                    features.append(fid)
                results = [tuple(ray.get(id)) for id in features]
                sampled_rewards = [r for _, r in results]
                eilte_ix = np.argsort(sampled_rewards)[-n_elites:]
                elite_actions = np.array([a for a, _ in results])[eilte_ix]
                p = np.mean(elite_actions, axis=0)
            actions = np.random.binomial(n=1, p=p, size=self.look_ahead)
        else:
            raise ValueError("Unknown distribution")
        return actions

    def optimize(self):
        for i in range(self.num_ep):
            self.env.reset()
            ep_reward = 0
            done = False
            j = 0
            while not done:
                print(j)
                j += 1
                if self.optimizer == "cem":
                    actions = self.cross_ent_optimizer()
                else:
                    raise ValueError("Unknown Optimizer")
                if self.control == "open-loop":
                    for a in actions:
                        obs, reward, done, info = self.env.step(a)
                        ep_reward += reward
                        if done:
                            break
                elif self.control == "closed-loop":
                    obs, reward, done, info = self.env.step(actions[0])
                    ep_reward += reward
                else:
                    raise ValueError("Unknown control type.")
                if done:
                    self.episode_rewards.append(ep_reward)
                    self.num_episodes += 1
                    print(f"Episode {i}, reward: {ep_reward}")

    @property
    def avg_reward(self):
        return np.mean(self.episode_rewards)


if __name__ == "__main__":
    np.random.seed(0)
    ray.init()
    cem = CEM(
        env_name="CartPole-v0",
        optimizer="cem",
        look_ahead=10,
        num_parallel=50,
        elite_frac=0.1,
        num_ep=10,
        opt_iters=5,
        dist="Bernoulli",
        control="closed-loop",
    )
    cem.optimize()
    print(f"Average episode reward in {cem.num_episodes} is {cem.avg_reward}")
    