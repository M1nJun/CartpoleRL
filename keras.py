import gymnasium as gym
import numpy as np
from collections import namedtuple
from tensorflow import keras
from keras import layers, optimizers

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

class Net(keras.Model):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = keras.Sequential([
            layers.Dense(hidden_size, activation='relu', input_shape=(obs_size,)),
            layers.Dense(n_actions)
        ])

    def call(self, x):
        return self.net(x)

def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    while True:
        obs = np.expand_dims(obs, axis=0)
        act_probs = net(obs, training=False)[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step = EpisodeStep(observation=obs[0], action=action)
        episode_steps.append(step)
        if terminated or truncated:
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []
            obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs

def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for reward, steps in batch:
        if reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, steps))
        train_act.extend(map(lambda step: step.action, steps))

    train_obs = np.array(train_obs)
    train_act = np.array(train_act)
    return train_obs, train_act, reward_bound, reward_mean

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = optimizers.Adam(learning_rate=0.01)
    
    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs, acts, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        with tf.GradientTape() as tape:
            logits = net(obs, training=True)
            loss_value = objective(acts, logits)
        grads = tape.gradient(loss_value, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
            iter_no, loss_value, reward_m, reward_b))
        if reward_m > 199:
            print("Solved!")
            break
