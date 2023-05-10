import numpy as np

def evaluate(agent, env, num_episodes=100, q_table=None):
    final_rewards = []

    for episode in range(num_episodes):
        rewards = []
        mid_price = []
        bid_price = []
        ask_price = []
        volume = []
        values = []
        disc_reward = 0

        env.reset()
        state = env.state()

        while env.t < env.T:
            if agent.__class__.__name__ == 'QAgent':
                np.array(np.unravel_index(q_table[state].argmax(), q_table[state].shape))
            else:
                action = agent.get_action()
            state, action_reward, done = env.step(np.array(action))
            disc_reward += action_reward
            rewards.append(disc_reward)
            mid_price.append(env.mid)
            bid_price.append(env.mm_bid)
            ask_price.append(env.mm_ask)
            volume.append(state[0])
            values.append(env.V_t)

        if agent.__class__.__name__ == 'QAgent':
            opt_action = np.unravel_index(agent.q_table[(0,0)].argmax(), agent.q_table[(0,0)].shape)
            Q_star = agent.q_table[(0,0)][opt_action]

        logs = {'rewards':rewards, 
                'mid_price':mid_price, 
                'bid_price':bid_price, 
                'ask_price':ask_price, 
                'volume':volume, 
                'values':values}
        
        final_rewards.append(disc_reward)

    mean_reward = np.mean(final_rewards)
    min_reward = np.min(final_rewards)
    max_reward = np.max(final_rewards)
    std_reward = np.std(final_rewards)

    reward_stats = {'mean_reward': mean_reward,
                    'min_reward': min_reward,
                    'max_reward': max_reward,
                    'std_reward': std_reward}

    return logs, reward_stats