import matplotlib.pyplot as plt 
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

def plot_single_experiemnt(agent_name, mid_price, bid_price, ask_price, rewards):

    # Plot spread
    plt.figure(figsize=(10,4), dpi=200)
    plt.plot(mid_price, label='mid price')
    plt.fill_between(range(len(mid_price)), bid_price, ask_price, alpha=0.5)
    plt.plot(bid_price, label='bid price')
    plt.plot(ask_price, label='ask price')
    plt.legend()
    plt.title(f'Spread Over Time ({agent_name})')
    plt.ylabel('Price')
    plt.xlabel('Time')
    plt.show()
    
    # Plot rewards
    plt.figure(figsize=(10,4), dpi=200)
    plt.plot(rewards)
    plt.title(f'Cumulative Reward Over Time ({agent_name})')
    plt.ylabel('Cumulative Reward')
    plt.xlabel('Time')
    plt.show()

def single_experiment(agent_name, agent, env, show=True):

    # Record
    rewards = []
    mid_price = []
    bid_price = []
    ask_price = []
    volume = []
    value = []

    # Initialize
    disc_reward = 0
    state = env.state()

    # Experiment
    while env.t < env.T:

        # Random agent
        if agent.__class__.__name__ == 'RandomAgent':
            action = agent.get_action()
            state, reward, done = env.step(np.array(action))

        # Q-learning agent
        elif agent.__class__.__name__ == 'QAgent':
            action = agent.get_action(state=state)
            next_state, reward, done = env.step(np.array(action))
            agent.update_Q(action, reward, state, next_state)
            agent.decay_epsilon()
            state = next_state

        # Multi-armed bandit agents
        elif agent.__class__.__name__ in ['ExploreFirstAgent','UCBAgent','EpsilonGreedyAgent','DecayEpsilonGreedyAgent']:
            action = agent.get_action()
            state, reward, done = env.step(np.array(action))
            agent.update_Q(action, reward)
            if agent.__class__.__name__ == 'DecayEpsilonGreedyAgent':
                agent.decay_epsilon()

        else:
            print('Undefined agent')
            return

    
        # Record
        disc_reward += reward  
        rewards.append(disc_reward)
        mid_price.append(env.mid)
        bid_price.append(env.mm_bid)
        ask_price.append(env.mm_ask)
        volume.append(state[0])
        value.append(env.V_t)

    # Show single experiment results
    if show:
        env.render()
        plot_single_experiemnt(agent_name, mid_price, bid_price, ask_price, rewards)

    log = {'rewards':rewards, 
            'mid_price':mid_price, 
            'bid_price':bid_price, 
            'ask_price':ask_price, 
            'volume':volume, 
            'value':value}
    
    return log

def multiple_experiment(agent_name, agent, env, num_episodes=100, q_table=None, 
                        show_each=False, show_last=True, show_average=True):
    # Record results
    final_rewards = []
    logs = []

    # Train and run multiple episodes
    for episode in range(num_episodes):
        agent.reset()
        env.reset()
        log = single_experiment(agent_name+f', Episode {episode}', agent, env, show=show_each)
        logs.append(log)
        final_rewards.append(log['rewards'][-1])

    # Show last episode
    if show_last:
        log = logs[-1]
        plot_single_experiemnt(agent_name+', Last Episode', log['mid_price'], log['bid_price'], log['ask_price'], log['rewards'])

    # Average rewards of each step across episode
    average_rewards = np.array([log['rewards'] for log in logs]).mean(axis=0)
    if show_average:
        plt.figure(figsize=(10,4), dpi=200)
        plt.plot(average_rewards)
        plt.title(f'Average Reward of {num_episodes} Episodes ({agent_name})')
        plt.ylabel('Cumulative Reward')
        plt.xlabel('Time')
        plt.show()

    return logs, final_rewards, average_rewards

