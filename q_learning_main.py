import simple_grid
from q_learning_skeleton import *
import gym


def act_loop(env, agent, num_episodes):
    policy = []
    for episode in range(num_episodes):
        episode_policy = []
        state = env.reset()
        agent.reset_episode()

        print('---episode %d---' % episode)
        renderit = False
        if episode % 10 == 0:
            renderit = True

        for t in range(MAX_EPISODE_LENGTH):
            if renderit:
                env.render()
                print("state:", state)
            printing=False
            if t % 500 == 499:
                printing = True

            if printing:
                print('---stage %d---' % t)
                agent.report()

            action = agent.select_action(state)
            episode_policy.append(action)
            new_state, reward, done, info = env.step(action)
            if printing:
                print("act:", action)
                print("new state", state)
                print("reward=%s" % reward)

            # Update the Q function for the new state
            agent.process_experience(state, action, new_state, reward, done)
            # Update the state
            state = new_state
            if done:
                policy.append(episode_policy)
                print("Episode finished after {} timesteps".format(t+1))
                env.render()
                agent.report()
                break
            if t == MAX_EPISODE_LENGTH - 1:
                policy.append(episode_policy)

    env.close()
    # optionally save the policies created
    #np.save("policy_alley_decay_deprpen_5000", np.array(policy))
    #np.save("q_values_alley_decay_deprpen_5000", agent.Q)

if __name__ == "__main__":
    # TODO - comment in/out based on which environment you want to run
    #env = simple_grid.DrunkenWalkEnv(map_name="walkInThePark")
    env = simple_grid.DrunkenWalkEnv(map_name="theAlley")
    num_a = env.action_space.n

    if (type(env.observation_space)  == gym.spaces.discrete.Discrete):
        num_o = env.observation_space.n
    else:
        raise("Qtable only works for discrete observations")

    discount = DEFAULT_DISCOUNT
    ql = QLearner(num_o, num_a, discount) #<- QTable
    act_loop(env, ql, NUM_EPISODES)


