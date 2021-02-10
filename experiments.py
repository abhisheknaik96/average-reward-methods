import numpy as np
from tqdm import tqdm
import copy
from utils.rl_glue import RLGlue
import json
import sys


def run_exp_learning_control(env, agent, config):
    num_runs = config['exp_parameters']['num_runs']
    max_steps = config['exp_parameters']['num_max_steps']
    eval_every_n_steps = config['exp_parameters']['eval_every_n_steps']
    max_steps_eval = config['exp_parameters']['num_max_steps_eval']
    save_weights = config['exp_parameters'].get('save_weights', 0)
    num_weights = config['exp_parameters'].get('num_weights', 1)

    env_info = config['env_parameters']
    agent_info = config['agent_parameters']

    print('Env: %s, Agent: %s' % (env.__name__, agent.__name__))

    log_data = {}
    rewards_all_train = np.zeros((num_runs, max_steps+1))
    # avg_rewards_all = np.zeros((num_runs, max_steps))
    rewards_eval = np.zeros(max_steps_eval)
    rewards_eval_avg = np.zeros((num_runs, int(max_steps/eval_every_n_steps) + 1))
    assert max_steps % eval_every_n_steps == 0  # ideally not necessary, but enforcing nonetheless
    weights_final = np.zeros((num_runs, num_weights))
    weights_all = np.zeros((num_runs, int(max_steps/eval_every_n_steps) + 1, num_weights))
    avg_v_all = np.zeros((num_runs, int(max_steps/eval_every_n_steps) + 1))
    avg_r_all = np.zeros((num_runs, int(max_steps/eval_every_n_steps) + 1))

    for run in tqdm(range(num_runs)):

        agent_info['random_seed'] = run
        env_info['random_seed'] = run

        # training instance
        rl_glue = RLGlue(env, agent)
        rl_glue.rl_init(agent_info, env_info)
        rl_glue.rl_start()
        assert num_weights == rl_glue.agent.weights.size, "exp_params:num_weights should match number of agent weights"

        # evaluation instance
        agent_eval_info = copy.deepcopy(agent_info)
        env_eval_info = copy.deepcopy(env_info)
        agent_eval_info['policy_type'] = 'greedy'
        agent_eval_info['alpha_w'] = 0.0
        agent_eval_info['alpha_r'] = 0.0
        if 'DifferentialQlearningAgent' in agent.__name__:
            agent_eval_info['alpha_w_f'] = 0.0
            agent_eval_info['alpha_r_f'] = 0.0
        rl_glue_eval = RLGlue(env, agent)
        eval_idx = 0

        for timestep in range(max_steps+1):

            if timestep % eval_every_n_steps == 0:
                weights_to_be_evaluated = np.copy(rl_glue.agent.weights)
                env_eval_info['timestep'] = timestep
                rl_glue_eval.rl_init(agent_eval_info, env_eval_info)
                rl_glue_eval.agent.set_weights(weights_to_be_evaluated)
                rl_glue_eval.rl_start()
                rewards_eval *= 0.0

                for t in range(max_steps_eval):
                    r, _, _, _ = rl_glue_eval.rl_step()
                    rewards_eval[t] = r

                rewards_eval_avg[run][eval_idx] = np.mean(rewards_eval)
                if save_weights:
                    weights_all[run][eval_idx] = rl_glue.agent.weights
                    avg_v_all[run][eval_idx] = rl_glue.agent.avg_value
                    avg_r_all[run][eval_idx] = rl_glue.agent.avg_reward

                eval_idx += 1

            reward, obs, action, _ = rl_glue.rl_step()
            rewards_all_train[run][timestep] = reward
            # avg_rewards_all[run][timestep] = rl_glue.agent.avg_reward
        weights_final[run] = rl_glue.agent.weights

    tqdm.write('Train_RewardRate_total\t= %f' % (np.mean(rewards_all_train)))
    tqdm.write('Train_RewardRate_lasthalf\t= %f\n' % np.mean(rewards_all_train[:,rewards_all_train.shape[1]//2:]))
    tqdm.write('Eval_RewardRate_total\t= %f' % np.mean(rewards_eval_avg))
    tqdm.write('Eval_RewardRate_lasthalf\t= %f\n' % np.mean(rewards_eval_avg[:,rewards_eval_avg.shape[1]//2:]))
    tqdm.write('AgentRewardRate_total\t= %f' % np.mean(avg_r_all))
    tqdm.write('AgentRewardRate_lasthalf\t= %f\n' % np.mean(avg_r_all[:,avg_r_all.shape[1]//2:]))
    log_data['rewards_all_train'] = rewards_all_train
    log_data['rewards_all_eval'] = rewards_eval_avg
    log_data['weights_final'] = weights_final
    if save_weights:
        log_data['weights_all'] = weights_all
        log_data['avg_v_all'] = avg_v_all
        log_data['avg_r_all'] = avg_r_all

    return log_data


def run_exp_learning_control_no_eval(env, agent, config):
    num_runs = config['exp_parameters']['num_runs']
    max_steps = config['exp_parameters']['num_max_steps']
    eval_every_n_steps = config['exp_parameters']['eval_every_n_steps']
    save_weights = config['exp_parameters'].get('save_weights', 0)
    num_weights = config['exp_parameters'].get('num_weights', 1)
    save_counts = config['exp_parameters'].get('save_visitation_counts', False)

    env_info = config['env_parameters']
    agent_info = config['agent_parameters']

    print('Env: %s, Agent: %s' % (env.__name__, agent.__name__))

    log_data = {}
    rewards_all_train = np.zeros((num_runs, max_steps+1))
    avg_rewards_all = np.zeros((num_runs, max_steps+1))
    assert max_steps % eval_every_n_steps == 0  # ideally not necessary, but enforcing nonetheless
    weights_final = np.zeros((num_runs, num_weights))
    weights_all = np.zeros((num_runs, int(max_steps/eval_every_n_steps) + 1, num_weights))
    avg_v_all = np.zeros((num_runs, int(max_steps/eval_every_n_steps) + 1))
    avg_r_all = np.zeros((num_runs, int(max_steps/eval_every_n_steps) + 1))
    counts_all = np.zeros((num_runs, 11, 4, 2))    # for collecting AccessControl visitations

    for run in tqdm(range(num_runs)):

        agent_info['random_seed'] = run
        env_info['random_seed'] = run

        # training instance
        rl_glue = RLGlue(env, agent)
        rl_glue.rl_init(agent_info, env_info)
        rl_glue.rl_start()
        assert num_weights == rl_glue.agent.weights.size, "exp_params:num_weights should match number of agent weights"

        eval_idx = 0

        for timestep in range(max_steps+1):

            if timestep % eval_every_n_steps == 0:
                if save_weights:
                    weights_all[run][eval_idx] = rl_glue.agent.weights
                    avg_v_all[run][eval_idx] = rl_glue.agent.avg_value
                    avg_r_all[run][eval_idx] = rl_glue.agent.avg_reward
                eval_idx += 1

            reward, obs, action, _ = rl_glue.rl_step()
            rewards_all_train[run][timestep] = reward
            avg_rewards_all[run][timestep] = rl_glue.agent.avg_reward
        weights_final[run] = rl_glue.agent.weights
        if save_counts:
            counts_all[run] = rl_glue.environment.counts
        # ToDo: add a rl_glue.cleanup() step here (PuckWorld visualization needs it, for instance)

    tqdm.write('Train_RewardRate_total\t= %f' % (np.mean(rewards_all_train)))
    tqdm.write('Train_RewardRate_lasthalf\t= %f\n' % np.mean(rewards_all_train[:,rewards_all_train.shape[1]//2:]))
    tqdm.write('AgentRewardRate_total\t= %f' % np.mean(avg_rewards_all))
    tqdm.write('AgentRewardRate_lasthalf\t= %f\n' % np.mean(avg_rewards_all[:,avg_rewards_all.shape[1]//2:]))
    log_data['rewards_all_train'] = rewards_all_train
    log_data['weights_final'] = weights_final

    if save_weights:
        log_data['weights_all'] = weights_all
        log_data['avg_v_all'] = avg_v_all
        log_data['avg_r_all'] = avg_r_all
    if save_counts:
        log_data['visitation_counts'] = counts_all

    return log_data


def get_centered_values(env, policy):
    with open("environments/centered_values.json") as f:
        centered_values_all = json.load(f)
    try:
        centered_values = centered_values_all[env][policy]
    except:
        print("Something went wrong. Have the centered_values for this policy in this environment not saved"
              "in environments/centered_values.json?")
        raise

    return centered_values


def compute_rmsve(target, weights, weighting):
    ### the standard RMSVE
    # ToDo: in the future, when actually using LFA,
    # get the features per state from the env, compute the values,
    # and then compute the MSVE.
    # Right now with one-hot features, the weights are the values.
    # if weighting == None:
    #     length = weights.size
    #     weighting = np.ones(length) / length
    rmsve = np.sqrt(np.dot((target - weights)**2, weighting))
    return rmsve


def compute_rmsve_tvr(target, weights, weighting):
    ### the RMSVE proposed by Tsitsiklis and Van Roy (1999)
    # ToDo: in the future, when actually using LFA,
    # get the features per state from the env, compute the values,
    # and then compute the MSVE.
    # Right now with one-hot features, the weights are the values.
    transformed_weights = weights - np.dot(weighting, weights)
    return compute_rmsve(target, transformed_weights, weighting)


def get_agent_weights(agent):
    agent_weights = agent.weights
    if 'Differential' in type(agent).__name__:
        offset = agent.avg_value
        agent_weights -= offset
    return agent_weights


def run_exp_learning_prediction(env, agent, config):
    num_runs = config['exp_parameters']['num_runs']
    max_steps = config['exp_parameters']['num_max_steps']
    eval_every_n_steps = config['exp_parameters']['eval_every_n_steps']
    save_weights = config['exp_parameters'].get('save_weights', 0)
    num_weights = config['exp_parameters'].get('num_weights', 1)
    reward_scale_factor = config['env_parameters'].get('reward_scale_factor', 1)

    env_info = config['env_parameters']
    agent_info = config['agent_parameters']

    print('Env: %s, Agent: %s' % (env.__name__, agent.__name__))

    log_data = {}
    assert max_steps % eval_every_n_steps == 0  # ideally not necessary, but enforcing nonetheless
    weights_all = np.zeros((num_runs, int(max_steps/eval_every_n_steps) + 1, num_weights))
    avg_v_all = np.zeros((num_runs, int(max_steps/eval_every_n_steps) + 1))
    avg_r_all = np.zeros((num_runs, int(max_steps/eval_every_n_steps) + 1))
    rmsve_all = np.zeros((num_runs, int(max_steps/eval_every_n_steps) + 1))
    rmsve_tvr_all = np.zeros((num_runs, int(max_steps/eval_every_n_steps) + 1))
    error_all = np.zeros((num_runs, int(max_steps/eval_every_n_steps) + 1)) # error used in update of reward rate estimate
    rre_all = np.zeros((num_runs, int(max_steps/eval_every_n_steps) + 1))   # error of current estimate with the true reward rate

    centered_values = get_centered_values(env.__name__, str(config['agent_parameters']['pi'][0]))

    for run in tqdm(range(num_runs)):

        agent_info['random_seed'] = run
        env_info['random_seed'] = run

        # training instance
        rl_glue = RLGlue(env, agent)
        rl_glue.rl_init(agent_info, env_info)
        rl_glue.rl_start()
        if save_weights:
            assert num_weights == rl_glue.agent.weights.size, "exp_params:num_weights should match number of agent weights"

        # evaluation instance
        eval_idx = 0

        for timestep in range(max_steps+1):

            if timestep % eval_every_n_steps == 0:
                # agent_weights = get_agent_weights(rl_glue.agent)
                agent_weights = rl_glue.agent.weights - rl_glue.agent.avg_value
                rmsve_all[run][eval_idx] = compute_rmsve(np.array(centered_values['v_pi'])*reward_scale_factor, agent_weights,
                                                         centered_values['d_pi'])
                rmsve_tvr_all[run][eval_idx] = compute_rmsve_tvr(np.array(centered_values['v_pi'])*reward_scale_factor, agent_weights,
                                                          centered_values['d_pi'])
                avg_r_all[run][eval_idx] = rl_glue.agent.avg_reward
                rre_all[run][eval_idx] = (centered_values['r_pi']*reward_scale_factor - rl_glue.agent.avg_reward)**2
                error_all[run][eval_idx] = rl_glue.agent.error

                if save_weights:
                    weights_all[run][eval_idx] = rl_glue.agent.weights
                    avg_v_all[run][eval_idx] = rl_glue.agent.avg_value
                eval_idx += 1

            reward, obs, action, _ = rl_glue.rl_step()

    tqdm.write('Eval_RMSVE_total\t= %f' % np.mean(rmsve_all))
    tqdm.write('Eval_RMSVE_lasthalf\t= %f' % np.mean(rmsve_all[:,rmsve_all.shape[1]//2:]))
    tqdm.write('Eval_R_total\t= %f' % np.mean(avg_r_all))
    tqdm.write('Eval_R_lasthalf\t= %f\n' % np.mean(avg_r_all[:,avg_r_all.shape[1]//2:]))
    tqdm.write('Eval_RMSVE_TVR_total\t= %f' % np.mean(rmsve_tvr_all))
    tqdm.write('Eval_RMSVE_TVR_lasthalf\t= %f' % np.mean(rmsve_tvr_all[:,rmsve_tvr_all.shape[1]//2:]))
    tqdm.write('Eval_RRE_total\t= %f' % np.mean(rre_all))
    tqdm.write('Eval_RRE_lasthalf\t= %f\n' % np.mean(rre_all[:,rre_all.shape[1]//2:]))

    log_data['rmsve_all'] = rmsve_all
    log_data['rmsve_tvr_all'] = rmsve_tvr_all
    log_data['avg_r_all'] = avg_r_all
    log_data['error_all'] = error_all
    log_data['rre_all'] = rre_all
    if save_weights:
        log_data['weights_all'] = weights_all
        log_data['avg_v_all'] = avg_v_all

    return log_data

def tabular_sample_based_planning(env_class, agent_class, config):
    num_runs = config['exp_parameters']['num_runs']
    max_steps = config['exp_parameters']['num_max_steps']
    eval_every_n_steps = config['exp_parameters']['eval_every_n_steps']
    save_weights = config['exp_parameters'].get('save_weights', 0)
    num_weights = config['exp_parameters'].get('num_weights', 1)
    save_counts = config['exp_parameters'].get('save_visitation_counts', False)

    env_info = config['env_parameters']
    agent_info = config['agent_parameters']

    print('Env: %s, Agent: %s' % (env_class.__name__, agent_class.__name__))

    log_data = {}
    rewards_all_train = np.zeros((num_runs, max_steps+1))
    avg_rewards_all = np.zeros((num_runs, max_steps+1))
    assert max_steps % eval_every_n_steps == 0  # ideally not necessary, but enforcing nonetheless
    weights_final = np.zeros((num_runs, num_weights))
    weights_all = np.zeros((num_runs, int(max_steps/eval_every_n_steps) + 1, num_weights))
    avg_v_all = np.zeros((num_runs, int(max_steps/eval_every_n_steps) + 1))
    avg_r_all = np.zeros((num_runs, int(max_steps/eval_every_n_steps) + 1))
    counts_all = np.zeros((num_runs, 11, 4, 2))    # for collecting AccessControl visitations

    for run in tqdm(range(num_runs)):

        agent_info['random_seed'] = run
        env_info['random_seed'] = run

        env = env_class()
        agent = agent_class({'num_states': env.num_states,
                             'num_actions': env.num_actions})
        env.env_init(env_info)
        agent.agent_init(agent_info)

        planning_dist = {'states': np.array(list(range(env.num_states))),
                         'actions':np.array(list(range(env.num_actions)))}

        eval_idx = 0

        for timestep in range(max_steps+1):

            if timestep % eval_every_n_steps == 0:
                if save_weights:
                    weights_all[run][eval_idx] = agent.weights
                    avg_v_all[run][eval_idx] = agent.avg_value
                    avg_r_all[run][eval_idx] = agent.avg_reward
                eval_idx += 1

            s = agent.rand_generator.choice(planning_dist['states'])
            a = agent.rand_generator.choice(planning_dist['actions'])
            obs, action, reward, obs_next = env.env_sample(s,a)
            agent.planning_update(obs, action, reward, obs_next)

            rewards_all_train[run][timestep] = reward
            avg_rewards_all[run][timestep] = agent.avg_reward
        weights_final[run] = agent.weights
        if save_counts:
            counts_all[run] = env.counts
        # ToDo: add a rl_glue.cleanup() step here (PuckWorld visualization needs it, for instance)

    tqdm.write('Train_RewardRate_total\t= %f' % (np.mean(rewards_all_train)))
    tqdm.write('Train_RewardRate_lasthalf\t= %f\n' % np.mean(rewards_all_train[:,rewards_all_train.shape[1]//2:]))
    tqdm.write('AgentRewardRate_total\t= %f' % np.mean(avg_rewards_all))
    tqdm.write('AgentRewardRate_lasthalf\t= %f\n' % np.mean(avg_rewards_all[:,avg_rewards_all.shape[1]//2:]))
    log_data['rewards_all_train'] = rewards_all_train
    log_data['weights_final'] = weights_final
    if save_weights:
        log_data['weights_all'] = weights_all
        log_data['avg_v_all'] = avg_v_all
        log_data['avg_r_all'] = avg_r_all
    if save_counts:
        log_data['visitation_counts'] = counts_all

    return log_data


def test_RMSVE():
    target = np.array([0.0, 0.0, 0.0, 0.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    weighting = np.array([0.25, 0.25, 0.25, 0.25])
    print('RMSVE: %.2f' % compute_rmsve(target, weights, weighting))
    print('RMSVE_TVR: %.2f' % compute_rmsve_tvr(target, weights, weighting))


if __name__=='__main__':
    test_RMSVE()