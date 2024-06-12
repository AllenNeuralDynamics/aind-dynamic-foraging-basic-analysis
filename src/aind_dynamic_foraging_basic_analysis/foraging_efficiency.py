from typing import Tuple, List, Literal, Union

import numpy as np


def foraging_efficiency(
    choice_history: Union[List, np.ndarray],
    reward_history: Union[List, np.ndarray],
    reward_probability: Union[List, np.ndarray],
    random_number: Union[List, np.ndarray] = None,
    baited: bool = True,
) -> Tuple[float, float]:
    """Compute foraging efficiency for baited or non-baited 2-arm bandit task.

    Definition of foraging efficiency (i.e., a single value measurement to quantify "performance") is a 
    complex and unsolved topic especially in the context of reward baiting. I have a presentation for this 
    (https://alleninstitute.sharepoint.com/:p:/s/NeuralDynamics/EejfBIEvFA5DjmfOZV8atWgBx7q68GsKnavkVrfghL9y8g?e=OnR5r4). 

    Simply speaking, foraging eff = actual reward of the mouse / reward of an optimal_forager in the same session.
    The question is how to define the optimal_forager. 

    1. For the coupled-block-with-baiting task (Jeremiah's 2019 Neuron paper), 
    I assume the optimal_forager knows the underlying reward probability and the baiting dynamics, 
    and do the optimal choice pattern ("fix-and-sample" in this case, see references on p.24 of my slides). 

    2. For the non-baiting task (Cooper Grossman), I assume the optimal_forager knows the underlying probability and 
    makes the greedy choice, i.e., always choose the better side. 

    This might not be the best way because the optimal_foragers I assumed is kind of cheating in the sense that 
    they already know the underlying probability, but it sets an upper bound for all realistic agents, 
    at least in an average sense. 

    For a single session, however, there are chances where the efficiency
    can be larger than 1 because of the randomness of the task (sometimes the mice are really 
    lucky that they get more reward than performing "optimally"). To **partially** alleviate this fluctuation, when the
    user provides the actual random_number that were generated in that session, I calculate the efficiency based on 
    simulating the optimal_forager's performance with the same set of random numbers, yielding the foraging_efficiency_actual_random_seed

    Parameters
    ----------
    choice_history : Union[List, np.ndarray]
        Choice history (0 = left choice, 1 = right choice, np.nan = ignored). 
        Notes: 
             1. choice_history should exclude free water trials.
             2. choice_history allows ignored trials, but we'll remove them in the foraging efficiency calculation.
    reward_history : Union[List, np.ndarray]
        Reward history (0 = unrewarded, 1 = rewarded).
    reward_probability : Union[List, np.ndarray]
        Reward probability for both sides. The size should be (2, len(choice_history)).
    random_number : Union[List, np.ndarray], optional
        The actual random numbers generated in the session (see above). Must be the same shape as reward_probability, by default None.
    baited : bool, optional
        Whether the task is baited or not, by default True.xus

    Returns
    -------
    Tuple[float, float]
        foraging_efficiency, foraging_efficiency_actual_random_seed
    """
    
    # Choose the optimal_forager function based on baiting
    if baited:
        reward_optimal_func = _reward_optimal_forager_baiting
    else:
        reward_optimal_func = _reward_optimal_forager_no_baiting
    
    # Formatting
    choice_history = np.array(choice_history, dtype=float) # Convert None to np.nan, if any
    reward_history = np.array(reward_history, dtype=float)
    n_trials = len(choice_history)
    
    if choice_history.shape != reward_history.shape:
        raise ValueError(f"choice_history and reward_history must have the same shape.")
        
    if reward_probability.shape != (2, n_trials):
        raise ValueError(f"reward_probability must have the shape (2, n_trials)")
        
    if random_number is not None and random_number.shape != reward_probability.shape:
        raise ValueError(f"random_number must have the same shape as reward_probability.")
        
    # Foraging_efficiency is calculated only on finished trials
    ignored = np.isnan(choice_history)
    choice_history = choice_history[~ignored]
    reward_history = reward_history[~ignored]
    reward_probability = reward_probability[:, ~ignored]
    random_number = random_number[:, ~ignored] if random_number is not None else None
    
    # Compute reward of the optimal forager
    reward_optimal, reward_optimal_random_seed = reward_optimal_func(
        p_Ls=reward_probability[0],
        p_Rs=reward_probability[1],
        random_number_L=random_number[0] if random_number is not None else None,
        random_number_R=random_number[1] if random_number is not None else None,
    )
    reward_actual = reward_history.sum()
    
    return reward_actual / reward_optimal, reward_actual / reward_optimal_random_seed


def _reward_optimal_forager_no_baiting(p_Ls, p_Rs, random_number_L, random_number_R):

    # --- Optimal-aver (use optimal expectation as 100% efficiency) ---
    reward_optimal = np.nanmean(np.max([p_Ls, p_Rs], axis=0)) * len(p_Ls)

    if random_number_L is None:
        return foraging_efficiency, np.nan

    # --- Optimal-actual (uses the actual random numbers by simulation)
    reward_refills = np.vstack(
        [p_Ls >= random_number_L, p_Rs >= random_number_R])
    # Greedy choice, assuming the agent knows the groundtruth
    optimal_choices = np.argmax([p_Ls, p_Rs], axis=0)
    reward_optimal_random_seed = reward_refills[0][optimal_choices == 0].sum(
    ) + reward_refills[1][optimal_choices == 1].sum()

    return reward_optimal, reward_optimal_random_seed


def _reward_optimal_forager_baiting(p_Ls, p_Rs, random_number_L, random_number_R):

    # --- Optimal-aver (use optimal expectation as 100% efficiency) ---
    p_stars = np.zeros_like(p_Ls)
    for i, (p_L, p_R) in enumerate(zip(p_Ls, p_Rs)):   # Sum over all ps
        p_max = np.max([p_L, p_R])
        p_min = np.min([p_L, p_R])
        if p_min == 0 or p_max >= 1:
            p_stars[i] = p_max
        else:
            m_star = np.floor(np.log(1-p_max)/np.log(1-p_min))
            p_stars[i] = p_max + \
                (1-(1-p_min)**(m_star + 1)-p_max**2)/(m_star+1)

    reward_optimal = np.nanmean(p_stars) * len(p_Ls)

    if random_number_L is None:
        return reward_optimal, np.nan

    # --- Optimal-actual (uses the actual random numbers by simulation)
    block_start_ind_left = np.where(
        np.diff(np.hstack([np.inf, p_Ls, np.inf])))[0].tolist()
    block_start_ind_right = np.where(
        np.diff(np.hstack([np.inf, p_Rs, np.inf])))[0].tolist()
    block_start_ind_effective = np.sort(
        np.unique(np.hstack([block_start_ind_left, block_start_ind_right])))

    reward_refills = [p_Ls >= random_number_L, p_Rs >= random_number_R]
    reward_optimal_random_seed = 0

    # Generate optimal choice pattern
    for b_start, b_end in zip(block_start_ind_effective[:-1], block_start_ind_effective[1:]):
        p_max = np.max([p_Ls[b_start], p_Rs[b_start]])
        p_min = np.min([p_Ls[b_start], p_Rs[b_start]])
        side_max = np.argmax([p_Ls[b_start], p_Rs[b_start]])

        # Get optimal choice pattern and expected optimal rate
        if p_min == 0 or p_max >= 1:
            # Greedy is obviously optimal
            this_choice = np.array([1] * (b_end-b_start))
        else:
            m_star = np.floor(np.log(1-p_max)/np.log(1-p_min))
            this_choice = np.array(
                (([1]*int(m_star)+[0]) * (1+int((b_end-b_start)/(m_star+1))))[:b_end-b_start])

        # Do simulation, using optimal choice pattern and actual random numbers
        reward_refill = np.vstack([reward_refills[1 - side_max][b_start:b_end],
                                   reward_refills[side_max][b_start:b_end]]).astype(int)  # Max = 1, Min = 0
        reward_remain = [0, 0]
        for t in range(b_end - b_start):
            reward_available = reward_remain | reward_refill[:, t]
            reward_optimal_random_seed += reward_available[this_choice[t]]
            reward_remain = reward_available.copy()
            reward_remain[this_choice[t]] = 0

    return reward_optimal, reward_optimal_random_seed if reward_optimal_random_seed else np.nan