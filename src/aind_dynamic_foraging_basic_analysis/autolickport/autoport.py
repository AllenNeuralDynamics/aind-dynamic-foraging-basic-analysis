# from aind_dynamic_foraging_multisession_analysis import multisession_plot as ms_plot
# from aind_dynamic_foraging_data_utils import nwb_utils as nu
### Notes
# Make a function that adds trial vector of decision points in autoport
# Then we can just pass into normal plotting functions


def annotate_autolickport(nwb):
    """
    annotates decision points, and decisions to move
    """

    # TODO Add input option for this
    params = {
        "max_water_reward_attempts": 10,
        "trial_interval": 50,
        "bias_upper_threshold": 0.7,
        "bias_lower_threshold": 0.3,
        "lick_spout_movement": {"range_um": 300.0, "step_size_um": 50.0},
    }

    # TODO add check for df_trials exisitence
    df = nwb.df_trials.copy()

    last_bias_intervention = 0
    water_reward_attempts = 0
    autoport_check =[]
    autoport_water = []
    autoport_movement = []
    for trial_number in range(len(df)):
        this_check=False
        this_water=False
        this_movement=False
        if params["trial_interval"] <= trial_number - last_bias_intervention:
            this_check=True
            if abs(df.loc[trial_number]["side_bias"]) > params["bias_upper_threshold"]:
                last_bias_intervention = trial_number

                # first try water intervention
                if water_reward_attempts < params["max_water_reward_attempts"]:
                    print(f"Bias over threshold. Attempting water intervention.")
                    this_water = True
                else:
                    print(f"Maximum watering attempts exceeded. Moving lickspouts for bias. ")
                    water_reward_attempts = 0
                    this_movement=True
        autoport_check.append(this_check)
        autoport_water.append(this_water)
        autoport_movement.append(this_movement)
    df['autoport_check'] = autoport_check
    df['autoport_water'] = autoport_water
    df['autoport_movement'] = autoport_movement 
    return df 
