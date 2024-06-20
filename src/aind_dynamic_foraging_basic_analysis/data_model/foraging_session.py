from pydantic import BaseModel, field_validator, model_validator
from typing import Optional, Union, Dict, List
import numpy as np

class PhotostimData(BaseModel):
    trial: Union[np.ndarray, List[int]]
    power: Union[np.ndarray, List[float]]
    stim_epoch: Optional[Union[np.ndarray, List[str]]] = None
    
    class Config:
        arbitrary_types_allowed = True


class ForagingSessionData(BaseModel):
    choice_history: Union[np.ndarray, list]
    reward_history: Union[np.ndarray, list]
    p_reward: Optional[Union[np.ndarray, list]] = None
    random_number: Optional[Union[np.ndarray, list]] = None
    autowater_offered: Optional[Union[np.ndarray, list]] = None
    fitted_data: Optional[Union[np.ndarray, list]] = None
    photostim: Optional[PhotostimData] = None

    class Config:
        arbitrary_types_allowed = True

    @field_validator('choice_history', 'reward_history', 'p_reward', 
                     'random_number', 'autowater_offered', 'fitted_data', mode='before')
    def convert_to_ndarray(cls, v):
        return np.array(v) if v is not None else None

    @model_validator(mode='after')
    def check_all_fields(cls, values):
        choice_history = values.choice_history
        reward_history = values.reward_history
        p_reward = values.p_reward
        random_number = values.random_number
        autowater_offered = values.autowater_offered
        fitted_data = values.fitted_data
        photostim = values.photostim

        if not np.all(np.isin(choice_history, [0.0, 1.0]) | np.isnan(choice_history)):
            raise ValueError("choice_history must contain only 0, 1, or np.nan.")

        if not np.all(np.isin(reward_history, [0.0, 1.0])):
            raise ValueError("reward_history must contain only 0 (False) or 1 (True).")

        if choice_history.shape != reward_history.shape:
            raise ValueError("choice_history and reward_history must have the same shape.")

        if p_reward.shape != (2, len(choice_history)):
            raise ValueError("reward_probability must have the shape (2, n_trials)")

        if random_number is not None:
            if random_number.shape != p_reward.shape:
                raise ValueError("random_number must have the same shape as reward_probability.")

        if autowater_offered is not None:
            if autowater_offered.shape != choice_history.shape:
                raise ValueError("autowater_offered must have the same shape as choice_history.")

        if fitted_data is not None:
            if fitted_data.shape[0] != choice_history.shape[0]:
                raise ValueError("fitted_data must have the same length as choice_history.")

        if photostim is not None:
            if len(photostim.trial) != len(photostim.power):
                raise ValueError("photostim.trial must have the same length as photostim.power.")
            if photostim.stim_epoch is not None:
                if len(photostim.stim_epoch) != len(photostim.power):
                    raise ValueError("photostim.stim_epoch must have the same length as photostim.power.")

        return values

