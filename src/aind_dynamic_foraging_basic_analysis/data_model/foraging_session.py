"""
Pydantic data model of foraging session data for shared validation.

Maybe this is an overkill...
"""

from typing import List, Optional

import numpy as np
from pydantic import BaseModel, field_validator, model_validator


class PhotostimData(BaseModel):
    """Photostimulation data"""

    trial: List[int]
    power: List[float]
    stim_epoch: Optional[List[str]] = None

    class Config:
        """Allow np.ndarray as input"""

        arbitrary_types_allowed = True


class ForagingSessionData(BaseModel):
    """Shared validation for foraging session data"""

    choice_history: np.ndarray
    reward_history: np.ndarray
    p_reward: Optional[np.ndarray] = None
    random_number: Optional[np.ndarray] = None
    autowater_offered: Optional[np.ndarray] = None
    fitted_data: Optional[np.ndarray] = None
    photostim: Optional[PhotostimData] = None

    class Config:
        """Allow np.ndarray as input"""

        arbitrary_types_allowed = True

    @field_validator(
        "choice_history",
        "reward_history",
        "p_reward",
        "random_number",
        "autowater_offered",
        "fitted_data",
        mode="before",
    )
    @classmethod
    def convert_to_ndarray(cls, v, info):
        """Always convert to numpy array"""
        return (
            np.array(
                v,
                dtype=(
                    "bool"
                    if info.field_name in ["reward_history", "autowater_offered"]  # Turn to bool
                    else None
                ),
            )
            if v is not None
            else None
        )

    @model_validator(mode="after")
    def check_all_fields(cls, values):  # noqa: C901
        """Check consistency of all fields"""

        choice_history = values.choice_history
        reward_history = values.reward_history
        p_reward = values.p_reward
        random_number = values.random_number
        autowater_offered = values.autowater_offered
        fitted_data = values.fitted_data
        photostim = values.photostim

        if not np.all(np.isin(choice_history, [0.0, 1.0]) | np.isnan(choice_history)):
            raise ValueError("choice_history must contain only 0, 1, or np.nan.")

        if choice_history.shape != reward_history.shape:
            raise ValueError("choice_history and reward_history must have the same shape.")

        if p_reward.shape != (2, len(choice_history)):
            raise ValueError("reward_probability must have the shape (2, n_trials)")

        if random_number is not None and random_number.shape != p_reward.shape:
            raise ValueError("random_number must have the same shape as reward_probability.")

        if autowater_offered is not None and autowater_offered.shape != choice_history.shape:
            raise ValueError("autowater_offered must have the same shape as choice_history.")

        if fitted_data is not None and fitted_data.shape[0] != choice_history.shape[0]:
            raise ValueError("fitted_data must have the same length as choice_history.")

        if photostim is not None:
            if len(photostim.trial) != len(photostim.power):
                raise ValueError("photostim.trial must have the same length as photostim.power.")
            if photostim.stim_epoch is not None and len(photostim.stim_epoch) != len(
                photostim.power
            ):
                raise ValueError(
                    "photostim.stim_epoch must have the same length as photostim.power."
                )

        return values
