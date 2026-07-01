"""Init package"""

__version__ = "0.4.0"

from .metrics.foraging_efficiency import compute_foraging_efficiency  # noqa: F401
from .plot.plot_foraging_session import plot_foraging_session  # noqa: F401
from .plot.plot_foraging_session_plotly import (  # noqa: F401
    plot_foraging_session_plotly,
    plot_foraging_session_nwb_plotly,
    plot_session_in_time_plotly,
    plot_session_in_time_nwb_plotly,

)
