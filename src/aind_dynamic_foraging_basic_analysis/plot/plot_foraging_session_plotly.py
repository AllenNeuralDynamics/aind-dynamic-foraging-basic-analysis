"""Interactive plotly figures that can be used in the Streamlit app as well as 
Jupyter Notebook
"""

import plotly.graph_objects as go


def plot_session_events_plotly(df_events, adjust_time=True):
    """A plotly version of plot_foraging_session.plot_session_scroller

    Creates a plot of the session.
    Plots left/right licks/rewards, and go cues as vertical lines from bottom to top.

    df_events: A tidy dataframe of session events.

    adjust_time (bool): If True, resets time=0 to the first event of the session.

    EXAMPLE:
    df_events = nwb_utils.create_events_df(nwb_object)
    plot_foraging_session_plotly.plot_session_events_plotly(df_events)
    """

    if adjust_time:
        df_events = df_events.copy()
        df_events["timestamps"] = df_events["timestamps"] - df_events.iloc[0]["timestamps"]

    xmin = df_events.iloc[0]["timestamps"]
    xmax = df_events.iloc[-1]["timestamps"]

    params = {
        "left_lick": 0.125,
        "right_lick": 0.875,
        "left_reward": 0.375,
        "right_reward": 0.625,
        "go_cue_bottom": 0,
        "go_cue_top": 1,
    }

    fig = go.Figure()

    left_licks = df_events.query('event == "left_lick_time"')
    left_times = left_licks.timestamps.values
    fig.add_trace(
        go.Scattergl(
            x=left_times,
            y=[params["left_lick"]] * len(left_times),
            mode="markers",
            marker=dict(symbol="line-ns", line_color="black", size=10, line_width=2),
            name="Left Lick",
        )
    )

    right_licks = df_events.query('event == "right_lick_time"')
    right_times = right_licks.timestamps.values
    fig.add_trace(
        go.Scattergl(
            x=right_times,
            y=[params["right_lick"]] * len(right_times),
            mode="markers",
            marker=dict(symbol="line-ns", line_color="black", size=10, line_width=2),
            name="Right Lick",
        )
    )

    left_reward_deliverys = df_events.query('event == "left_reward_delivery_time"')
    left_times = left_reward_deliverys.timestamps.values
    fig.add_trace(
        go.Scattergl(
            x=left_times,
            y=[params["left_reward"]] * len(left_times),
            mode="markers",
            marker=dict(symbol="line-ns", size=10, line_color="red", line_width=3),
            name="Left Reward",
        )
    )

    right_reward_deliverys = df_events.query('event == "right_reward_delivery_time"')
    right_times = right_reward_deliverys.timestamps.values
    fig.add_trace(
        go.Scattergl(
            x=right_times,
            y=[params["right_reward"]] * len(right_times),
            mode="markers",
            marker=dict(symbol="line-ns", size=10, line_color="red", line_width=3),
            name="Right Reward",
        )
    )

    go_cues = df_events.query('event == "goCue_start_time"')
    go_cue_times = go_cues.timestamps.values
    for n, time in enumerate(go_cue_times):
        fig.add_trace(
            go.Scattergl(
                x=[time, time],
                y=[params["go_cue_bottom"], params["go_cue_top"]],
                mode="lines",
                line=dict(color="blue", width=0.3),
                legendgroup="Go Cue group",
                showlegend=(n == 0),
                name="Go Cue",
                hovertemplate=f"Go Cue, Trial {n+1}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Session Scroller",
        xaxis_title="Time (s)",
        yaxis=dict(
            title="Events",
            tickvals=[0.125, 0.875, 0.375, 0.625],
            ticktext=["Left Licks", "Right Licks", "Left Reward", "Right Reward"],
        ),
        xaxis=dict(range=[xmin, xmax]),
        showlegend=True,
        height=400,
        width=1000,
    )

    return fig
