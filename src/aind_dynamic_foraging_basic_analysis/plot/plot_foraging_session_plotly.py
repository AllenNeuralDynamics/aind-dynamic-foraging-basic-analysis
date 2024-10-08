"""Interactive plotly figures that can be used in the Streamlit app as well as
Jupyter Notebook
"""

import plotly.graph_objects as go


def plot_session_in_time_plotly(  # noqa: C901 pragma: no cover
    df_events, adjust_time=True, fip_df=None
):
    """A plotly version of plot_foraging_session.plot_session_scroller

    Creates a plot of the session in time (not in trial).
    Plots left/right licks/rewards, and go cues as vertical lines from bottom to top.

    df_events: A tidy dataframe of session events.

    fip_df is a tidy dataframe of FIP measurements generated by
        aind_dynamic_foraging_data_utils.nwb_utils.create_fib_df(tidy=True)

    adjust_time (bool): If True, resets time=0 to the first event of the session.

    EXAMPLE:
    df_events = nwb_utils.create_events_df(nwb_object)
    fip_df = nwb_utils.create_fib_df(nwb_object, tidy=True)
    plot_foraging_session_plotly.plot_session_events_plotly(df_events)
    """

    if adjust_time:
        start_time = df_events.iloc[0]["timestamps"]
        df_events = df_events.copy()
        df_events["timestamps"] = df_events["timestamps"] - start_time

        if fip_df is not None:
            fip_df = fip_df.copy()
            fip_df["timestamps"] = fip_df["timestamps"] - start_time

    xmin = df_events.iloc[0]["timestamps"]
    xmax = df_events.iloc[-1]["timestamps"]

    params = {
        "left_lick": 0.125,
        "right_lick": 0.875,
        "left_reward": 0.375,
        "right_reward": 0.625,
        "go_cue_bottom": 0,
        "go_cue_top": 1,
        "G_1_preprocessed_bottom": 1,
        "G_1_preprocessed_top": 2,
        "G_2_preprocessed_bottom": 2,
        "G_2_preprocessed_top": 3,
        "R_1_preprocessed_bottom": 3,
        "R_1_preprocessed_top": 4,
        "R_2_preprocessed_bottom": 4,
        "R_2_preprocessed_top": 5,
    }

    yticks = [
        params["left_lick"],
        params["right_lick"],
        params["left_reward"],
        params["right_reward"],
    ]
    ylabels = ["left licks", "right licks", "left reward", "right reward"]
    ycolors = ["k", "k", "r", "r"]

    fig = go.Figure()

    # Add FIP traces
    if fip_df is not None:
        fip_channels = [
            "G_2_preprocessed",
            "G_1_preprocessed",
            "R_2_preprocessed",
            "R_1_preprocessed",
        ]
        present_channels = fip_df["event"].unique()
        for index, channel in enumerate(fip_channels):
            if channel in present_channels:
                yticks.append(
                    (params[channel + "_top"] - params[channel + "_bottom"]) / 2
                    + params[channel + "_bottom"]
                )
                ylabels.append(channel)
                if "G_1" in channel:
                    color = "green"
                elif "G_2" in channel:
                    color = "darkgreen"
                elif "R_1" in channel:
                    color = "red"
                elif "R_2" in channel:
                    color = "darkred"
                ycolors.append(color)
                C = fip_df.query("event == @channel").copy()
                C["data"] = C["data"] - C["data"].min()
                C["data"] = C["data"].values / C["data"].max()
                C["data"] += params[channel + "_bottom"]

                # Plot the data using go.Scattergl
                fig.add_trace(
                    go.Scattergl(
                        x=C.timestamps.values,
                        y=C.data.values,
                        mode="lines",
                        line=dict(color=color),
                        name=channel,
                    )
                )

                # Add a horizontal reference line (axhline equivalent)
                fig.add_trace(
                    go.Scattergl(
                        x=[C.timestamps.min(), C.timestamps.max()],
                        y=[params[channel + "_bottom"], params[channel + "_bottom"]],
                        mode="lines",
                        line=dict(color="black", width=1, dash="solid"),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

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
            tickvals=yticks,
            ticktext=ylabels,
        ),
        xaxis=dict(range=[xmin, xmax]),
        showlegend=True,
        height=800,
        width=1300,
    )

    return fig
