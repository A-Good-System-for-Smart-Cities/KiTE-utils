from plotly.subplots import make_subplots
import plotly.graph_objs as go
import kaleido


# Plot Histogram of X-Test, prob_pos
def plot_probability_frequency(prob_pos, ELCE2_, name="Name_of_model"):
    hist = go.Histogram(x=prob_pos, name=name)
    ELCE_trace = plot_ELCE2_number_line(ELCE2_)
    fig = make_subplots(rows=2, cols=1, row_heights=[0.85, 0.15])
    fig.append_trace(hist, 1, 1)
    fig.append_trace(ELCE_trace, 2, 1)
    fig.update_layout(title_text=f"{name} and ELCE2 Estimator")
    return fig


def plot_ELCE2_number_line(ELCE2_):
    ELCE_trace = go.Scatter(
        x=[ELCE2_ * 100], y=[0, 0], mode="markers", marker_size=20, name="ELCE2"
    )
    return ELCE_trace


"""
kwargs --  label axis
Q-Q reliability curves?
"""
