import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import Figure


def _empty_figure(message: str) -> Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, font=dict(size=14))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    return fig


def emotion_distribution_chart(frequency_df: pd.DataFrame) -> Figure:
    """Create a bar chart showing emotion distribution by count."""
    if frequency_df.empty:
        return _empty_figure("No emotion distribution data available.")

    fig = px.bar(
        frequency_df,
        x="emotion",
        y="count",
        color="emotion",
        title="Emotion Distribution",
        labels={"emotion": "Emotion", "count": "Count"},
    )
    fig.update_layout(showlegend=False)
    return fig


def confidence_scores_chart(confidence_df: pd.DataFrame) -> Figure:
    """Create a bar chart comparing average confidence across models."""
    if confidence_df.empty:
        return _empty_figure("No confidence data available.")

    fig = px.bar(
        confidence_df,
        x="model",
        y="average_confidence",
        color="model",
        title="Average Model Confidence",
        labels={"model": "Model", "average_confidence": "Average Confidence"},
        text="average_confidence",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(yaxis=dict(range=[0, 1]), showlegend=False, margin=dict(b=50))
    return fig


def model_comparison_chart(comparison_df: pd.DataFrame) -> Figure:
    """Create a bar chart showing model agreement versus disagreement."""
    if comparison_df.empty:
        return _empty_figure("No model comparison data available.")

    fig = px.bar(
        comparison_df,
        x="comparison",
        y="count",
        color="comparison",
        title="BiLSTM vs BERT Prediction Agreement",
        labels={"comparison": "Comparison", "count": "Count"},
    )
    fig.update_layout(showlegend=False)
    return fig


def daily_trend_chart(daily_df: pd.DataFrame) -> Figure:
    """Create a line chart for daily interaction counts and confidence trends."""
    if daily_df.empty:
        return _empty_figure("No daily trend data available.")

    if "date" not in daily_df.columns or daily_df["date"].isnull().all():
        return _empty_figure("Daily trend data is missing date values.")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=daily_df["date"],
            y=daily_df["count"],
            name="Interactions",
            marker_color="#636efa",
        )
    )
    if "average_confidence" in daily_df.columns:
        fig.add_trace(
            go.Scatter(
                x=daily_df["date"],
                y=daily_df["average_confidence"],
                mode="lines+markers",
                name="Average Confidence",
                marker=dict(color="#ef553b"),
                yaxis="y2",
            )
        )
    fig.update_layout(
        title="Daily Predictions and Confidence Trends",
        xaxis_title="Date",
        yaxis_title="Predictions",
        yaxis2=dict(
            title="Average Confidence",
            overlaying="y",
            side="right",
            range=[0, 1],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def mixed_emotion_chart(mixed_df: pd.DataFrame) -> Figure:
    """Create a bar chart for mixed emotion frequency counts."""
    if mixed_df.empty:
        return _empty_figure("No mixed emotion data available.")

    fig = px.bar(
        mixed_df,
        x="mixed_emotion",
        y="count",
        color="mixed_emotion",
        title="Mixed Emotion Frequency",
        labels={"mixed_emotion": "Mixed Emotion", "count": "Count"},
    )
    fig.update_layout(showlegend=False)
    return fig


def confidence_trend_chart(confidence_trend_df: pd.DataFrame) -> Figure:
    """Create a line chart for confidence trend over time."""
    if confidence_trend_df.empty:
        return _empty_figure("No confidence trend data available.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=confidence_trend_df["date"], y=confidence_trend_df["average_confidence"], mode="lines+markers", name="Daily Confidence"))
    fig.add_trace(go.Scatter(x=confidence_trend_df["date"], y=confidence_trend_df["rolling_confidence"], mode="lines", name="7-Day Rolling Average"))
    fig.update_layout(title="Confidence Trend", xaxis_title="Date", yaxis_title="Confidence", yaxis=dict(range=[0, 1]))
    return fig


def subject_distribution_chart(subject_df: pd.DataFrame) -> Figure:
    """Create a bar chart for subject distribution."""
    if subject_df.empty:
        return _empty_figure("No subject distribution data available.")

    fig = px.bar(subject_df, x="subject", y="count", color="subject", title="Subject Distribution")
    fig.update_layout(showlegend=False)
    return fig


def emotion_timeline_chart(timeline_df: pd.DataFrame) -> Figure:
    """Create a line chart showing emotion changes over time."""
    if timeline_df.empty:
        return _empty_figure("No emotion timeline data available.")

    fig = px.line(timeline_df, x="date", y="count", color="final_prediction", title="Emotion Timeline")
    fig.update_layout(xaxis_title="Date", yaxis_title="Count")
    return fig


def confidence_by_emotion_chart(confidence_df: pd.DataFrame) -> Figure:
    """Create a bar chart for average confidence by emotion."""
    if confidence_df.empty:
        return _empty_figure("No confidence-by-emotion data available.")

    fig = px.bar(confidence_df, x="emotion", y="average_confidence", color="emotion", title="Average Confidence by Emotion")
    fig.update_layout(showlegend=False, yaxis=dict(range=[0, 1]))
    return fig
