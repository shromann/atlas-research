# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import polars as pl
import altair as alt
import streamlit as st


# alt.data_transformers.enable("vegafusion")
# alt.theme.enable('carbong100')


df = (pl.read_parquet("../data/processed/btc-sample.parquet")
    .with_columns(
        date = pl.col('timestamp').dt.date(), 
        dt   = pl.col('timestamp').diff().dt.total_minutes())
    .with_columns(
        log_return  = pl.col('close').log().diff() / pl.col('dt'),
        vwap_return = pl.col('vwap').log().diff() / pl.col('dt'))
    .with_columns(
        volatility  = pl.col('log_return').abs())
    .with_columns(
        log_return_ema5  = pl.col('log_return').ewm_mean(span=5),
        log_return_ema10 = pl.col('log_return').ewm_mean(span=10),
        log_return_ema30 = pl.col('log_return').ewm_mean(span=30),
        log_return_ema60 = pl.col('log_return').ewm_mean(span=60),
        log_return_ema90 = pl.col('log_return').ewm_mean(span=90),
        log_return_ema120 = pl.col('log_return').ewm_mean(span=120))
    .with_columns(
        recency_5_10  = pl.col('log_return_ema5')  / pl.col('log_return_ema10'),
        recency_5_60  = pl.col('log_return_ema5')  / pl.col('log_return_ema60'),
        recency_5_90 = pl.col('log_return_ema5') / pl.col('log_return_ema90'),
        recency_5_120 = pl.col('log_return_ema5') / pl.col('log_return_ema120'))
    .with_columns(
        next_log_return = pl.col('log_return').shift(-1),
        net_return = pl.col('log_return').cum_sum().exp())
    .drop_nulls(subset=['dt'])
)

# %%
brush = alt.selection_interval()

close = alt.Chart(df).mark_point(size=1).encode(
    x='timestamp',
    y=alt.Y('net_return', scale=alt.Scale(
        domain=[
            df.select('net_return').min().item() - 0.5 * df.select('net_return').std().item(),
            df.select('net_return').max().item() + 0.5 * df.select('net_return').std().item()
        ]
    )),
    color=alt.when(brush).then(alt.value('#7ecbff')).otherwise(alt.value("lightgray")),
    tooltip=[
        alt.Tooltip('timestamp', title='Date'),
        alt.Tooltip('net_return', title='Net Return')]
).properties(
    title='Close Price'
).add_params(brush)

log_ret = alt.Chart(df).mark_point().encode(
    x='timestamp',
    y='log_return',
    color=alt.value("#547cff")
).properties(
    title='Log Returns'
).transform_filter(brush)


bin_width = 0.00025
w = max([abs(df.select('log_return').min().item()), df.select('log_return').max().item()])
bin_extent = [-w, w]
log_ret_hist = alt.Chart(df).mark_bar(color="#547cff").encode(
    y=alt.Y('count()', title='Frequency'),
    x=alt.X(
        'log_return',
        bin=alt.Bin(step=bin_width, extent=bin_extent),
        title='Log Return',
        scale=alt.Scale(domain=bin_extent)
    )
).properties(
    title='Histogram of Log Returns'
).transform_filter(brush)

mo_score_5_10 = alt.Chart(df).mark_point(color='#7ecbff').encode(
    x='timestamp',
    y='recency_5_10'
).properties(
    title='recency_5_10'
).transform_filter(brush)

mo_score_5_60 = alt.Chart(df).mark_point(color='#7ecbff').encode(
    x='timestamp',
    y='recency_5_60'
).properties(
    title='Recency_5_60'
).transform_filter(brush)

mo_score_5_90 = alt.Chart(df).mark_point(color='#7ecbff').encode(
    x='timestamp',
    y='recency_5_90'
).properties(
    title='Recency_5_90'
).transform_filter(brush)

mo_score_5_120 = alt.Chart(df).mark_point(color='#7ecbff').encode(
    x='timestamp',
    y='recency_5_120'
).properties(
    title='Recency_5_120'
).transform_filter(brush)


# %%

dashboard = (
    close & 
    (log_ret | log_ret_hist) & 
    (mo_score_5_10 | mo_score_5_60 | mo_score_5_90 | mo_score_5_120)
)


st.altair_chart(
    dashboard,
    use_container_width=False,
    theme="streamlit"
)

# %%
