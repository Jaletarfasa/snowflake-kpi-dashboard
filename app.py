import streamlit as st
import pandas as pd
import numpy as np
import snowflake.connector

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# -----------------------------------------------------------------------------
# 0. STREAMLIT PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Snowflake KPI Dashboard",
    layout="wide",
)

st.title("Snowflake KPI Dashboard")
st.caption("Customers · Revenue · Channels · Funnel (Snowflake, filtered)")


# -----------------------------------------------------------------------------
# 1. SNOWFLAKE CONNECTION HELPERS
# -----------------------------------------------------------------------------
import streamlit as st
import snowflake.connector

def get_connection():
    """
    Create a Snowflake connection using Streamlit secrets.
    In local dev, you can also create a .streamlit/secrets.toml file.
    """
    return snowflake.connector.connect(
        account=st.secrets["SNOWFLAKE_ACCOUNT"],
        user=st.secrets["SNOWFLAKE_USER"],
        password=st.secrets["SNOWFLAKE_PASSWORD"],
        warehouse=st.secrets["SNOWFLAKE_WAREHOUSE"],
        database=st.secrets["SNOWFLAKE_DATABASE"],
        schema=st.secrets["SNOWFLAKE_SCHEMA"],
        role=st.secrets["SNOWFLAKE_ROLE"],
    )



def run_query(sql: str) -> pd.DataFrame:
    """
    Run a SQL query in Snowflake and return a pandas DataFrame.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(sql)
        df = cur.fetch_pandas_all()
    finally:
        cur.close()
        conn.close()
    return df


# -----------------------------------------------------------------------------
# 2. LOAD DATA (CACHED)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data from Snowflake...")
def load_data():
    df_users = run_query(
        """
        SELECT
            USER_ID,
            EMAIL,
            COUNTRY,
            MARKETING_OPT_IN
        FROM CORE.DIM_USER
        """
    )

    df_orders = run_query(
        """
        SELECT
            ORDER_ID,
            USER_ID,
            STATUS,
            ORDER_TS,
            GROSS_REVENUE
        FROM CORE.FACT_ORDER
        """
    )

    df_oi = run_query(
        """
        SELECT
            ORDER_ID,
            STATUS,
            ORDER_TS,
            QTY,
            CHANNEL,
            LINE_AMOUNT
        FROM CORE.FACT_ORDER_ITEM
        """
    )

    df_events = run_query(
        """
        SELECT
            USER_ID,
            SESSION_ID,
            EVENT_TYPE,
            TS
        FROM CORE.FACT_EVENT
        """
    )

    # Ensure datetime types
    if not pd.api.types.is_datetime64_any_dtype(df_orders["ORDER_TS"]):
        df_orders["ORDER_TS"] = pd.to_datetime(df_orders["ORDER_TS"])

    if not pd.api.types.is_datetime64_any_dtype(df_oi["ORDER_TS"]):
        df_oi["ORDER_TS"] = pd.to_datetime(df_oi["ORDER_TS"])

    if not df_events.empty and not pd.api.types.is_datetime64_any_dtype(df_events["TS"]):
        df_events["TS"] = pd.to_datetime(df_events["TS"])

    return df_users, df_orders, df_oi, df_events


df_users, df_orders, df_oi, df_events = load_data()


# -----------------------------------------------------------------------------
# 3. SIDEBAR FILTERS (DATE RANGE + COUNTRY)
# -----------------------------------------------------------------------------
st.sidebar.header("Filters")

# Date range from orders table
min_date = df_orders["ORDER_TS"].min().date()
max_date = df_orders["ORDER_TS"].max().date()

date_range = st.sidebar.date_input(
    "Order date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# `st.date_input` can return a single date or a tuple
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

# Country filter
all_countries = sorted(df_users["COUNTRY"].dropna().unique().tolist())
selected_countries = st.sidebar.multiselect(
    "Countries",
    options=all_countries,
    default=all_countries,
)

st.sidebar.markdown("---")
view = st.sidebar.radio(
    "View",
    ["Users by Country", "Monthly Revenue & Units", "Channel Performance", "Funnel Conversion"],
)


# -----------------------------------------------------------------------------
# 4. APPLY FILTERS TO DATA
# -----------------------------------------------------------------------------
def apply_filters(
    df_users: pd.DataFrame,
    df_orders: pd.DataFrame,
    df_oi: pd.DataFrame,
    df_events: pd.DataFrame,
    start_date,
    end_date,
    selected_countries,
):
    # Users: filter only by country (users are "static")
    users_f = df_users.copy()
    if selected_countries and len(selected_countries) != len(all_countries):
        users_f = users_f[users_f["COUNTRY"].isin(selected_countries)]

    # Orders: filter by date, then by country via user
    orders_f = df_orders.copy()
    orders_f = orders_f[
        (orders_f["ORDER_TS"].dt.date >= start_date)
        & (orders_f["ORDER_TS"].dt.date <= end_date)
    ]

    orders_f = orders_f.merge(
        df_users[["USER_ID", "COUNTRY"]],
        on="USER_ID",
        how="left",
    )

    if selected_countries and len(selected_countries) != len(all_countries):
        orders_f = orders_f[orders_f["COUNTRY"].isin(selected_countries)]

    # Order items: filter by date, then tie to orders/country
    oi_f = df_oi.copy()
    oi_f = oi_f[
        (oi_f["ORDER_TS"].dt.date >= start_date)
        & (oi_f["ORDER_TS"].dt.date <= end_date)
    ]

    # Join to orders to attach COUNTRY
    oi_f = oi_f.merge(
        orders_f[["ORDER_ID", "USER_ID", "COUNTRY"]].drop_duplicates(),
        on="ORDER_ID",
        how="inner",
    )

    # Events: filter by date and country
    ev_f = df_events.copy()
    if not ev_f.empty:
        ev_f = ev_f[
            (ev_f["TS"].dt.date >= start_date)
            & (ev_f["TS"].dt.date <= end_date)
        ]
        ev_f = ev_f.merge(
            df_users[["USER_ID", "COUNTRY"]],
            on="USER_ID",
            how="left",
        )
        if selected_countries and len(selected_countries) != len(all_countries):
            ev_f = ev_f[ev_f["COUNTRY"].isin(selected_countries)]

    return users_f, orders_f, oi_f, ev_f


users_f, orders_f, oi_f, events_f = apply_filters(
    df_users, df_orders, df_oi, df_events, start_date, end_date, selected_countries
)


# -----------------------------------------------------------------------------
# 5. KPI METRICS (FILTERED)
# -----------------------------------------------------------------------------
# Active users = users who appear in filtered orders or events
user_ids_orders = set(orders_f["USER_ID"].dropna().unique().tolist())
user_ids_events = set(events_f["USER_ID"].dropna().unique().tolist()) if not events_f.empty else set()
active_user_ids = user_ids_orders.union(user_ids_events)

if active_user_ids:
    users_active = users_f[users_f["USER_ID"].isin(active_user_ids)]
else:
    users_active = users_f.iloc[0:0]  # empty with same columns

total_users_kpi = int(len(users_active))

# Paid orders (filtered)
orders_paid_f = orders_f[orders_f["STATUS"] == "PAID"]
paid_orders_kpi = int(orders_paid_f["ORDER_ID"].nunique())

# Opt-in rate among active users (if any)
if not users_active.empty and "MARKETING_OPT_IN" in users_active.columns:
    optin_rate_kpi = float(users_active["MARKETING_OPT_IN"].astype(float).mean())
else:
    optin_rate_kpi = float("nan")

# Sessions (events)
if not events_f.empty:
    sessions_kpi = int(events_f["SESSION_ID"].nunique())
else:
    sessions_kpi = 0

# KPI cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric("Total Users (active, filtered)", f"{total_users_kpi:,}")
kpi2.metric("Paid Orders (filtered)", f"{paid_orders_kpi:,}")
kpi3.metric(
    "Opt-in Rate (active users)",
    f"{optin_rate_kpi * 100:.1f}%" if not np.isnan(optin_rate_kpi) else "N/A",
)
kpi4.metric("Sessions (events, filtered)", f"{sessions_kpi:,}")

st.caption(
    "KPIs respect the selected date range and countries. "
    "Sessions are based on FACT_EVENT; active users are those with orders or events in the filter window."
)


# -----------------------------------------------------------------------------
# 6. PREP AGGREGATIONS FOR PLOTS (FILTERED)
# -----------------------------------------------------------------------------
# Country counts (active users only)
if not users_active.empty:
    country_counts = (
        users_active["COUNTRY"].value_counts().sort_values(ascending=False)
    )
else:
    country_counts = pd.Series(dtype=int)

# Daily & monthly revenue/units from paid orders/items
if not orders_paid_f.empty:
    daily_orders = (
        orders_paid_f
        .assign(DAY=orders_paid_f["ORDER_TS"].dt.date)
        .groupby("DAY", as_index=False)
        .agg(
            ORDERS=("ORDER_ID", "nunique"),
            REVENUE=("GROSS_REVENUE", "sum"),
        )
    )
else:
    daily_orders = pd.DataFrame(columns=["DAY", "ORDERS", "REVENUE"])

if not oi_f.empty:
    oi_paid_f = oi_f[oi_f["STATUS"] == "PAID"].copy()
    daily_units = (
        oi_paid_f
        .assign(DAY=oi_paid_f["ORDER_TS"].dt.date)
        .groupby("DAY", as_index=False)["QTY"]
        .sum()
        .rename(columns={"QTY": "UNITS"})
    )
else:
    daily_units = pd.DataFrame(columns=["DAY", "UNITS"])

df_daily = daily_orders.merge(daily_units, on="DAY", how="left").fillna({"UNITS": 0})

if not df_daily.empty:
    df_monthly = (
        df_daily
        .assign(
            MONTH=pd.to_datetime(df_daily["DAY"]).dt.to_period("M").dt.to_timestamp()
        )
        .groupby("MONTH", as_index=False)
        .agg(
            ORDERS=("ORDERS", "sum"),
            REVENUE=("REVENUE", "sum"),
            UNITS=("UNITS", "sum"),
        )
    )
else:
    df_monthly = pd.DataFrame(columns=["MONTH", "ORDERS", "REVENUE", "UNITS"])


# Channel performance (paid only)
if not oi_f.empty:
    channel_perf = (
        oi_paid_f
        .groupby("CHANNEL", as_index=False)
        .agg(
            ORDERS=("ORDER_ID", "nunique"),
            UNITS=("QTY", "sum"),
            REVENUE=("LINE_AMOUNT", "sum"),
        )
    )
    # Average order value per channel
    channel_perf["AVG_ORDER_VALUE"] = np.where(
        channel_perf["ORDERS"] > 0,
        channel_perf["REVENUE"] / channel_perf["ORDERS"],
        0.0,
    )
    # Sort by revenue for nicer plot
    channel_perf = channel_perf.sort_values("REVENUE", ascending=False)
else:
    channel_perf = pd.DataFrame(columns=["CHANNEL", "ORDERS", "UNITS", "REVENUE", "AVG_ORDER_VALUE"])


# Funnel metrics (from events_f)
monthly_funnel = pd.DataFrame()
total_sessions_funnel = 0
with_pv = with_atc = with_purchase = 0
pv_to_atc = atc_to_purchase = 0.0

if not events_f.empty and events_f["SESSION_ID"].notna().any():
    ev_simple = (
        events_f[events_f["SESSION_ID"].notna()]
        .groupby(["USER_ID", "SESSION_ID", "EVENT_TYPE"], as_index=False)
        .agg(TS=("TS", "min"))
    )

    wide = (
        ev_simple.pivot_table(
            index="SESSION_ID",
            columns="EVENT_TYPE",
            values="TS",
            aggfunc="min",
        )
        .reset_index()
        .fillna(pd.NaT)
    )

    wide["HAS_PV"] = wide.get("page_view").notna()
    wide["HAS_ATC"] = wide.get("add_to_cart").notna()
    wide["HAS_PUR"] = wide.get("purchase").notna()

    total_sessions_funnel = len(wide)
    with_pv = int(wide["HAS_PV"].sum())
    with_atc = int(wide["HAS_ATC"].sum())
    with_purchase = int(wide["HAS_PUR"].sum())

    pv_to_atc = with_atc / with_pv if with_pv else 0.0
    atc_to_purchase = with_purchase / with_atc if with_atc else 0.0

    # Daily funnel aggregation
    ev_daily = events_f[events_f["SESSION_ID"].notna()].copy()
    ev_daily = ev_daily.assign(DAY=ev_daily["TS"].dt.date)

    sess_daily = (
        ev_daily.groupby("SESSION_ID")
        .agg(
            DAY=("DAY", "min"),
            EVENT_SET=("EVENT_TYPE", lambda s: set(s)),
        )
        .reset_index()
    )

    sess_daily["HAS_PV"] = sess_daily["EVENT_SET"].apply(lambda s: "page_view" in s)
    sess_daily["HAS_ATC"] = sess_daily["EVENT_SET"].apply(lambda s: "add_to_cart" in s)
    sess_daily["HAS_PURCHASE"] = sess_daily["EVENT_SET"].apply(lambda s: "purchase" in s)

    daily_funnel = (
        sess_daily
        .groupby("DAY")
        .agg(
            SESSIONS=("SESSION_ID", "nunique"),
            PV_SESSIONS=("HAS_PV", "sum"),
            ATC_SESSIONS=("HAS_ATC", "sum"),
            PURCHASE_SESSIONS=("HAS_PURCHASE", "sum"),
            PV_TO_ATC_SESSIONS=("HAS_PV", lambda x: ((x) & (sess_daily.loc[x.index, "HAS_ATC"])).sum()),
            ATC_TO_PURCHASE_SESSIONS=("HAS_ATC", lambda x: ((x) & (sess_daily.loc[x.index, "HAS_PURCHASE"])).sum()),
        )
        .reset_index()
    )

    monthly_funnel_raw = (
        daily_funnel
        .assign(
            MONTH=pd.to_datetime(daily_funnel["DAY"]).dt.to_period("M").dt.to_timestamp()
        )
        .groupby("MONTH", as_index=False)
        .agg(
            SESSIONS=("SESSIONS", "sum"),
            PV_SESSIONS=("PV_SESSIONS", "sum"),
            ATC_SESSIONS=("ATC_SESSIONS", "sum"),
            PV_TO_ATC_SESSIONS=("PV_TO_ATC_SESSIONS", "sum"),
            ATC_TO_PURCHASE_SESSIONS=("ATC_TO_PURCHASE_SESSIONS", "sum"),
        )
    )

    monthly_funnel = monthly_funnel_raw.copy()
    monthly_funnel["PV_TO_ATC_RATE"] = np.where(
        monthly_funnel["PV_SESSIONS"] > 0,
        monthly_funnel["PV_TO_ATC_SESSIONS"] / monthly_funnel["PV_SESSIONS"] * 100,
        0.0,
    )

    monthly_funnel["ATC_TO_PURCHASE_RATE"] = np.where(
        monthly_funnel["ATC_SESSIONS"] > 0,
        monthly_funnel["ATC_TO_PURCHASE_SESSIONS"] / monthly_funnel["ATC_SESSIONS"] * 100,
        0.0,
    )


# -----------------------------------------------------------------------------
# 7. PLOTS – ONE PER VIEW (PLOTLY WITH HOVER)
# -----------------------------------------------------------------------------
st.markdown("---")

if view == "Users by Country":
    st.subheader("Users by Country (Active, Filtered)")

    if country_counts.empty:
        st.info("No active users for the current filters.")
    else:
        fig = px.bar(
            x=country_counts.index,
            y=country_counts.values,
            labels={"x": "Country", "y": "Number of Users"},
            title="Users by Country",
            hover_name=country_counts.index,
        )

        fig.update_layout(
            template="plotly_white",
            margin=dict(l=40, r=20, t=60, b=40),
        )

        st.plotly_chart(fig, use_container_width=True)


elif view == "Monthly Revenue & Units":
    st.subheader("Monthly Revenue & Units (Paid Orders, Filtered)")

    if df_monthly.empty:
        st.info("No monthly data available for the current filters.")
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Revenue
        fig.add_trace(
            go.Scatter(
                x=df_monthly["MONTH"],
                y=df_monthly["REVENUE"],
                mode="lines+markers",
                name="Revenue",
                hovertemplate="Month: %{x|%Y-%m}<br>Revenue: %{y:,.2f}<extra></extra>",
            ),
            secondary_y=False,
        )

        # Units
        fig.add_trace(
            go.Scatter(
                x=df_monthly["MONTH"],
                y=df_monthly["UNITS"],
                mode="lines+markers",
                name="Units Sold",
                line=dict(dash="dash"),
                hovertemplate="Month: %{x|%Y-%m}<br>Units: %{y:,.0f}<extra></extra>",
            ),
            secondary_y=True,
        )

        fig.update_xaxes(title_text="Month")
        fig.update_yaxes(title_text="Revenue", secondary_y=False)
        fig.update_yaxes(title_text="Units Sold", secondary_y=True)

        fig.update_layout(
            template="plotly_white",
            title_text="Monthly Revenue & Units (Paid Orders, Filtered)",
            hovermode="x unified",
            margin=dict(l=40, r=20, t=60, b=40),
        )

        st.plotly_chart(fig, use_container_width=True)


elif view == "Channel Performance":
    st.subheader("Channel Performance (Paid Orders, Filtered)")

    if channel_perf.empty:
        st.info("No channel data available for the current filters.")
    else:
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=channel_perf["CHANNEL"],
                y=channel_perf["REVENUE"],
                name="Revenue (Paid Orders)",
                hovertemplate=(
                    "Channel: %{x}<br>"
                    "Revenue: %{y:,.2f}<br>"
                    "Orders: %{customdata[0]}<br>"
                    "Units: %{customdata[1]:,.0f}<br>"
                    "AOV (Revenue/Order): %{customdata[2]:,.2f}<extra></extra>"
                ),
                customdata=np.stack(
                    [
                        channel_perf["ORDERS"],
                        channel_perf["UNITS"],
                        channel_perf["AVG_ORDER_VALUE"],
                    ],
                    axis=-1,
                ),
            )
        )

        fig.update_layout(
            template="plotly_white",
            title="Channel Revenue & Volume (Paid Orders, Filtered)",
            xaxis_title="Channel",
            yaxis_title="Revenue",
            margin=dict(l=40, r=20, t=60, b=40),
        )

        st.plotly_chart(fig, use_container_width=True)


elif view == "Funnel Conversion":
    st.subheader("Monthly Funnel Conversion Rates (Filtered Events)")

    if monthly_funnel.empty or total_sessions_funnel == 0:
        st.info("No funnel data available for the current filters.")
    else:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=monthly_funnel["MONTH"],
                y=monthly_funnel["PV_TO_ATC_RATE"],
                mode="lines+markers",
                name="PV → ATC (%)",
                hovertemplate="Month: %{x|%Y-%m}<br>PV → ATC: %{y:.1f}%<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=monthly_funnel["MONTH"],
                y=monthly_funnel["ATC_TO_PURCHASE_RATE"],
                mode="lines+markers",
                name="ATC → Purchase (%)",
                hovertemplate="Month: %{x|%Y-%m}<br>ATC → Purchase: %{y:.1f}%<extra></extra>",
            )
        )

        summary_text = (
            f"Sessions: {total_sessions_funnel}<br>"
            f"PV: {with_pv} · ATC: {with_atc} · PUR: {with_purchase}<br>"
            f"PV→ATC (overall): {pv_to_atc * 100:.1f}%<br>"
            f"ATC→PUR (overall): {atc_to_purchase * 100:.1f}%"
        )

        fig.update_layout(
            template="plotly_white",
            title="Monthly Funnel Conversion Rates (Filtered Events)",
            xaxis_title="Month",
            yaxis_title="Conversion Rate (%)",
            hovermode="x unified",
            margin=dict(l=40, r=20, t=60, b=40),
            annotations=[
                dict(
                    xref="paper",
                    yref="paper",
                    x=0.01,
                    y=0.99,
                    text=summary_text,
                    showarrow=False,
                    align="left",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1,
                    bgcolor="rgba(255,255,255,0.85)",
                    font=dict(size=10),
                )
            ],
        )

        st.plotly_chart(fig, use_container_width=True)
