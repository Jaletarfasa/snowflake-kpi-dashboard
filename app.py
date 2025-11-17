import snowflake.connector
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# -----------------------------------------------------------------------------
# 0. PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Snowflake KPI Dashboard",
    layout="wide",
)

st.title("Snowflake KPI Dashboard")
st.caption("Customers · Revenue · Channels · Funnel (Snowflake, Filtered)")

st.markdown(
    "<div style='font-size:12px; margin-bottom:10px;'>Built by "
    "<b>Solomon Tarfasa</b> · Streamlit + Snowflake</div>",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 1. SNOWFLAKE CONNECTION + HELPERS  (uses Streamlit secrets)
# -----------------------------------------------------------------------------

def get_connection():
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
# 2. LOAD BASE DATA (CACHED)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=True)
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

    # ensure datetimes
    df_orders["ORDER_TS"] = pd.to_datetime(df_orders["ORDER_TS"])
    df_oi["ORDER_TS"] = pd.to_datetime(df_oi["ORDER_TS"])
    df_events["TS"] = pd.to_datetime(df_events["TS"])

    return df_users, df_orders, df_oi, df_events


with st.spinner("Loading data from Snowflake..."):
    df_users, df_orders, df_oi, df_events = load_data()

# -----------------------------------------------------------------------------
# 3. FILTERS (SIDEBAR)
# -----------------------------------------------------------------------------
st.sidebar.header("Filters")

# Date range filter from orders
min_date = df_orders["ORDER_TS"].dt.date.min()
max_date = df_orders["ORDER_TS"].dt.date.max()

date_range = st.sidebar.date_input(
    "Order date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, tuple):
    start_date, end_date = date_range
else:
    # if user selects a single date
    start_date = date_range
    end_date = date_range

# Country filter
all_countries = sorted(df_users["COUNTRY"].dropna().unique())
selected_countries = st.sidebar.multiselect(
    "Countries",
    all_countries,
    default=all_countries,
)

# Menu for which chart to show
view = st.sidebar.radio(
    "View",
    [
        "Users by Country",
        "Monthly Revenue & Units",
        "Channel Performance",
        "Funnel Conversion",
    ],
)

# -----------------------------------------------------------------------------
# 4. APPLY FILTERS TO DATA
# -----------------------------------------------------------------------------
# Filter users by country
if selected_countries:
    df_users_f = df_users[df_users["COUNTRY"].isin(selected_countries)].copy()
else:
    df_users_f = df_users.copy()

# Filter orders (paid + date + users)
mask_orders = (
    (df_orders["STATUS"] == "PAID")
    & (df_orders["ORDER_TS"].dt.date >= start_date)
    & (df_orders["ORDER_TS"].dt.date <= end_date)
)
df_orders_f = df_orders[mask_orders].copy()

if not df_users_f.empty:
    df_orders_f = df_orders_f[df_orders_f["USER_ID"].isin(df_users_f["USER_ID"])]

# Filter order items (paid + matching orders)
df_oi_f = df_oi[
    (df_oi["STATUS"] == "PAID")
    & (df_oi["ORDER_TS"].dt.date >= start_date)
    & (df_oi["ORDER_TS"].dt.date <= end_date)
].copy()

if not df_orders_f.empty:
    df_oi_f = df_oi_f[df_oi_f["ORDER_ID"].isin(df_orders_f["ORDER_ID"])]

# Filter events by date + users
df_events_f = df_events[
    (df_events["TS"].dt.date >= start_date)
    & (df_events["TS"].dt.date <= end_date)
].copy()

if not df_users_f.empty:
    df_events_f = df_events_f[df_events_f["USER_ID"].isin(df_users_f["USER_ID"])]

# -----------------------------------------------------------------------------
# 5. KPIs (FILTER AWARE)
# -----------------------------------------------------------------------------
total_users = df_users_f["USER_ID"].nunique()
paid_orders = df_orders_f["ORDER_ID"].nunique()
optin_rate = df_users_f["MARKETING_OPT_IN"].mean() if total_users > 0 else 0.0
sessions = df_events_f["SESSION_ID"].nunique()

kpi_cols = st.columns(4)
kpi_cols[0].metric("Total Users (active, filtered)", f"{total_users:,}")
kpi_cols[1].metric("Paid Orders (filtered)", f"{paid_orders:,}")
kpi_cols[2].metric("Opt-in Rate (filtered)", f"{optin_rate*100:0.1f}%")
kpi_cols[3].metric("Sessions (events, filtered)", f"{sessions:,}")

st.caption(
    "KPIs respect the selected date range and country filters. "
    "Sessions are based on FACT_EVENT within the filter window."
)

# -----------------------------------------------------------------------------
# 6. AGGREGATIONS FOR CHARTS (FILTERED)
# -----------------------------------------------------------------------------
# Monthly revenue & units
if not df_orders_f.empty:
    daily_orders = (
        df_orders_f.assign(DAY=df_orders_f["ORDER_TS"].dt.date)
        .groupby("DAY", as_index=False)
        .agg(
            ORDERS=("ORDER_ID", "nunique"),
            REVENUE=("GROSS_REVENUE", "sum"),
        )
    )
else:
    daily_orders = pd.DataFrame(columns=["DAY", "ORDERS", "REVENUE"])

if not df_oi_f.empty:
    daily_units = (
        df_oi_f.assign(DAY=df_oi_f["ORDER_TS"].dt.date)
        .groupby("DAY", as_index=False)["QTY"]
        .sum()
        .rename(columns={"QTY": "UNITS"})
    )
else:
    daily_units = pd.DataFrame(columns=["DAY", "UNITS"])

df_daily = daily_orders.merge(daily_units, on="DAY", how="left").fillna({"UNITS": 0})

if not df_daily.empty:
    df_monthly = (
        df_daily.assign(
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

# Country counts (filtered users)
country_counts = (
    df_users_f["COUNTRY"].value_counts().rename_axis("COUNTRY").reset_index(name="USERS")
)

# Channel performance
if not df_oi_f.empty:
    channel_perf = (
        df_oi_f.groupby("CHANNEL", as_index=False)
        .agg(
            ORDERS=("ORDER_ID", "nunique"),
            UNITS=("QTY", "sum"),
            REVENUE=("LINE_AMOUNT", "sum"),
        )
        .assign(AVG_ORDER_VALUE=lambda d: d["REVENUE"] / d["ORDERS"])
    )
else:
    channel_perf = pd.DataFrame(
        columns=["CHANNEL", "ORDERS", "UNITS", "REVENUE", "AVG_ORDER_VALUE"]
    )

# Funnel: PV -> ATC -> Purchase
if not df_events_f.empty:
    ev_sessions = (
        df_events_f[df_events_f["SESSION_ID"].notna()]
        .groupby(["SESSION_ID", "EVENT_TYPE"], as_index=False)
        .agg(TS=("TS", "min"))
    )

    wide = (
        ev_sessions.pivot_table(
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

    total_sessions_f = len(wide)
    with_pv = wide["HAS_PV"].sum()
    with_atc = wide["HAS_ATC"].sum()
    with_purchase = wide["HAS_PUR"].sum()

    pv_to_atc = with_atc / with_pv if with_pv else 0.0
    atc_to_purchase = with_purchase / with_atc if with_atc else 0.0

    # daily funnel (for rates over time)
    ev_daily = df_events_f[df_events_f["SESSION_ID"].notna()].assign(
        DAY=df_events_f["TS"].dt.date
    )
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
    sess_daily["HAS_PURCHASE"] = sess_daily["EVENT_SET"].apply(
        lambda s: "purchase" in s
    )

    daily_funnel = (
        sess_daily.groupby("DAY")
        .agg(
            SESSIONS=("SESSION_ID", "nunique"),
            PV_SESSIONS=("HAS_PV", "sum"),
            ATC_SESSIONS=("HAS_ATC", "sum"),
            PURCHASE_SESSIONS=("HAS_PURCHASE", "sum"),
        )
        .reset_index()
    )

    # monthly funnel rates
    if not daily_funnel.empty:
        monthly_funnel = (
            daily_funnel.assign(
                MONTH=pd.to_datetime(daily_funnel["DAY"])
                .dt.to_period("M")
                .dt.to_timestamp()
            )
            .groupby("MONTH", as_index=False)
            .agg(
                SESSIONS=("SESSIONS", "sum"),
                PV_SESSIONS=("PV_SESSIONS", "sum"),
                ATC_SESSIONS=("ATC_SESSIONS", "sum"),
                PURCHASE_SESSIONS=("PURCHASE_SESSIONS", "sum"),
            )
        )

        monthly_funnel["PV_TO_ATC_RATE"] = np.where(
            monthly_funnel["PV_SESSIONS"] > 0,
            monthly_funnel["ATC_SESSIONS"] / monthly_funnel["PV_SESSIONS"] * 100,
            0.0,
        )
        monthly_funnel["ATC_TO_PURCHASE_RATE"] = np.where(
            monthly_funnel["ATC_SESSIONS"] > 0,
            monthly_funnel["PURCHASE_SESSIONS"] / monthly_funnel["ATC_SESSIONS"] * 100,
            0.0,
        )
    else:
        monthly_funnel = pd.DataFrame(
            columns=[
                "MONTH",
                "SESSIONS",
                "PV_SESSIONS",
                "ATC_SESSIONS",
                "PURCHASE_SESSIONS",
                "PV_TO_ATC_RATE",
                "ATC_TO_PURCHASE_RATE",
            ]
        )
else:
    total_sessions_f = with_pv = with_atc = with_purchase = 0
    pv_to_atc = atc_to_purchase = 0.0
    monthly_funnel = pd.DataFrame(
        columns=[
            "MONTH",
            "SESSIONS",
            "PV_SESSIONS",
            "ATC_SESSIONS",
            "PURCHASE_SESSIONS",
            "PV_TO_ATC_RATE",
            "ATC_TO_PURCHASE_RATE",
        ]
    )

# -----------------------------------------------------------------------------
# 7. MAIN VIEW (ONE PLOT PER MENU ITEM)
# -----------------------------------------------------------------------------
st.markdown("---")

if view == "Users by Country":
    st.subheader("Users by Country (Filtered)")

    if not country_counts.empty:
        fig_country = px.bar(
            country_counts,
            x="COUNTRY",
            y="USERS",
            title="Users by Country",
            labels={"USERS": "Number of Users"},
        )
        st.plotly_chart(fig_country, use_container_width=True)
    else:
        st.info("No users for the selected filters.")

    # download
    csv_ctry = country_counts.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download country breakdown (CSV)",
        csv_ctry,
        "users_by_country.csv",
        "text/csv",
    )

elif view == "Monthly Revenue & Units":
    st.subheader("Monthly Revenue & Units (Paid Orders, Filtered)")

    if not df_monthly.empty:
        df_melt = df_monthly.melt(
            id_vars="MONTH",
            value_vars=["REVENUE", "UNITS"],
            var_name="Metric",
            value_name="Value",
        )
        fig_month = px.line(
            df_melt,
            x="MONTH",
            y="Value",
            color="Metric",
            markers=True,
            title="Monthly Revenue & Units (Paid Orders, Filtered)",
        )
        st.plotly_chart(fig_month, use_container_width=True)
    else:
        st.info("No paid orders for the selected filters.")

    csv_month = df_monthly.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download monthly revenue & units (CSV)",
        csv_month,
        "monthly_revenue_units.csv",
        "text/csv",
    )

elif view == "Channel Performance":
    st.subheader("Channel Performance (Paid Orders, Filtered)")

    if not channel_perf.empty:
        fig_channel = px.bar(
            channel_perf,
            x="CHANNEL",
            y="REVENUE",
            color="CHANNEL",
            title="Revenue by Channel (Paid Orders, Filtered)",
            hover_data=["ORDERS", "UNITS", "AVG_ORDER_VALUE"],
        )
        st.plotly_chart(fig_channel, use_container_width=True)
    else:
        st.info("No channel data for the selected filters.")

    csv_ch = channel_perf.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download channel performance (CSV)",
        csv_ch,
        "channel_performance.csv",
        "text/csv",
    )

elif view == "Funnel Conversion":
    st.subheader("Funnel Conversion (PV → ATC → Purchase)")

    col_l, col_r = st.columns([2, 1])

    with col_l:
        if not monthly_funnel.empty:
            fig_funnel = px.line(
                monthly_funnel,
                x="MONTH",
                y=["PV_TO_ATC_RATE", "ATC_TO_PURCHASE_RATE"],
                markers=True,
                labels={
                    "value": "Conversion Rate (%)",
                    "variable": "Stage",
                },
                title="Monthly Funnel Conversion Rates (Filtered)",
            )
            st.plotly_chart(fig_funnel, use_container_width=True)
        else:
            st.info("No funnel events for the selected filters.")

    with col_r:
        st.markdown("**Overall Funnel (Filtered)**")
        st.write(f"Sessions: **{total_sessions_f:,}**")
        st.write(f"Sessions with page view: **{with_pv:,}**")
        st.write(f"Sessions with add-to-cart: **{with_atc:,}**")
        st.write(f"Sessions with purchase: **{with_purchase:,}**")
        st.write(f"PV → ATC: **{pv_to_atc*100:0.1f}%**")
        st.write(f"ATC → Purchase: **{atc_to_purchase*100:0.1f}%**")

        csv_f = monthly_funnel.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download monthly funnel stats (CSV)",
            csv_f,
            "monthly_funnel.csv",
            "text/csv",
        )
