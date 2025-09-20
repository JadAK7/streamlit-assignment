# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO

st.set_page_config(page_title="Trade & Debt Explorer", page_icon="📈", layout="wide")

# ---------- Sidebar: Data Ingestion ----------
st.sidebar.header("📦 Data")
st.sidebar.write(
    "Upload the CSV used in your Plotly assignment. "
    "It must include at least these columns: "
    "`refArea`, `Indicator Code`, `refPeriod`, `Value`."
)

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
df = None

@st.cache_data
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return df

if uploaded:
    df = load_csv(uploaded)
else:
    st.info(
        "No CSV uploaded yet. You can still read through the page or upload the dataset to see the charts. "
        "If you kept it locally, just drag-drop it here."
    )

# ---------- Mapping dictionary (from your code) ----------
indicator_mapping = {
  'BM.GSR.TOTL.CD':'Imports of goods, services and primary income (BoP, current US$)',
  'BN.CAB.XOKA.CD':'Current account balance (BoP, current US$)',
  'BX.GRT.EXTA.CD.WD':'Grants, excluding technical cooperation (BoP, current US$)',
  'BX.GRT.TECH.CD.WD':'Technical cooperation grants (BoP, current US$)',
  'BX.GSR.TOTL.CD':'Exports of goods, services and primary income (BoP, current US$)',
  'BX.KLT.DINV.CD.WD':'Foreign direct investment, net inflows (BoP, current US$)',
  'BX.PEF.TOTL.CD.WD':'Portfolio equity, net inflows (BoP, current US$)',
  'BX.TRF.PWKR.CD.DT':'Personal remittances, received (current US$)',

  # External Debt Statistics
  'DT.DOD.DECT.CD':'Total external debt stocks (current US$)',
  'DT.DOD.DECT.GN.ZS':'External debt stocks (% of GNI)',
  'DT.DOD.DIMF.CD':'Use of IMF credit (current US$)',
  'DT.DOD.DLXF.CD':'Long-term debt from official creditors (current US$)',
  'DT.DOD.DPNG.CD':'Long-term debt from private nonguaranteed creditors (current US$)',
  'DT.DOD.DPPG.CD':'Long-term debt from public and publicly guaranteed creditors (current US$)',
  'DT.DOD.DSTC.CD':'Short-term external debt stocks (current US$)',
  'DT.DOD.DSTC.IR.ZS':'Short-term debt (% of total reserves plus gold)',
  'DT.DOD.DSTC.XP.ZS':'Short-term debt (% of exports of goods, services and primary income)',
  'DT.DOD.DSTC.ZS':'Short-term debt (% of total external debt)',
  'DT.DOD.MIBR.CD':'IBRD loans and IDA credits (current US$)',
  'DT.DOD.MIDA.CD':'IDA credits (current US$)',
  'DT.DOD.MWBG.CD':'World Bank loans (IBRD and IDA) (current US$)',
  'DT.DOD.PVLX.CD':'Present value of external debt (current US$)',
  'DT.DOD.PVLX.EX.ZS':'Present value of external debt (% of exports of goods, services and primary income)',

  # Net Financial Flows
  'DT.NFL.BLAT.CD':'Net flows on external debt, bilateral (NFL, current US$)',
  'DT.NFL.BOND.CD':'Net flows on external debt, bonds (NFL, current US$)',
  'DT.NFL.DPNG.CD':'Net flows on external debt, private nonguaranteed (PNG) (NFL, current US$)',
  'DT.NFL.IMFN.CD':'Net flows from IMF nonconcessional (NFL, current US$)',
  'DT.NFL.MIBR.CD':'Net flows from IBRD (NFL, current US$)',
  'DT.NFL.MIDA.CD':'Net flows from IDA (NFL, current US$)',
  'DT.NFL.MLAT.CD':'Net flows on external debt, multilateral (NFL, current US$)',
  'DT.NFL.MOTH.CD':'Net flows on external debt, other (NFL, current US$)',
  'DT.NFL.NIFC.CD':'Net flows on external debt, short-term (NFL, current US$)',
  'DT.NFL.OFFT.CD':'Net official flows from UN agencies (NFL, current US$)',
  'DT.NFL.PBND.CD':'Net flows on external debt, public and publicly guaranteed bonds (NFL, current US$)',
  'DT.NFL.PCBK.CD':'Net flows on external debt, commercial banks (NFL, current US$)',
  'DT.NFL.PCBO.CD':'Net flows on external debt, commercial bonds (NFL, current US$)',
  'DT.NFL.PNGB.CD':'Net flows on external debt, private nonguaranteed bonds (NFL, current US$)',
  'DT.NFL.PNGC.CD':'Net flows on external debt, private nonguaranteed commercial banks (NFL, current US$)',
  'DT.NFL.PROP.CD':'Net flows on external debt, other private creditors (NFL, current US$)',
  'DT.NFL.PRVT.CD':'Net flows on external debt, private creditors (NFL, current US$)',

  # Official Development Assistance (ODA)
  'DT.ODA.ODAT.CD':'Net ODA received (current US$)',
  'DT.ODA.ODAT.GN.ZS':'Net ODA received (% of GNI)',
  'DT.ODA.ODAT.PC.ZS':'Net ODA received per capita (current US$)',

  # Total Debt Service
  'DT.TDS.DECT.CD':'Total debt service on external debt (TDS, current US$)',
  'DT.TDS.DECT.EX.ZS':'Total debt service (% of exports of goods, services and primary income)',
  'DT.TDS.DECT.GN.ZS':'Total debt service (% of GNI)',
  'DT.TDS.DIMF.CD':'Total debt service on IMF only (TDS, current US$)',
  'DT.TDS.DPPF.XP.ZS':'Debt service on external debt, PPG and PNG (TDS, % of exports)',
  'DT.TDS.DPPG.CD':'Total debt service on external debt, public and publicly guaranteed (PPG) (TDS, current US$)',
  'DT.TDS.DPPG.GN.ZS':'Debt service on external debt, public and publicly guaranteed (PPG) (TDS, % of GNI)',
  'DT.TDS.DPPG.XP.ZS':'Debt service on external debt, public and publicly guaranteed (PPG) (TDS, % of exports)',
  'DT.TDS.MLAT.CD':'Total debt service on external debt, multilateral (TDS, current US$)',
  'DT.TDS.MLAT.PG.ZS':'Debt service on external debt, multilateral (TDS, % of public and publicly guaranteed debt service)',

  # Foreign Reserves & National Income
  'FI.RES.TOTL.CD':'Total reserves (includes gold, current US$)',
  'FI.RES.TOTL.DT.ZS':'Total reserves (% of total external debt)',
  'FI.RES.TOTL.MO':'Total reserves in months of imports',
  'NY.GNP.MKTP.CD':'GNP (current US$)'
}

def prep_dataframe(df):
    df = df.copy()
    # Extract country from refArea tail segment
    if "Country" not in df.columns:
        df["Country"] = df["refArea"].str.extract(r"/([^/]+)$")[0].fillna(df["refArea"])
    # Clean indicator names
    df["Indicator_Name"] = df["Indicator Code"].map(indicator_mapping).fillna(df["Indicator Code"])
    # Year as int if possible
    try:
        df["refPeriod"] = df["refPeriod"].astype(int)
    except Exception:
        pass
    return df

if df is not None:
    df = prep_dataframe(df)

# ---------- Sidebar: Controls ----------
st.sidebar.header("⚙️ Controls")

if df is not None:
    countries = sorted(df["Country"].dropna().unique().tolist())
    default_country = ["Lebanon"] if "Lebanon" in countries else (countries[:1] if countries else [])
    selected_countries = st.sidebar.multiselect("Country", countries, default=default_country)

    # Year range slider
    yr_min, yr_max = int(df["refPeriod"].min()), int(df["refPeriod"].max())
    year_range = st.sidebar.slider("Year range", min_value=yr_min, max_value=yr_max, value=(yr_min, yr_max), step=1)

    # Smoothing + scale
    smooth = st.sidebar.slider("Smoothing (moving average, years)", 1, 7, 1, 1)
    use_log = st.sidebar.checkbox("Log scale (y axis)", value=False)

    # Interaction presets (both charts respect filters, but this switches the 2nd chart’s focus)
    second_chart_mode = st.sidebar.radio(
        "Second visualization mode",
        ["Debt vs GNP (+ Ratio)", "Top Indicators Heatmap"],
        index=0
    )

# ---------- Page Header ----------
st.title("📊 Trade & Debt Explorer")
st.write(
    "Explore two related themes from your previous Plotly assignment: "
    "**Trade (Imports vs Exports)** and **External Debt vs National Income (GNP)**. "
    "Use the controls to filter by country, adjust the year window, toggle log scale, "
    "and apply smoothing. Insights update live."
)

if df is None:
    st.stop()

# ---------- Helper: filter + smooth ----------
def filter_df(base: pd.DataFrame) -> pd.DataFrame:
    d = base.copy()
    if selected_countries:
        d = d[d["Country"].isin(selected_countries)]
    d = d[(d["refPeriod"] >= year_range[0]) & (d["refPeriod"] <= year_range[1])]
    return d

def moving_average(d: pd.DataFrame, by_cols=("Country", "Indicator Code")) -> pd.DataFrame:
    if smooth <= 1:
        return d
    d = d.sort_values(["Country", "Indicator Code", "refPeriod"]).copy()
    d["Value"] = d.groupby(list(by_cols))["Value"].transform(lambda s: s.rolling(smooth, min_periods=1).mean())
    return d

filtered = filter_df(df)

# ==============================
# Visualization 1: Imports vs Exports
# ==============================
trade_codes = ["BM.GSR.TOTL.CD", "BX.GSR.TOTL.CD"]  # Imports / Exports
trade = filtered[filtered["Indicator Code"].isin(trade_codes)].copy()
trade = moving_average(trade)

if trade.empty:
    st.warning("No trade data found for the current filters.")
else:
    subtitle = ", ".join(selected_countries) if selected_countries else "All countries"
    fig_trade = px.area(
        trade,
        x="refPeriod",
        y="Value",
        color="Indicator_Name",
        facet_row="Country" if len(selected_countries) > 1 else None,
        title=f"Imports vs Exports Over Time — {subtitle}",
        labels={"refPeriod": "Year", "Value": "Value (current US$)", "Indicator_Name": "Series"},
        template="plotly_white"
    )
    if use_log:
        fig_trade.update_yaxes(type="log", matches=None)
    fig_trade.update_layout(hovermode="x unified")
    st.plotly_chart(fig_trade, use_container_width=True)

    # Quick insight block
    with st.expander("💡 Quick insights (Trade)"):
        # Compute latest year and balance
        latest_year = int(trade["refPeriod"].max())
        t_latest = trade[trade["refPeriod"] == latest_year]
        # Aggregate across chosen countries
        pivot = t_latest.pivot_table(index="Indicator_Name", values="Value", aggfunc="sum")
        imports = float(pivot.get("Imports of goods, services and primary income (BoP, current US$)", pd.Series([np.nan])).iloc[0]) if "Imports of goods, services and primary income (BoP, current US$)" in pivot.index else np.nan
        exports = float(pivot.get("Exports of goods, services and primary income (BoP, current US$)", pd.Series([np.nan])).iloc[0]) if "Exports of goods, services and primary income (BoP, current US$)" in pivot.index else np.nan
        if not np.isnan(imports) and not np.isnan(exports):
            balance = exports - imports
            st.markdown(
                f"- **Latest year ({latest_year}) trade balance:** "
                f"{balance:,.0f} US$ ({'surplus' if balance>=0 else 'deficit'})."
            )
        # CAGR over selected range if possible
        try:
            start_year = int(trade["refPeriod"].min())
            end_year = int(trade["refPeriod"].max())
            years = max(1, end_year - start_year)
            def cagr(series):
                s = series.dropna()
                if s.empty: return np.nan
                first, last = s.iloc[0], s.iloc[-1]
                if first <= 0 or last <= 0: return np.nan
                return (last/first)**(1/years) - 1
            cagr_df = trade.pivot_table(index="refPeriod", columns="Indicator_Name", values="Value").sort_index()
            imp_cagr = cagr(cagr_df.get("Imports of goods, services and primary income (BoP, current US$)", pd.Series(dtype=float)))
            exp_cagr = cagr(cagr_df.get("Exports of goods, services and primary income (BoP, current US$)", pd.Series(dtype=float)))
            if not np.isnan(imp_cagr) and not np.isnan(exp_cagr):
                st.markdown(
                    f"- **CAGR ({start_year}–{end_year}):** Imports {imp_cagr*100:.1f}%, Exports {exp_cagr*100:.1f}%."
                )
        except Exception:
            pass

# ==============================
# Visualization 2: Debt vs GNP (+ Ratio) OR Heatmap
# ==============================
col1, col2 = st.columns([3, 2])

with col1:
    if second_chart_mode == "Debt vs GNP (+ Ratio)":
        needed = ["DT.DOD.DECT.CD", "NY.GNP.MKTP.CD"]
        d2 = filtered[filtered["Indicator Code"].isin(needed)].copy()
        d2 = moving_average(d2)
        if d2.empty:
            st.warning("No debt/GNP data found for the current filters.")
        else:
            # Aggregate across countries (sum; switch to mean if needed)
            agg = d2.groupby(["refPeriod", "Indicator_Name"], as_index=False)["Value"].sum()

            # Build dual-axis + ratio
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            debt_series = agg[agg["Indicator_Name"] == "Total external debt stocks (current US$)"]
            gnp_series  = agg[agg["Indicator_Name"] == "GNP (current US$)"]

            fig.add_trace(
                go.Scatter(x=debt_series["refPeriod"], y=debt_series["Value"], name="External Debt", mode="lines"),
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=gnp_series["refPeriod"], y=gnp_series["Value"], name="GNP", mode="lines"),
                secondary_y=False
            )
            # Ratio (Debt/GNP)
            ratio = pd.merge(
                debt_series[["refPeriod", "Value"]].rename(columns={"Value": "Debt"}),
                gnp_series[["refPeriod", "Value"]].rename(columns={"Value": "GNP"}),
                on="refPeriod", how="inner"
            )
            ratio["Debt_to_GNP"] = ratio["Debt"] / ratio["GNP"]
            fig.add_trace(
                go.Bar(x=ratio["refPeriod"], y=ratio["Debt_to_GNP"], name="Debt/GNP Ratio"),
                secondary_y=True
            )

            fig.update_layout(
                title=f"External Debt vs GNP and Debt/GNP Ratio — {', '.join(selected_countries) if selected_countries else 'All countries'}",
                template="plotly_white",
                barmode="overlay",
                hovermode="x unified"
            )
            if use_log:
                fig.update_yaxes(type="log", secondary_y=False)
                # Keep ratio in linear space
            fig.update_yaxes(title_text="US$ (current)", secondary_y=False)
            fig.update_yaxes(title_text="Debt/GNP (×)", secondary_y=True, rangemode="tozero")
            st.plotly_chart(fig, use_container_width=True)

            # Insight
            if not ratio.empty:
                latest = ratio.iloc[-1]
                st.caption(f"Latest Debt/GNP ratio ({int(latest['refPeriod'])}): **{latest['Debt_to_GNP']:.2f}×**")

    else:
        # Heatmap of top-variance indicators in selection window
        d2 = filtered.copy()
        # Pick indicators with most variance across years (signal > noise)
        var_tbl = (
            d2.groupby("Indicator_Name")["Value"]
              .agg(lambda s: np.nanvar(s.values))
              .sort_values(ascending=False)
              .head(12)
        )
        d2 = d2[d2["Indicator_Name"].isin(var_tbl.index)]
        if d2.empty:
            st.warning("Not enough data to render the heatmap for the current filters.")
        else:
            pivot = d2.pivot_table(index="Indicator_Name", columns="refPeriod", values="Value", aggfunc="sum")
            # Signed log transform to compress range but preserve sign
            heat = np.sign(pivot) * np.log10(np.abs(pivot) + 1)
            fig2 = go.Figure(data=go.Heatmap(
                z=heat.values, x=heat.columns, y=heat.index, colorscale="RdYlBu_r", hoverongaps=False,
                colorbar=dict(title="signed log₁₀")
            ))
            fig2.update_layout(
                title=f"Top-Changing Indicators Heatmap — {', '.join(selected_countries) if selected_countries else 'All countries'}",
                xaxis_title="Year", yaxis_title="Indicator", template="plotly_white"
            )
            st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.markdown("### ℹ️ What you’re seeing & how to use it")
    st.write(
        """
- **Trade view** compares **Imports vs Exports** over time. Use *Year range* to focus on crises/recoveries and **Smoothing** to clarify trend vs noise.
- **Debt view** shows **External Debt vs GNP** with the **Debt/GNP ratio** for a quick sustainability signal.
- **Log scale** helps when a few extreme years dominate the y-axis.
- **Heatmap mode** surfaces which indicators changed the most in your selected window.
"""
    )
    # Small KPI cards (if data exists)
    try:
        kpi = filtered.copy()
        latest_year = int(kpi["refPeriod"].max())
        latest = kpi[kpi["refPeriod"] == latest_year]
        def get_val(code):
            t = latest[latest["Indicator Code"] == code]["Value"].sum()
            return np.nan if pd.isna(t) else float(t)
        imp = get_val("BM.GSR.TOTL.CD")
        exp = get_val("BX.GSR.TOTL.CD")
        debt = get_val("DT.DOD.DECT.CD")
        gnp  = get_val("NY.GNP.MKTP.CD")
        st.metric("Latest year", latest_year)
        if not np.isnan(imp) and not np.isnan(exp):
            st.metric("Trade balance (US$)", f"{(exp-imp):,.0f}")
        if not np.isnan(debt) and not np.isnan(gnp) and gnp != 0:
            st.metric("Debt/GNP (×)", f"{debt/gnp:.2f}")
    except Exception:
        pass

st.markdown("---")
st.subheader("📖 Context & Insights")
st.write(
    """
**Why these two visuals together?**  
Trade dynamics (exports vs imports) shape foreign currency flows, which in turn affect borrowing needs and **external debt**. By pairing them, users can see whether persistent trade deficits correlate with rising **Debt/GNP**.

**Questions to explore:**
1. When the trade balance improves, does the **Debt/GNP** ratio stabilize after a lag?
2. Are there structural breaks (policy, shocks) visible when you adjust the **year range**?
3. Does smoothing reveal persistent trends masked by volatility?
"""
)
