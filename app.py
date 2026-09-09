import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="E-Waste Analytics Dashboard | Topic #14",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# 2. FIGMA-STYLE DARK MODE CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
        font-family: 'Inter', sans-serif;
    }
    .figma-header {
        background: linear-gradient(90deg, #1F2937 0%, #111827 100%);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #374151;
        margin-bottom: 24px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }
    .team-pill {
        display: inline-block;
        background: #111827;
        border: 1px solid #374151;
        color: #D1D5DB;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        margin-right: 8px;
        margin-top: 6px;
    }
    div[data-testid="stMetric"] {
        background: #1F2937;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 16px 20px;
    }
    div[data-testid="stMetric"] label {
        color: #9CA3AF !important;
        font-size: 14px !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #60A5FA !important;
        font-size: 26px !important;
        font-weight: 700 !important;
    }
    .section-title {
        font-size: 22px;
        font-weight: 800;
        color: #F9FAFB;
        margin-top: 28px;
        margin-bottom: 10px;
        border-left: 5px solid #3B82F6;
        padding-left: 12px;
    }
    .chart-title {
        font-size: 17px;
        font-weight: 700;
        color: #F3F4F6;
        margin-bottom: 4px;
    }
    .insight-box {
        background-color: #1F2937;
        border-left: 4px solid #3B82F6;
        padding: 14px 18px;
        border-radius: 8px;
        margin-top: 10px;
        margin-bottom: 22px;
        font-size: 14px;
        color: #D1D5DB;
        line-height: 1.55;
    }
    .problem-box {
        background-color: #1F1315;
        border-left: 4px solid #EF4444;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 14px;
        font-size: 14px;
        color: #D1D5DB;
        line-height: 1.55;
    }
    .recommendation-box {
        background-color: #0F1A17;
        border-left: 4px solid #10B981;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 14px;
        font-size: 14px;
        color: #D1D5DB;
        line-height: 1.6;
    }
    .verdict-box {
        background: linear-gradient(135deg, #1E3A8A 0%, #111827 100%);
        border: 1px solid #3B82F6;
        padding: 20px 24px;
        border-radius: 14px;
        margin-top: 12px;
        margin-bottom: 24px;
        font-size: 15px;
        color: #E5E7EB;
        line-height: 1.7;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 3. LOAD DATASET
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("goal11_wastegdp.csv")
    df.columns = [c.strip() for c in df.columns]
    return df

df = load_data()

# Precompute correlations
r_pop_waste_lin   = df['pop'].corr(df['waste'])
r_pop_waste_log   = np.log(df['pop']).corr(np.log(df['waste']))
r_pop_percap      = df['pop'].corr(df['wastepercap'])
r_gdp_percap_lin  = df['gdp'].corr(df['wastepercap'])
r_gdp_percap_log  = np.log(df['gdp']).corr(np.log(df['wastepercap']))

# Wealth tier x Urban tier matrix
df['gdp_tier'] = pd.qcut(df['gdp'], 3, labels=['Low GDP', 'Middle GDP', 'High GDP'])
df['urban_tier'] = pd.cut(df['sharepopurban'], bins=[0, 50, 80, 100],
                           labels=['Low Urban (<50%)', 'Medium Urban (50-80%)', 'High Urban (>80%)'])
tier_pivot = df.pivot_table(index='gdp_tier', columns='urban_tier',
                             values='wastepercap', aggfunc='mean')

# ============================================================
# 4. HEADER
# ============================================================
st.markdown("""
<div class="figma-header">
    <span style="background: #2563EB; color: #FFF; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;">TOPIC #14</span>
    <h1 style="margin: 10px 0 2px 0; font-size: 32px; font-weight: 800; color: #F9FAFB;">Population vs. E-Waste Generation</h1>
    <p style="margin: 0 0 6px 0; color: #9CA3AF; font-size: 14px;">Is More People = More Waste? A country-level statistical investigation of e-waste drivers.</p>
    <div>
        <span class="team-pill">👤 Vishal Prajapati — 24101B0047</span>
        <span class="team-pill">👤 Altamash Ansari — 24101B0048</span>
        <span class="team-pill">👤 Meet Alshi — 24101B0049</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 5. TOP METRIC CARDS
# ============================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Global Total Waste", f"{df['waste'].sum():,.0f} units")
col2.metric("Avg Per-Capita", f"{df['wastepercap'].mean():.1f} kg/person")
peak_vol_row = df.loc[df['waste'].idxmax()]
col3.metric("Peak National Volume", f"{peak_vol_row['waste']:,.0f} ({peak_vol_row['iso3c']})")
peak_pc_row = df.loc[df['wastepercap'].idxmax()]
col4.metric("Peak Per-Capita", f"{peak_pc_row['wastepercap']:.1f} kg ({peak_pc_row['iso3c']})")

st.write("")

# Dark theme chart layout settings
dark_layout = dict(
    paper_bgcolor='#1F2937',
    plot_bgcolor='#1F2937',
    font=dict(color='#E5E7EB', family="Inter"),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(bgcolor='rgba(0,0,0,0)')
)

# ============================================================
# 6. SECTION 2 — GLOBAL SPATIAL DISTRIBUTION (CHOROPLETH)
# ============================================================
st.markdown('<div class="section-title">🌍 Global Spatial Distribution — Per-Capita E-Waste</div>', unsafe_allow_html=True)

fig_map = px.choropleth(
    df, locations="iso3c", color="wastepercap",
    hover_name="iso3c",
    hover_data={"pop": ":,.0f", "gdp": ":,.0f", "wastepercap": ":.1f"},
    color_continuous_scale="OrRd",
    range_color=(df['wastepercap'].min(), df['wastepercap'].quantile(0.98)),
    labels={"wastepercap": "kg/person"}
)
fig_map.update_layout(**dark_layout, height=520,
                       geo=dict(bgcolor='#1F2937', showframe=False, showcoastlines=True,
                                projection_type="natural earth", lakecolor='#1F2937'))
st.plotly_chart(fig_map, use_container_width=True)

st.markdown(f"""
<div class="insight-box">
    <strong>💡 Key Insight:</strong> Per-capita e-waste is spatially concentrated in wealthy Western economies and small high-income territories, not in high-population developing regions. Highest values reach ~{df['wastepercap'].max():.0f} kg/person, while the lowest sit near ~{df['wastepercap'].min():.0f} kg/person — a gap driven by purchasing power, not headcount.
</div>
""", unsafe_allow_html=True)

# ============================================================
# 7. SECTION 3 — POPULATION VS TOTAL VOLUME (LOG-LOG BUBBLE)
# ============================================================
st.markdown('<div class="section-title">📦 Population vs. Total National Volume</div>', unsafe_allow_html=True)

fig_bubble = px.scatter(
    df, x="pop", y="waste", size="gdp", color="sharepopurban",
    hover_name="iso3c", log_x=True, log_y=True, size_max=45,
    color_continuous_scale="Viridis",
    labels={"pop": "Population (log)", "waste": "Total E-Waste Volume (log)", "sharepopurban": "Urban Pop %"}
)
fig_bubble.update_layout(**dark_layout, height=480)
st.plotly_chart(fig_bubble, use_container_width=True)

st.markdown(f"""
<div class="insight-box">
    <strong>💡 Key Insight:</strong> Population strongly predicts <em>total</em> national e-waste volume
    (linear r = {r_pop_waste_lin:.2f}, log-log r = {r_pop_waste_log:.2f}). Large-population nations post
    enormous total volumes almost regardless of wealth — scale, not affluence, drives this chart.
</div>
""", unsafe_allow_html=True)

# ============================================================
# 8. SECTION 4 — POPULATION VS PER-CAPITA INTENSITY (FACETED)
# ============================================================
# 8. SECTION 4 — POPULATION VS PER-CAPITA INTENSITY (FACETED)
# ============================================================
st.markdown('<div class="section-title">⚖️ Population vs. Per-Capita Intensity — The Direct Test</div>', unsafe_allow_html=True)

# Precompute natural log variables for exact Seaborn-style linear alignment
df['log_pop'] = np.log(df['pop'])
df['log_gdp'] = np.log(df['gdp'])

facet_left, facet_right = st.columns(2)

# --- LEFT PANEL: Population vs Per-Capita ---
with facet_left:
    fig_pop_pc = px.scatter(
        df, 
        x="log_pop", 
        y="wastepercap", 
        hover_name="iso3c",
        trendline="ols",
        title=f"Population vs Per-Capita Waste  (r = {r_pop_percap:.2f})",
        labels={
            "log_pop": "log(Population)", 
            "wastepercap": "E-Waste per Capita (kg/person)"
        }
    )
    
    # Match the reference styling: translucent blue points and dark red trendline
    fig_pop_pc.update_traces(
        marker=dict(color='#4C72B0', opacity=0.6, size=7),
        selector=dict(mode='markers')
    )
    fig_pop_pc.update_traces(
        line=dict(color='#D90429', width=2.5),
        selector=dict(mode='lines')
    )
    fig_pop_pc.update_layout(**dark_layout, height=420)
    st.plotly_chart(fig_pop_pc, use_container_width=True)

# --- RIGHT PANEL: GDP per Capita vs Per-Capita ---
with facet_right:
    fig_gdp_pc = px.scatter(
        df, 
        x="log_gdp", 
        y="wastepercap", 
        hover_name="iso3c",
        trendline="ols",
        title=f"GDP per Capita vs Per-Capita Waste  (r = {r_gdp_percap_log:.2f}, log-log)",
        labels={
            "log_gdp": "log(GDP per Capita, USD)", 
            "wastepercap": "E-Waste per Capita (kg/person)"
        }
    )
    
    # Match the reference styling: translucent blue points and dark red trendline
    fig_gdp_pc.update_traces(
        marker=dict(color='#4C72B0', opacity=0.6, size=7),
        selector=dict(mode='markers')
    )
    fig_gdp_pc.update_traces(
        line=dict(color='#D90429', width=2.5),
        selector=dict(mode='lines')
    )
    fig_gdp_pc.update_layout(**dark_layout, height=420)
    st.plotly_chart(fig_gdp_pc, use_container_width=True)

st.markdown(f"""
<div class="insight-box">
    <strong>💡 Key Insight:</strong> Population has essentially <strong>no relationship</strong> with per-capita
    waste (r = {r_pop_percap:.2f}) — the left panel is nearly flat. GDP per capita, however, is a strong predictor
    (linear r = {r_gdp_percap_lin:.2f}, log-log r = {r_gdp_percap_log:.2f}). The same dataset tells opposite stories
    depending on whether you look at totals or per-person rates.
</div>
""", unsafe_allow_html=True)

# ============================================================
# 9. SECTION 5 — WEALTH TIER x URBANIZATION HEATMAP
# ============================================================
st.markdown('<div class="section-title">🔥 Urbanization × Wealth Tier Matrix</div>', unsafe_allow_html=True)

fig_heat = px.imshow(
    tier_pivot,
    text_auto=".0f",
    color_continuous_scale="YlOrRd",
    labels=dict(x="Urbanization Bracket", y="GDP Tier", color="kg/person"),
    aspect="auto"
)
fig_heat.update_layout(**dark_layout, height=420)
st.plotly_chart(fig_heat, use_container_width=True)

gdp_tier_means = df.groupby('gdp_tier')['wastepercap'].mean()
st.markdown(f"""
<div class="insight-box">
    <strong>💡 Key Insight:</strong> Mean per-capita waste rises sharply across GDP tiers — from
    {gdp_tier_means['Low GDP']:.0f} kg/person (Low GDP) to {gdp_tier_means['Middle GDP']:.0f} kg/person
    (Middle GDP) to {gdp_tier_means['High GDP']:.0f} kg/person (High GDP) — largely independent of the
    urbanization bracket. Wealth tier is the dominant driver in this matrix; urban density is a secondary,
    less consistent effect.
</div>
""", unsafe_allow_html=True)

# ============================================================
# 10. BONUS — TOP 10 HIGH-INTENSITY NATIONS
# ============================================================
st.markdown('<div class="section-title">🏆 Top 10 High-Intensity Per-Capita E-Waste Nations</div>', unsafe_allow_html=True)

top10_pc = df.nlargest(10, "wastepercap")
fig_bar = px.bar(
    top10_pc, x="iso3c", y="wastepercap", color="gdp",
    color_continuous_scale="Blugrn",
    labels={"iso3c": "Country ISO", "wastepercap": "kg/person", "gdp": "GDP ($)"}
)
fig_bar.update_layout(**dark_layout, height=420)
st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("""
<div class="insight-box">
    <strong>💡 Key Insight:</strong> Small, high-income economies dominate the per-capita leaderboard —
    confirming that per-capita rankings surface a completely different set of "risk" countries than a
    ranking by total national volume would.
</div>
""", unsafe_allow_html=True)

# ============================================================
# 11. PROBLEMS IDENTIFIED
# ============================================================
st.markdown('<div class="section-title">🚨 Problems Identified</div>', unsafe_allow_html=True)

p1, p2 = st.columns(2)
with p1:
    st.markdown("""
    <div class="problem-box"><strong>1. Consumption-Driven Overload:</strong>
    High-income populations replace electronics on short 18–24 month cycles, concentrating
    disposal intensity regardless of population size.
    </div>
    <div class="problem-box"><strong>2. Infrastructure Overwhelm at Scale:</strong>
    In large, rapidly growing developing nations, sheer total volume overwhelms local collection
    systems, pushing waste into hazardous open-air burning and informal recycling.
    </div>
    """, unsafe_allow_html=True)
with p2:
    st.markdown("""
    <div class="problem-box"><strong>3. Transboundary Dumping Risk:</strong>
    Wealthy markets' fast device turnover also feeds export of used/e-waste electronics into
    poorer regions with weaker regulatory capacity.
    </div>
    <div class="problem-box"><strong>4. Urban Convenience Bottleneck:</strong>
    Dense, high-income cities concentrate large volumes of discarded electronics with insufficient
    accessible drop-off infrastructure.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 12. STRATEGIC RECOMMENDATIONS & ACTION PLAN
# ============================================================
st.markdown('<div class="section-title">📌 Strategic Recommendations & Action Plan</div>', unsafe_allow_html=True)

st.markdown("""
<div class="recommendation-box">
    <ol style="margin: 0; padding-left: 20px;">
        <li style="margin-bottom: 10px;"><strong>Right-to-Repair Legislation:</strong> Ban planned
        obsolescence and mandate extended warranties in high-intensity, high-GDP regions to slow
        device replacement cycles.</li>
        <li style="margin-bottom: 10px;"><strong>Extended Producer Responsibility (EPR):</strong>
        Require electronics manufacturers in high-GDP markets to fund formal take-back, buy-back,
        and certified recycling logistics proportional to units sold.</li>
        <li style="margin-bottom: 10px;"><strong>National Collection Grids & Recycling Hubs:</strong>
        Build formal municipal collection infrastructure and industrial recycling hubs in large,
        high-total-volume developing nations (e.g., China, India, Indonesia).</li>
        <li><strong>Urban Smart Collection Kiosks & Retail Trade-In Mandates:</strong> Deploy automated
        drop-off kiosks and require retail trade-in programs in dense, high-income urban centers to
        close the convenience bottleneck.</li>
    </ol>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 13. FINAL VERDICT
# ============================================================
st.markdown('<div class="section-title">🎯 Final Verdict</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="verdict-box">
    <strong>"Does more people mean more e-waste?"</strong><br><br>
    <strong>YES for national volume — NO for individual impact.</strong><br><br>
    • More people <strong>does</strong> mean more total national volume: population sets the baseline
    scale a country must manage (linear r = {r_pop_waste_lin:.2f}, log-log r = {r_pop_waste_log:.2f}).<br>
    • More people <strong>does NOT</strong> mean higher individual waste intensity (r = {r_pop_percap:.2f}).
    A person in a high-GDP country generates several times more e-waste than a person in a low-GDP
    country, independent of that country's population.<br>
    • <strong>Final synthesis:</strong> population dictates the scale of infrastructure a country needs;
    economic wealth drives the personal consumption behavior behind each individual's waste footprint.
</div>
""", unsafe_allow_html=True)

# ============================================================
# 14. DATASET INSPECTION TABLE
# ============================================================
st.markdown('<div class="section-title">📋 Filtered Country Dataset</div>', unsafe_allow_html=True)

search = st.text_input("🔎 Search by ISO3 code", "")
display_df = df.copy()
if search:
    display_df = display_df[display_df['iso3c'].str.contains(search.upper(), na=False)]

st.dataframe(
    display_df[['iso3c', 'pop', 'gdp', 'waste', 'wastepercap', 'sharepopurban', 'gdp_tier', 'urban_tier']],
    use_container_width=True,
    height=420
)