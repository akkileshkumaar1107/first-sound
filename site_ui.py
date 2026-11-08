"""FIRST SOUND imaging site - visual system (forest green #1f302b + sand beige #e8dcc4)."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

FOREST = '#1f302b'; FOREST2 = '#2b4239'; FOREST3 = '#3e5d51'; SAGE = '#7f9c86'
SAND = '#e8dcc4'; SAND_L = '#f5efe3'; SAND_D = '#d5c4a1'; LINE = '#d9ccb0'
INK = '#1f302b'; MUTED = '#5d6b64'; CLAY = '#a5603a'; GOLD = '#c08f4f'; MIST = '#4d7184'
SERIES = [FOREST, CLAY, SAGE, GOLD, MIST, '#7b5c78']
# backwards-compatible aliases used by older code
TEXT = INK; TEAL = FOREST3; AMBER = GOLD; CORAL = CLAY; VIOLET = MIST; GREEN = SAGE; SURF = SAND_L; BG = SAND

SOS = [[0, '#14201c'], [0.2, '#2b4a3f'], [0.42, '#6f9479'], [0.6, '#cfd3ad'], [0.72, '#e8dcc4'],
       [0.86, '#d39b62'], [1, '#7f3519']]
CONF = [[0, '#f5efe3'], [0.5, '#9fb59b'], [1, '#1f302b']]
HEAT = [[0, '#f5efe3'], [0.35, '#d9b27a'], [0.7, '#a5603a'], [1, '#3a1a0e']]
DIV = [[0, '#2b4a3f'], [0.5, '#f5efe3'], [1, '#a5603a']]

LABEL_COLORS = {0: '#1f302b', 1: '#e2c58e', 2: '#b3664b', 3: '#8a4a3e', 4: '#a9c3be',
                5: '#c78358', 6: '#7f9c86', 7: '#fffaf0', 8: '#4c6d61', 9: '#ffffff'}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');
:root {{ color-scheme: light; --forest:{FOREST}; --sand:{SAND}; }}
html, body, [data-testid="stAppViewContainer"], .stApp {{ background:{SAND}; color:{INK};
  font-family:'Inter',system-ui,-apple-system,sans-serif; }}
[data-testid="stHeader"] {{ background:transparent; }}
[data-testid="stToolbar"] {{ color:{INK}; }}
.block-container {{ max-width:1440px; padding:1.4rem 2.6rem 4rem; }}
#MainMenu, footer {{ visibility:hidden; }}
h1, h2, h3, h4 {{ font-family:'Fraunces',Georgia,serif !important; color:{INK} !important; letter-spacing:-.01em; }}
p, li, label, .stMarkdown {{ color:{INK}; }}
[data-testid="stCaptionContainer"] p {{ color:{MUTED} !important; }}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {{ background:linear-gradient(180deg,{FOREST} 0%,#18251f 100%); border-right:0; }}
[data-testid="stSidebar"] * {{ color:{SAND} !important; }}
[data-testid="stSidebarNav"] a {{ border-radius:10px; padding:6px 10px; }}
[data-testid="stSidebarNav"] a:hover {{ background:rgba(232,220,196,.08); }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{ background:rgba(232,220,196,.16); }}
[data-testid="stSidebarNavSeparator"] {{ border-color:rgba(232,220,196,.15); }}
[data-testid="stNavSectionHeader"], [data-testid="stSidebarNav"] header {{ color:{SAND_D} !important; letter-spacing:.14em; font-size:10.5px !important; text-transform:uppercase; }}
[data-testid="stSidebarNav"] span[data-testid="stIconMaterial"] {{ color:{SAND_D} !important; }}

/* ---------- widgets ---------- */
[data-testid="stMetric"] {{ background:{SAND_L}; border:1px solid {LINE}; border-radius:16px; padding:16px 18px;
  box-shadow:0 1px 0 rgba(31,48,43,.04); }}
[data-testid="stMetricLabel"] p {{ color:{MUTED} !important; font-size:11.5px; letter-spacing:.08em; text-transform:uppercase; }}
[data-testid="stMetricValue"] {{ font-family:"Fraunces",serif; font-size:24px; color:{INK}; }}
[data-testid="stPlotlyChart"] {{ background:{SAND_L}; border:1px solid {LINE}; border-radius:18px; padding:6px; }}
[data-testid="stDataFrame"] {{ border:1px solid {LINE}; border-radius:14px; overflow:hidden; }}
[data-testid="stExpander"] {{ background:{SAND_L}; border:1px solid {LINE} !important; border-radius:14px; }}
[data-testid="stExpander"] summary p {{ font-weight:600; }}
[data-testid="stFileUploaderDropzone"] {{ background:{SAND_L}; border:1.5px dashed {SAND_D}; border-radius:14px; }}
[data-testid="stForm"] {{ background:{SAND_L}; border:1px solid {LINE}; border-radius:18px; padding:18px; }}
div[data-baseweb="select"] > div, .stNumberInput input, .stTextInput input {{ background:#fbf8f1 !important; border-color:{LINE} !important; }}
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button, [data-testid="stPageLink"] a {{
  border-radius:999px; font-weight:600; border:1px solid {FOREST3}; background:transparent; color:{INK}; }}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{
  background:{FOREST}; border-color:{FOREST}; }}
.stButton > button[kind="primary"] p, .stFormSubmitButton > button[kind="primary"] p, .stDownloadButton > button[kind="primary"] p {{ color:{SAND} !important; }}
.stButton > button:hover {{ border-color:{CLAY}; color:{CLAY}; }}
.stTabs [data-baseweb="tab-list"] {{ gap:4px; border-bottom:1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ background:transparent; padding:8px 16px; border-radius:10px 10px 0 0; }}
.stTabs [aria-selected="true"] p {{ color:{FOREST} !important; font-weight:700; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:{CLAY}; }}
[data-testid="stSlider"] [role="slider"] {{ background:{FOREST}; border-color:{FOREST}; }}
.stAlert {{ border-radius:14px; }}

/* ---------- page header ---------- */
.fs-kicker {{ color:{CLAY}; font-size:11.5px; font-weight:700; letter-spacing:.22em; text-transform:uppercase; margin:4px 0 2px; }}
.fs-title {{ font-family:'Fraunces',serif; font-size:2.6rem; font-weight:600; margin:0 0 6px; line-height:1.08; color:{INK}; }}
.fs-sub {{ color:{MUTED}; font-size:15.5px; line-height:1.65; margin-bottom:14px; max-width:920px; }}
.fs-badge {{ display:inline-block; font-size:10.5px; font-weight:700; letter-spacing:.12em; padding:5px 11px; border-radius:999px; margin:0 6px 12px 0; }}
.fs-syn {{ background:rgba(192,143,79,.16); color:#7a5420; border:1px solid rgba(192,143,79,.45); }}
.fs-real {{ background:rgba(31,48,43,.08); color:{FOREST}; border:1px solid rgba(31,48,43,.35); }}
.fs-val {{ background:rgba(77,113,132,.12); color:{MIST}; border:1px solid rgba(77,113,132,.4); }}
.fs-unit {{ background:{FOREST}; color:{SAND}; border:1px solid {FOREST}; }}
.fs-note {{ background:{SAND_L}; border:1px solid {LINE}; border-left:4px solid {GOLD}; border-radius:12px;
  padding:13px 16px; font-size:14px; color:{INK}; margin:8px 0 16px; line-height:1.6; }}
.fs-ok {{ border-left-color:{FOREST3}; }}
.fs-cap {{ color:{MUTED}; font-size:12.5px; margin:-4px 0 18px; font-style:italic; }}
.fs-card {{ margin-bottom:6px; }}
.fs-slot {{ aspect-ratio:4/3; border:1.5px dashed {SAND_D}; border-radius:14px; background:repeating-linear-gradient(135deg,{SAND_L},{SAND_L} 12px,#efe6d3 12px,#efe6d3 24px);
  display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center; padding:14px; gap:6px; margin-bottom:8px; }}
.fs-slot b {{ font-family:'Fraunces',serif; font-size:15px; color:{INK}; }}
.fs-slot span {{ font-size:12.5px; color:{MUTED}; line-height:1.5; max-width:260px; }}
.fs-slot code {{ font-size:11px; background:{FOREST}; color:{SAND}; padding:2px 8px; border-radius:6px; }}
[data-testid="stImage"] img {{ border-radius:14px; }}
.fs-eq {{ font-family:'JetBrains Mono',monospace; background:{FOREST}; color:{SAND}; border-radius:12px; padding:10px 16px; display:inline-block; margin:4px 0 12px; font-size:14px; }}
.fs-table table {{ border-collapse:collapse; width:100%; font-size:13.5px; background:{SAND_L}; border-radius:14px; overflow:hidden; }}
.fs-table th {{ background:{FOREST}; color:{SAND}; text-align:left; padding:9px 12px; font-weight:600; }}
.fs-table td {{ border-top:1px solid {LINE}; padding:8px 12px; vertical-align:top; }}
.fs-foot {{ color:{MUTED}; font-size:12px; border-top:1px solid {LINE}; padding-top:18px; margin-top:40px; }}
.fs-brand {{ font-family:'Fraunces',serif; font-weight:600; font-size:22px; letter-spacing:.02em; color:{SAND}; }}
.fs-card {{ background:{SAND_L}; border:1px solid {LINE}; border-radius:18px; padding:18px 20px; height:100%; }}
.fs-card h4 {{ margin:0 0 6px; font-size:17px; }}
.fs-card p {{ margin:0; color:{MUTED}; font-size:13.5px; line-height:1.6; }}
.fs-chain {{ display:flex; gap:8px; flex-wrap:wrap; align-items:stretch; margin:6px 0 18px; }}
.fs-chain div {{ flex:1 1 130px; background:{SAND_L}; border:1px solid {LINE}; border-radius:14px; padding:10px 12px; font-size:12.5px; color:{MUTED}; }}
.fs-chain b {{ display:block; color:{INK}; font-size:13.5px; margin-bottom:2px; }}
.fs-chain i {{ font-style:normal; font-size:10.5px; letter-spacing:.12em; color:{CLAY}; font-weight:700; }}
@media (max-width:760px) {{ .block-container {{ padding:1rem; }} .fs-title {{ font-size:2rem; }} }}
</style>
"""


_BANNER = None


def _banner_css():
    """Chapter banner (AI-generated abstract art in the project colours), embedded once."""
    global _BANNER
    if _BANNER is None:
        import base64
        from pathlib import Path
        p = Path(__file__).resolve().parent / 'assets' / 'ai' / 'banner.jpg'
        _BANNER = ''
        if p.exists():
            _BANNER = ('<style>.fs-banner{height:104px;border-radius:18px;margin:0 0 16px;'
                       'background:url(data:image/jpeg;base64,' + base64.b64encode(p.read_bytes()).decode() +
                       ') center 42%/cover no-repeat;box-shadow:inset 0 0 0 1px rgba(31,48,43,.15)}</style>')
    return _BANNER


def inject():
    st.markdown(CSS + _banner_css(), unsafe_allow_html=True)


def page_header(kicker, title, sub, badges=()):
    b = ''.join(f'<span class="fs-badge fs-{k}">{t}</span>' for k, t in badges)
    st.markdown(('<div class="fs-banner"></div>' if _banner_css() else '') + f'<div class="fs-kicker">{kicker}</div><div class="fs-title">{title}</div>'
                f'<div class="fs-sub">{sub}</div>{b}', unsafe_allow_html=True)


def note(html, ok=False):
    st.markdown(f'<div class="fs-note{" fs-ok" if ok else ""}">{html}</div>', unsafe_allow_html=True)


def chain(steps):
    st.markdown('<div class="fs-chain">' + ''.join(
        f'<div><i>{u}</i><b>{t}</b>{d}</div>' for u, t, d in steps) + '</div>', unsafe_allow_html=True)


def base_layout(title='', h=420, x='', y=''):
    return dict(title=dict(text=title, font=dict(size=15, color=INK, family='Fraunces, Georgia, serif'), x=0.02),
                paper_bgcolor=SAND_L, plot_bgcolor=SAND_L, font=dict(color=MUTED, family='Inter', size=12),
                height=h, margin=dict(l=58, r=20, t=50, b=48), colorway=SERIES,
                xaxis=dict(title=x, gridcolor='#e6dcc8', zerolinecolor=LINE, linecolor=LINE),
                yaxis=dict(title=y, gridcolor='#e6dcc8', zerolinecolor=LINE, linecolor=LINE),
                legend=dict(orientation='h', y=1.0, yanchor='bottom', x=1, xanchor='right', bgcolor='rgba(0,0,0,0)'))


CFG = {'displaylogo': False, 'modeBarButtonsToRemove': ['lasso2d', 'select2d']}

CFG = {'displaylogo': False, 'modeBarButtonsToRemove': ['lasso2d', 'select2d']}


def footer():
    st.markdown('<div class="fs-foot"><b>FIRST SOUND</b> · Ultrasound Acoustic Mapping and Reconstruction of Maternal, '
                'Fetal and Placental Tissues · Medical Imaging mini project · SRM Institute of Science and Technology · '
                'Real images: HC18 (CC BY 4.0), FETAL_PLANES_DB and Wikimedia Commons (credited under each photo) · '
                'Hero and banner art: AI-generated illustrations · Teaching software, not a medical device</div>',
                unsafe_allow_html=True)


def cards(items, cols=3):
    """items: list of (title, text). Simple sand cards in a grid."""
    for i in range(0, len(items), cols):
        cs = st.columns(cols)
        for c, (t, d) in zip(cs, items[i:i + cols]):
            c.markdown(f'<div class="fs-card"><h4>{t}</h4><p>{d}</p></div>', unsafe_allow_html=True)
        st.write('')


def figure(path, caption):
    st.image(path, use_container_width=True)
    st.markdown(f'<div class="fs-cap">{caption}</div>', unsafe_allow_html=True)


def section(title, kicker=None):
    k = f'<div class="fs-kicker">{kicker}</div>' if kicker else ''
    st.markdown(f'{k}<h3 style="margin:22px 0 6px">{title}</h3>', unsafe_allow_html=True)
