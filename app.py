"""FIRST SOUND - image processing of real ultrasound scans, step by step.

  Hero (full screen, looping video)  ->  Upload or pick a real sample scan  ->  5 processing steps  ->  result

Run from this folder:   python3 -m streamlit run app.py
"""
import base64
import json
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from scipy import ndimage as ndi
from skimage.metrics import structural_similarity as ssim

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ASSETS = HERE / 'assets'
SAMPLES = ASSETS / 'samples'
STATIC = HERE / 'static'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'image_processing'))

import site_ui as ui  # noqa: E402
import first_sound_ip as ip  # noqa: E402

st.set_page_config(page_title='FIRST SOUND · Ultrasound Image Processing', page_icon='🫧', layout='wide',
                   initial_sidebar_state='collapsed')
ui.inject()
FOREST, CLAY, GOLD, SAGE = ui.FOREST, ui.CLAY, ui.GOLD, ui.SAGE

st.markdown("""
<style>
header[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"], #MainMenu, footer, .fs-banner {display:none !important;}
.block-container, [data-testid="stMainBlockContainer"] {max-width:100% !important; padding:0 !important;}
.stMain, [data-testid="stAppViewContainer"], [data-testid="stMain"] {padding:0 !important; margin:0 !important;}
[data-testid="stElementContainer"]:has(style):not(:has(img)):not(:has(video)) {display:none !important;}
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {gap:0 !important;}

/* ---------------- hero: plain HTML in the page, 100% of the window ---------------- */
.fs-hero {position:relative; height:100vh; min-height:560px; width:100%; overflow:hidden; background:#15211d; color:#e8dcc4;}
.fs-hero video, .fs-hero .still {position:absolute; inset:0; width:100%; height:100%; object-fit:cover; background-size:cover;
  background-position:center;}
.fs-hero .veil {position:absolute; inset:0; background:
  linear-gradient(90deg, rgba(15,24,21,.93) 0%, rgba(15,24,21,.80) 34%, rgba(15,24,21,.28) 64%, rgba(15,24,21,.06) 100%),
  linear-gradient(0deg, rgba(15,24,21,.8) 0%, rgba(15,24,21,0) 35%);}
.fs-hero .content {position:absolute; left:clamp(24px,7vw,140px); top:50%; transform:translateY(-50%); width:min(680px,86vw);
  animation:fsUp 1s cubic-bezier(.2,.7,.2,1) .15s both;}
@keyframes fsUp {from {opacity:0; margin-top:24px;} to {opacity:1; margin-top:0;}}
.fs-hero .eyebrow {font-weight:700; font-size:12px; letter-spacing:.26em; text-transform:uppercase; color:#c9b58f; margin-bottom:18px;}
.fs-hero h1 {font-family:'Fraunces',Georgia,serif !important; font-weight:600; font-size:clamp(56px,8.4vw,132px) !important;
  line-height:.92 !important; letter-spacing:-.02em; margin:0 0 14px !important; padding:0 !important; color:#f1e7d3 !important;}
.fs-hero .tag {font-family:'Fraunces',Georgia,serif; font-style:italic; font-size:clamp(19px,1.9vw,27px); color:#d9b98a; margin-bottom:18px;}
.fs-hero p {font-size:clamp(14.5px,1.15vw,17px) !important; line-height:1.7; color:#ddd1b9 !important; margin:0 0 28px; max-width:580px;}
.fs-hero .stats {display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:22px; max-width:620px;}
.fs-hero .stats div {border-top:1px solid rgba(232,220,196,.3); padding-top:12px; min-width:0;}
.fs-hero .stats b {display:block; font-family:'Fraunces',Georgia,serif; font-weight:600; font-size:clamp(22px,2.2vw,32px); color:#f1e7d3; line-height:1.1;}
.fs-hero .stats span {display:block; margin-top:6px; font-size:10.5px; line-height:1.35; letter-spacing:.09em; text-transform:uppercase; color:#bfae8c;}
.fs-hero .cue {position:absolute; left:50%; bottom:22px; transform:translateX(-50%); display:flex; flex-direction:column;
  align-items:center; gap:6px; font-size:11px; font-weight:600; letter-spacing:.24em; text-transform:uppercase;
  color:#d9c7a3 !important; text-decoration:none !important; cursor:pointer; z-index:3;
  opacity:calc(1 - var(--p, 0) * 3);}
.fs-hero .cue .arc {fill:none; stroke:#e0a458; stroke-width:2.2; stroke-linecap:round; opacity:0;
  animation:fsEcho 1.8s ease-out infinite;}
.fs-hero .cue .a2 {animation-delay:.25s;} .fs-hero .cue .a3 {animation-delay:.5s;}
@keyframes fsEcho {0% {opacity:0; transform:translateY(-4px);} 25% {opacity:1;} 100% {opacity:0; transform:translateY(5px);}}
.fs-hero .cue:hover .arc {stroke:#f1e7d3;}
.fs-hero .scanline {position:absolute; left:0; right:0; height:2px; top:calc(var(--p, 0) * 100%); z-index:2; pointer-events:none;
  background:linear-gradient(90deg, transparent, #e0a458 20%, #f6d9a6 50%, #e0a458 80%, transparent);
  box-shadow:0 0 18px 4px rgba(224,164,88,.55); opacity:calc(var(--p, 0) * 6);}
.fs-hero video, .fs-hero .still {transform:scale(calc(1 + var(--p, 0) * .08)); filter:brightness(calc(1 - var(--p, 0) * .55));
  will-change:transform;}
.fs-hero .content {opacity:calc(1 - var(--p, 0) * 1.7);}
.fs-hero .content > * {transform:translateY(calc(var(--p, 0) * -60px));}
/* steps "scan in" from top to bottom, like a B-mode frame being drawn */
.fs-wait {clip-path:inset(0 0 100% 0); opacity:.2;}
.fs-show {clip-path:inset(0 0 0 0); opacity:1; transition:clip-path 1s cubic-bezier(.3,.6,.2,1), opacity .6s ease;}
.fs-hero .credit {position:absolute; right:20px; bottom:16px; font-size:10.5px; color:rgba(232,220,196,.55);}
@media (max-width:700px) {.fs-hero .stats {grid-template-columns:repeat(2, minmax(0,1fr));} .fs-hero .veil {background:rgba(15,24,21,.75);}}

/* ---------------- page body ---------------- */
.st-key-main {max-width:min(1560px, 92vw); margin:0 auto; padding:clamp(64px,8vh,110px) clamp(20px,3vw,48px) 64px !important;
  box-sizing:border-box; background:#e8dcc4; box-shadow:0 0 0 100vmax #e8dcc4; clip-path:inset(0 -100vmax);}
.st-key-end {max-width:min(1560px, 92vw); margin:0 auto; padding:40px clamp(20px,3vw,48px) !important; box-sizing:border-box;
  background:#1f302b; box-shadow:0 0 0 100vmax #1f302b; clip-path:inset(0 -100vmax);}
.st-key-end div {color:#d8ccb3;}
.fs-chap {display:flex; gap:28px; align-items:flex-end; margin-bottom:34px; border-bottom:1px solid #d5c4a1; padding-bottom:22px;}
.fs-chap .k {font-size:11.5px; font-weight:700; letter-spacing:.24em; text-transform:uppercase; color:#a5603a; margin-bottom:6px;}
.fs-chap h2 {font-family:'Fraunces',serif !important; font-size:clamp(40px,3.3vw,60px) !important; margin:0 !important; line-height:1.05; padding:0 !important;}
.fs-chap p {color:#5d6b64; font-size:clamp(15.5px,1.05vw,18px); margin:8px 0 0; max-width:900px; line-height:1.6;}
.fs-step {position:relative; padding-left:64px; margin-top:6px;}
.fs-step .n {position:absolute; left:0; top:0; width:46px; height:46px; border-radius:50%; background:#1f302b; color:#e8dcc4;
  font-family:'Fraunces',serif; font-size:21px; display:flex; align-items:center; justify-content:center;}
.fs-step .t {font-family:'Fraunces',serif; font-size:clamp(24px,1.7vw,30px); font-weight:600; color:#1f302b; line-height:1.15; margin:4px 0 6px;}
.fs-step .w {font-size:11px; font-weight:700; letter-spacing:.18em; text-transform:uppercase; color:#a5603a;}
.fs-step p {font-size:clamp(14.5px,0.95vw,17px); line-height:1.65; color:#2c3a35; margin:8px 0;}
.fs-step .how {font-size:clamp(13px,0.85vw,15px); color:#5d6b64; border-left:3px solid #c08f4f; padding:4px 0 4px 12px; margin:10px 0;}
.fs-eq {font-family:'JetBrains Mono',monospace; background:#1f302b; color:#e8dcc4; border-radius:10px; padding:8px 14px;
  display:inline-block; margin:4px 0 10px; font-size:clamp(13px,0.85vw,15px);}
.fs-rule {height:1px; background:#d5c4a1; margin:34px 0 30px;}
.fs-pill {display:inline-block; font-size:11px; font-weight:700; letter-spacing:.1em; padding:4px 10px; border-radius:999px;
  background:rgba(31,48,43,.08); color:#1f302b; border:1px solid rgba(31,48,43,.25); margin:0 6px 6px 0;}
.fs-cap {color:#5d6b64; font-size:clamp(12.5px,0.8vw,14px); margin:-4px 0 6px; font-style:italic;}
.fs-imgbox {width:100%; margin-bottom:8px; border-radius:12px; overflow:hidden; background:#101814;}
.fs-img {width:100%; height:100%; display:block; object-fit:contain;}
/* no grey-out flicker while Streamlit recomputes */
[data-stale="true"], .stale-element, [data-testid="stElementContainer"][data-stale] {opacity:1 !important; transition:none !important;}
.stMain {overflow-anchor:auto;}
.fs-thumb {width:100%; aspect-ratio:4/3; object-fit:cover; display:block; border-radius:12px; margin-bottom:8px;
  outline:3px solid transparent; outline-offset:2px;}
.fs-thumb.on {outline-color:#a5603a;}
.fs-samp {margin:2px 0 8px;}
.fs-samp b {display:block; font-size:14.5px; line-height:1.3; color:#1f302b; min-height:2.6em;}
.fs-samp span {display:block; font-size:11px; letter-spacing:.12em; text-transform:uppercase; color:#a5603a; margin-top:2px;}
.st-key-samples [data-testid="stButton"] button, .st-key-samples [data-testid="stDownloadButton"] button {
  white-space:nowrap; min-height:38px; padding:4px 10px; font-size:14px;}
.st-key-samples [data-testid="stButton"] button p, .st-key-samples [data-testid="stDownloadButton"] button p {
  white-space:nowrap; font-size:14px;}
.st-key-step_go {background:#f5efe3; border:1.5px solid #c08f4f; border-radius:20px; padding:22px 24px !important;}
[data-testid="stPlotlyChart"] {overflow:hidden; box-sizing:border-box;}
.st-key-samples [data-testid="stImage"] img {aspect-ratio:4/3; object-fit:cover;}
[data-testid="stFileUploaderDropzone"] {padding:28px !important;}
</style>
""", unsafe_allow_html=True)


# =====================================================================  helpers
def u8(a):
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


def step_text(n, what, title, body, how=None, eq=None):
    h = f'<div class="how">{how}</div>' if how else ''
    e = f'<div class="fs-eq">{eq}</div>' if eq else ''
    st.markdown(f'<div class="fs-step"><div class="n">{n}</div><div class="w">{what}</div><div class="t">{title}</div>'
                f'<p>{body}</p>{h}{e}</div>', unsafe_allow_html=True)


def rule():
    st.markdown('<div class="fs-rule"></div>', unsafe_allow_html=True)


def img_uri(arr, quality=86):
    """Encode an image (array in 0..1, or a file path) as an inline JPEG data URI.
    Images are put straight into the page, so they never go missing when Streamlit reruns."""
    if isinstance(arr, str):
        im = Image.open(arr).convert('RGB')
    else:
        a_ = u8(arr)
        im = Image.fromarray(a_ if a_.ndim == 2 else a_[..., :3])
        if im.mode != 'RGB':
            im = im.convert('RGB')
    if max(im.size) > 1000:
        im.thumbnail((1000, 1000))
    buf = io.BytesIO(); im.save(buf, format='JPEG', quality=quality)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode(), im.size


@st.cache_data(show_spinner=False)
def thumb_uri(path):
    im = Image.open(path).convert('RGB'); im.thumbnail((420, 420))
    buf = io.BytesIO(); im.save(buf, format='JPEG', quality=85)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


def pic(col, arr, caption):
    # the box has its final size before the picture decodes, so nothing on the page jumps when it is replaced
    uri, (w, h) = img_uri(arr)
    col.markdown(f'<div class="fs-imgbox" style="aspect-ratio:{w}/{h}"><img class="fs-img" src="{uri}" width="{w}" '
                 f'height="{h}" alt="{caption}"></div><div class="fs-cap">{caption}</div>', unsafe_allow_html=True)


def lay(fig, title='', h=260, x='', y=''):
    fig.update_layout(**ui.base_layout(title, h, x, y))
    fig.update_layout(margin=dict(l=54, r=16, t=44 if title else 16, b=44))
    return fig


def plot(fig):
    st.plotly_chart(fig, use_container_width=True, config=ui.CFG)  # plotly: keep


@st.cache_data(show_spinner=False, max_entries=4)
def data_uri(path, mime, mtime):
    return 'data:' + mime + ';base64,' + base64.b64encode(Path(path).read_bytes()).decode()


@st.cache_data(show_spinner=False)
def n_real_images():
    n = 0
    tr = ROOT / 'data' / 'training_set'
    fp = ROOT / 'data' / 'FETAL_PLANES_ZENODO' / 'Images'
    if tr.exists():
        n += sum(1 for _ in tr.glob('*HC.png'))
    if fp.exists():
        n += sum(1 for _ in fp.glob('*.png'))
    return n


def pct(a, b):
    return (b - a) / (abs(a) + 1e-12) * 100


# what the grey structures are in each kind of scan (used to explain every step for THIS image)
WORDS = {
    'head':    dict(scan='fetal head scan', tissue='the fetal brain inside the skull', bright='the skull bone',
                    dark='the amniotic fluid around the head', edge='skull outline', inner='brain (inside the skull)',
                    outer='fluid and soft tissue just outside the skull'),
    'brain':   dict(scan='fetal brain scan', tissue='the fetal brain tissue', bright='the skull and the midline',
                    dark='the fluid-filled ventricles and amniotic fluid', edge='skull outline', inner='central region',
                    outer='ring around it'),
    'cervix':  dict(scan='transvaginal scan of the maternal cervix', tissue='the cervical tissue',
                    bright='the walls of the cervix', dark='the cervical canal and the fluid near it',
                    edge='cervical canal and walls', inner='central region', outer='ring around it'),
    'abdomen': dict(scan='fetal abdomen scan', tissue='the fetal liver and abdominal wall', bright='the spine and ribs',
                    dark='the stomach bubble and umbilical vein', edge='abdominal outline', inner='central region',
                    outer='ring around it'),
    'femur':   dict(scan='fetal femur scan', tissue='the soft tissue of the thigh', bright='the femur bone',
                    dark='the surrounding amniotic fluid', edge='bone outline', inner='central region', outer='ring around it'),
    'upload':  dict(scan='uploaded scan', tissue='the soft tissue', bright='bone and strong boundaries',
                    dark='fluid-filled areas', edge='tissue edges', inner='central region', outer='ring around it'),
}

# =====================================================================  HERO
vid_html = ''
hero_file = None
for name in ('hero_loop.mp4', 'hero_loop.webm'):
    if (STATIC / name).exists():
        hero_file = STATIC / name
        break
if hero_file is None and STATIC.exists():
    cands = sorted(p for p in STATIC.iterdir() if p.suffix.lower() in ('.mp4', '.webm'))
    hero_file = cands[0] if cands else None
poster = ASSETS / 'ai' / 'hero_still.jpg'
poster_uri = data_uri(str(poster), 'image/jpeg', poster.stat().st_mtime) if poster.exists() else ''
if hero_file is not None:
    mime = 'video/webm' if hero_file.suffix.lower() == '.webm' else 'video/mp4'
    vid_html = (f'<video autoplay muted loop playsinline preload="auto" poster="{poster_uri}" '
                f'src="{data_uri(str(hero_file), mime, hero_file.stat().st_mtime)}"></video>')
elif poster_uri:
    vid_html = f'<div class="still" style="background-image:url({poster_uri})"></div>'
n_img = n_real_images() or 13399
st.markdown(f"""
<div class="fs-hero">
  {vid_html}
  <div class="veil"></div>
  <div class="content">
    <div class="eyebrow">Medical imaging &middot; Obstetric ultrasound</div>
    <h1>FIRST<br>SOUND</h1>
    <div class="tag">Listening to Life Before Birth</div>
    <p>Basic image processing of real maternal and fetal ultrasound scans. Upload a scan, or pick one of the real
       samples, and watch it being cleaned step by step: speckle removed, contrast fixed, edges sharpened,
       every change explained and measured.</p>
    <div class="stats">
      <div><b>{n_img:,}</b><span>real scans in the databases</span></div>
      <div><b>7</b><span>sample scans to try</span></div>
      <div><b>5</b><span>processing steps</span></div>
      <div><b>7</b><span>quality measures</span></div>
    </div>
  </div>
  <div class="scanline"></div>
  <a class="cue" href="#fs-start" aria-label="Scroll to the image processing">
    <svg viewBox="0 0 60 64" width="46" height="50" aria-hidden="true">
      <path d="M22 4 h16 v12 q0 4 -4 4 h-8 q-4 0 -4 -4 z" fill="#e8dcc4"/>
      <path d="M18 20 h24 q2 0 2 2 q-14 6 -28 0 q0 -2 2 -2 z" fill="#c08f4f"/>
      <path class="arc a1" d="M17 30 q13 7 26 0" />
      <path class="arc a2" d="M13 39 q17 9 34 0" />
      <path class="arc a3" d="M9 49 q21 11 42 0" />
    </svg>
    <span>Scroll to scan</span></a>
  <div class="credit">AI-generated video</div>
</div>
<div id="fs-start"></div>
""", unsafe_allow_html=True)

FX_CSS = """
#fs-trans {position:fixed; inset:0; z-index:99999; pointer-events:none; display:none;}
#fs-trans.on {display:block;}
#fs-trans .shot {position:absolute; inset:0; overflow:hidden; background:#15211d;}
#fs-trans .shot .fs-hero {height:100% !important;}
#fs-trans canvas.fx {position:absolute; inset:0; width:100%; height:100%;}
.fs-wait {clip-path:inset(0 0 100% 0); opacity:.15;}
.fs-show {clip-path:inset(0 0 0 0); opacity:1; transition:clip-path 1s cubic-bezier(.3,.6,.2,1), opacity .6s ease;}
"""

FX_JS = r"""
(function () {
  var W = window, D = document;
  if (W.__fsFx) return; W.__fsFx = true;
  function q(s) { return D.querySelector(s); }
  function main() { return q('section.stMain') || q('[data-testid="stMain"]'); }
  var ov = D.createElement('div'); ov.id = 'fs-trans';
  ov.innerHTML = '<div class="shot"></div><canvas class="fx"></canvas>';
  D.body.appendChild(ov);
  var shot = ov.querySelector('.shot'), cv = ov.querySelector('canvas'), cx = cv.getContext('2d');
  var busy = false, lockUntil = 0;
  function size() { var m = main(), e = q('section.stMain .fs-hero'); if (m && e) e.style.height = (m.clientHeight || W.innerHeight) + 'px'; }
  function startY() { var m = main(), t = D.getElementById('fs-start'); if (!m || !t) return 0;
    return t.getBoundingClientRect().top - m.getBoundingClientRect().top + m.scrollTop; }
  function ease(k) { return k < .5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2; }
  function smooth(y, T, done) {
    var m = main(), s = m.scrollTop, d = y - s, t0 = 0;
    function f(t) { if (!t0) t0 = t; var k = Math.min(1, (t - t0) / T); m.scrollTop = s + d * ease(k);
      if (k < 1) W.requestAnimationFrame(f); else if (done) done(); }
    W.requestAnimationFrame(f);
  }
  // ---- transition: ultrasound fan opens, a baby scans in, zoom into the head, land on the next page
  var noise = null;
  function makeNoise(w, h) {
    var c = D.createElement('canvas'); c.width = Math.ceil(w / 3); c.height = Math.ceil(h / 3);
    var g = c.getContext('2d'), im = g.createImageData(c.width, c.height);
    for (var i = 0; i < im.data.length; i += 4) { var v = Math.pow(Math.random(), 3) * 255;
      im.data[i] = v * .95; im.data[i + 1] = v * .9; im.data[i + 2] = v * .8; im.data[i + 3] = 255; }
    g.putImageData(im, 0, 0); return c;
  }
  function babyPath(g) {
    // head: skull from the forehead over the top and back of the head, down to the nape
    g.beginPath();
    g.moveTo(-0.62, -0.30);                                   // forehead
    g.bezierCurveTo(-0.58, -0.62, -0.12, -0.70, 0.02, -0.40); // crown to back of head
    g.bezierCurveTo(0.10, -0.22, 0.06, -0.06, -0.02, 0.02);   // occiput to nape
    // back (curled spine) to the bottom
    g.bezierCurveTo(0.30, 0.02, 0.62, 0.30, 0.52, 0.66);
    g.bezierCurveTo(0.46, 0.86, 0.22, 0.90, 0.10, 0.80);      // buttocks
    // thigh up toward the tummy, knee, lower leg folded back down, foot
    g.bezierCurveTo(-0.02, 0.70, -0.10, 0.56, -0.16, 0.46);   // thigh to knee
    g.bezierCurveTo(-0.20, 0.40, -0.12, 0.36, -0.04, 0.42);   // knee bend
    g.bezierCurveTo(0.06, 0.50, 0.12, 0.60, 0.04, 0.66);      // shin to foot
    // tummy up to the chest and chin
    g.moveTo(-0.16, 0.46);
    g.bezierCurveTo(-0.26, 0.34, -0.30, 0.18, -0.36, 0.08);   // tummy / chest
    // face profile: chin, lips, nose, forehead
    g.bezierCurveTo(-0.44, 0.04, -0.52, 0.02, -0.56, -0.04);  // under the chin
    g.quadraticCurveTo(-0.62, -0.06, -0.62, -0.10);           // lips
    g.quadraticCurveTo(-0.70, -0.14, -0.66, -0.20);           // nose
    g.quadraticCurveTo(-0.64, -0.26, -0.62, -0.30);           // brow back to the forehead
    // arm: shoulder, elbow, little hand by the face
    g.moveTo(-0.30, 0.10);
    g.quadraticCurveTo(-0.24, 0.30, -0.36, 0.28);
    g.quadraticCurveTo(-0.46, 0.26, -0.48, 0.12);
    g.moveTo(-0.48, 0.12); g.lineTo(-0.52, 0.08); g.moveTo(-0.48, 0.12); g.lineTo(-0.47, 0.06);
  }
  function drawBaby(g, x, y, s, beat, a) {
    g.save(); g.translate(x, y); g.scale(s, s); g.globalAlpha = a;
    g.lineCap = 'round'; g.lineJoin = 'round';
    g.fillStyle = 'rgba(200,185,160,.07)'; g.beginPath(); g.ellipse(-0.30, -0.30, 0.31, 0.30, 0, 0, Math.PI * 2); g.fill();
    g.shadowColor = 'rgba(246,226,190,.9)'; g.shadowBlur = 18;
    g.strokeStyle = 'rgba(240,228,205,.95)'; g.lineWidth = 0.03; babyPath(g); g.stroke();           // bright echo outline
    g.shadowBlur = 0; g.strokeStyle = 'rgba(240,228,205,.22)'; g.lineWidth = 0.09; babyPath(g); g.stroke();  // soft halo
    g.fillStyle = 'rgba(240,228,205,.6)'; g.beginPath(); g.ellipse(-0.50, -0.18, 0.035, 0.02, 0, 0, Math.PI * 2); g.fill();  // eye
    g.fillStyle = 'rgba(240,228,205,.45)';                                                          // spine
    for (var i = 0; i < 10; i++) { var t = i / 9, px = 0.0 + 0.44 * t + 0.10 * Math.sin(t * 3.0), py = 0.08 + 0.52 * t * t;
      g.beginPath(); g.arc(px, py, 0.018, 0, Math.PI * 2); g.fill(); }
    g.strokeStyle = 'rgba(240,228,205,.35)'; g.lineWidth = 0.022; g.beginPath();                  // umbilical cord
    for (var j = 0; j <= 30; j++) { var u = j / 30, cxp = -0.20 + 0.10 * u + 0.05 * Math.sin(u * 16), cyp = 0.40 + 0.80 * u;
      if (j) g.lineTo(cxp, cyp); else g.moveTo(cxp, cyp); } g.stroke();
    var hr = 0.035 + 0.025 * beat;                                                                  // the beating heart
    g.shadowColor = 'rgba(224,120,88,.95)'; g.shadowBlur = 26; g.fillStyle = 'rgba(232,140,100,' + (0.55 + 0.45 * beat) + ')';
    g.beginPath(); g.arc(-0.14, 0.20, hr, 0, Math.PI * 2); g.fill();
    g.restore();
  }
  function fanDown() {
    var m = main(), hero = q('section.stMain .fs-hero'); if (!m || !hero) return;
    busy = true; lockUntil = Date.now() + 3600;
    var w = W.innerWidth, h = W.innerHeight, dpr = W.devicePixelRatio || 1;
    shot.innerHTML = ''; var cl = hero.cloneNode(true); cl.style.setProperty('--p', '0');
    var v = hero.querySelector('video'), cv2 = cl.querySelector('video');
    if (v && cv2) { var c2 = D.createElement('canvas'); c2.width = w; c2.height = h;
      c2.style.cssText = 'position:absolute;inset:0;width:100%;height:100%';
      try { var g2 = c2.getContext('2d'), vw = v.videoWidth || 16, vh = v.videoHeight || 9,
            sc = Math.max(w / vw, h / vh); g2.drawImage(v, (w - vw * sc) / 2, (h - vh * sc) / 2, vw * sc, vh * sc); } catch (e) {}
      cv2.parentNode.replaceChild(c2, cv2); }
    var cue = cl.querySelector('.cue'); if (cue) cue.remove();
    shot.appendChild(cl); shot.style.clipPath = ''; shot.style.opacity = '1'; shot.style.filter = '';
    cv.width = w * dpr; cv.height = h * dpr; cx.setTransform(dpr, 0, 0, dpr, 0, 0); cv.style.opacity = '1';
    if (!noise || noise.width < w / 3) noise = makeNoise(w, h);
    ov.classList.add('on');
    m.scrollTop = startY(); showInView();                       // the next page waits underneath
    var ax = w / 2, ay = h * 0.06, R = Math.min(h * 0.86, w * 0.62), half = 0.62, T = 3000, t0 = 0;
    var bx = ax + R * 0.06, by = ay + R * 0.56, bs = R * 0.40;   // where the baby lies inside the fan
    function seg(k, a, b) { return Math.max(0, Math.min(1, (k - a) / (b - a))); }
    function frame(t) {
      if (!t0) t0 = t; var k = Math.min(1, (t - t0) / T);
      var open = ease(seg(k, 0.0, 0.22)), sweep = seg(k, 0.18, 0.62), zoom = ease(seg(k, 0.70, 1.0)), out = seg(k, 0.80, 1.0);
      shot.style.opacity = String(1 - seg(k, 0.05, 0.30));       // the first page fades behind the darkness
      cx.setTransform(dpr, 0, 0, dpr, 0, 0); cx.clearRect(0, 0, w, h);
      cx.fillStyle = 'rgba(12,18,16,' + (0.97 * (1 - out)) + ')'; cx.fillRect(0, 0, w, h);
      // camera: zoom into the baby's head at the end
      var zx = bx + (-0.30) * bs, zy = by + (-0.30) * bs, Z = 1 + zoom * 6;
      cx.translate(w / 2, h / 2); cx.scale(Z, Z); cx.translate(-(w / 2 + (zx - w / 2) * zoom), -(h / 2 + (zy - h / 2) * zoom));
      cx.globalAlpha = 1 - out;
      var r = R * (0.08 + 0.92 * open);
      // fan (the ultrasound image) with speckle inside
      cx.save(); cx.beginPath(); cx.moveTo(ax, ay); cx.arc(ax, ay, r, Math.PI / 2 - half, Math.PI / 2 + half); cx.closePath();
      cx.clip(); cx.globalAlpha = (1 - out) * 0.55; cx.drawImage(noise, 0, 0, w, h);
      cx.globalAlpha = 1 - out;
      // what the beam has already swept is "scanned"
      var beamA = (Math.PI / 2 + half) - 2 * half * sweep;     // from the left edge to the right edge
      cx.beginPath(); cx.moveTo(ax, ay); cx.arc(ax, ay, r + 2, Math.PI / 2 + half, beamA, true); cx.closePath(); cx.clip();
      // uterine wall: a bright curved band low in the fan
      cx.strokeStyle = 'rgba(230,214,186,.35)'; cx.lineWidth = R * 0.05;
      cx.beginPath(); cx.arc(ax, ay, R * 0.93, Math.PI / 2 - half * 0.95, Math.PI / 2 + half * 0.95); cx.stroke();
      var beat = Math.pow(Math.max(0, Math.sin(t / 1000 * Math.PI * 2 * 2.3)), 6);   // ~140 beats per minute
      drawBaby(cx, bx, by, bs, beat, 1);
      cx.restore();
      // gold fan edge, depth arcs and the sweeping beam
      cx.save(); cx.globalAlpha = (1 - out);
      cx.strokeStyle = 'rgba(224,164,88,.9)'; cx.lineWidth = 2; cx.shadowColor = 'rgba(224,164,88,.8)'; cx.shadowBlur = 16;
      cx.beginPath(); cx.moveTo(ax, ay); cx.arc(ax, ay, r, Math.PI / 2 - half, Math.PI / 2 + half); cx.closePath(); cx.stroke();
      cx.shadowBlur = 0; cx.lineWidth = 1;
      for (var j = 1; j <= 4; j++) { cx.strokeStyle = 'rgba(224,164,88,.16)'; cx.beginPath();
        cx.arc(ax, ay, r * j / 4.4, Math.PI / 2 - half, Math.PI / 2 + half); cx.stroke(); }
      if (sweep > 0 && sweep < 1) { cx.strokeStyle = 'rgba(250,224,170,.95)'; cx.lineWidth = 3; cx.shadowColor = 'rgba(224,164,88,1)';
        cx.shadowBlur = 24; cx.beginPath(); cx.moveTo(ax, ay); cx.lineTo(ax + r * Math.cos(beamA), ay + r * Math.sin(beamA)); cx.stroke(); }
      // probe at the apex
      cx.shadowBlur = 0; cx.fillStyle = '#e8dcc4'; cx.beginPath();
      cx.moveTo(ax - 26, ay - 40); cx.lineTo(ax + 26, ay - 40); cx.lineTo(ax + 22, ay - 6); cx.quadraticCurveTo(ax, ay + 4, ax - 22, ay - 6);
      cx.closePath(); cx.fill();
      cx.restore();
      // words under the fan
      cx.setTransform(dpr, 0, 0, dpr, 0, 0);
      var wa = seg(k, 0.42, 0.55) * (1 - seg(k, 0.70, 0.80));
      if (wa > 0) { cx.globalAlpha = wa; cx.fillStyle = '#e8dcc4'; cx.textAlign = 'center';
        cx.font = 'italic 600 ' + Math.round(Math.max(20, h * 0.032)) + 'px Fraunces, Georgia, serif';
        cx.fillText('Listening to life before birth', w / 2, Math.min(h - 40, ay + R + 54));
        cx.font = '600 11px Inter, sans-serif'; cx.fillStyle = '#c9b58f';
        cx.fillText('♥  ' + 140 + ' BPM  ·  FIRST SOUND', w / 2, Math.min(h - 16, ay + R + 82)); cx.globalAlpha = 1; }
      if (k < 1) W.requestAnimationFrame(frame);
      else { ov.classList.remove('on'); shot.innerHTML = ''; cx.clearRect(0, 0, w, h);
             setTimeout(function () { busy = false; }, 400); reveal(); }
    }
    W.requestAnimationFrame(frame);
  }
  // ---------- three more transitions: womb curtains, scan-line wipe, zoom into the monitor
  function freezeHero() {
    var hero = q('section.stMain .fs-hero'), w = W.innerWidth, h = W.innerHeight;
    var cl = hero.cloneNode(true); cl.style.setProperty('--p', '0'); cl.style.height = h + 'px';
    var v = hero.querySelector('video'), cv2 = cl.querySelector('video');
    if (v && cv2) { var c2 = D.createElement('canvas'); c2.width = w; c2.height = h;
      c2.style.cssText = 'position:absolute;inset:0;width:100%;height:100%';
      try { var g2 = c2.getContext('2d'), vw = v.videoWidth || 16, vh = v.videoHeight || 9,
            sc = Math.max(w / vw, h / vh); g2.drawImage(v, (w - vw * sc) / 2, (h - vh * sc) / 2, vw * sc, vh * sc); } catch (e) {}
      cv2.parentNode.replaceChild(c2, cv2); }
    var cue = cl.querySelector('.cue'); if (cue) cue.remove();
    return cl;
  }
  function layer(el, css) { var d = D.createElement('div'); d.style.cssText = 'position:absolute;inset:0;overflow:hidden;' + (css || '');
    if (el) d.appendChild(el); shot.appendChild(d); return d; }
  function begin() {
    var m = main(); busy = true; lockUntil = Date.now() + 3200;
    shot.innerHTML = ''; shot.style.cssText = 'background:transparent'; cv.style.opacity = '0';
    cx.setTransform(1, 0, 0, 1, 0, 0); cx.clearRect(0, 0, cv.width, cv.height);
    ov.classList.add('on'); return m;
  }
  function finish() { ov.classList.remove('on'); shot.innerHTML = ''; shot.style.cssText = ''; cv.style.opacity = '1'; cv.style.zIndex = '';
    cx.setTransform(1, 0, 0, 1, 0, 0); cx.clearRect(0, 0, cv.width, cv.height);
    setTimeout(function () { busy = false; }, 400); reveal(); }
  function run(T, step) { var t0 = 0; function f(t) { if (!t0) t0 = t; var k = Math.min(1, (t - t0) / T); step(k);
      if (k < 1) W.requestAnimationFrame(f); else finish(); } W.requestAnimationFrame(f); }
  function seg(k, a, b) { return Math.max(0, Math.min(1, (k - a) / (b - a))); }

  function curtainDown() {                       // the first page splits and parts like soft womb curtains
    var m = begin(), w = W.innerWidth, h = W.innerHeight;
    shot.style.background = 'transparent'; shot.style.perspective = '1400px';
    var glow = layer(null, 'background:radial-gradient(ellipse 38% 70% at 50% 50%, rgba(255,236,200,.95) 0%, rgba(232,190,140,.55) 35%, rgba(31,48,43,0) 75%);opacity:0;');
    var L = layer(freezeHero(), 'clip-path:inset(0 50% 0 0);transform-origin:0% 50%;');
    var Rr = layer(freezeHero(), 'clip-path:inset(0 0 0 50%);transform-origin:100% 50%;');
    var seam = layer(null, 'left:50%;right:auto;width:2px;margin-left:-1px;background:linear-gradient(180deg,transparent,#ffe9c4 30%,#ffe9c4 70%,transparent);box-shadow:0 0 40px 12px rgba(255,214,160,.7);opacity:0;');
    m.scrollTop = startY(); showInView();
    run(1900, function (k) {
      var a = seg(k, 0.0, 0.22), o = ease(seg(k, 0.18, 0.95)), f = seg(k, 0.70, 1.0);
      seam.style.opacity = String(a * (1 - seg(k, 0.3, 0.5)));
      seam.style.transform = 'scaleY(' + (0.2 + 0.8 * ease(a)) + ')';
      glow.style.opacity = String(Math.min(1, a * 1.2) * (1 - f));
      var dx = o * (w * 0.56);
      L.style.transform = 'translateX(' + (-dx) + 'px) rotateY(' + (o * 18) + 'deg)';
      Rr.style.transform = 'translateX(' + dx + 'px) rotateY(' + (-o * 18) + 'deg)';
      L.style.filter = Rr.style.filter = 'brightness(' + (1 - 0.35 * o) + ')';
      L.style.borderRadius = '0 ' + (o * 40) + '% ' + (o * 40) + '% 0 / 0 50% 50% 0';
      Rr.style.borderRadius = (o * 40) + '% 0 0 ' + (o * 40) + '% / 50% 0 0 50%';
    });
  }

  function wipeDown() {                          // a glowing scan line builds page 2 line by line
    var m = begin(), w = W.innerWidth, h = W.innerHeight;
    shot.style.background = 'transparent';
    var hv = layer(freezeHero(), '');
    var trail = layer(null, 'bottom:auto;height:0;background:linear-gradient(180deg, rgba(232,220,196,0) 0%, rgba(232,220,196,.0) 60%, rgba(224,164,88,.28) 100%);');
    var line = layer(null, 'bottom:auto;height:3px;background:linear-gradient(90deg,transparent,#e0a458 12%,#fff1d6 50%,#e0a458 88%,transparent);box-shadow:0 0 22px 6px rgba(224,164,88,.75), 0 0 80px 18px rgba(224,164,88,.25);');
    m.scrollTop = startY(); showInView();
    run(1500, function (k) {
      var y = ease(k) * (h + 6);
      hv.style.clipPath = 'inset(' + y + 'px 0 0 0)';
      line.style.top = (y - 2) + 'px'; line.style.opacity = String(1 - seg(k, 0.9, 1));
      trail.style.top = Math.max(0, y - 140) + 'px'; trail.style.height = Math.min(140, y) + 'px';
      trail.style.opacity = String(1 - seg(k, 0.85, 1));
    });
  }

  var MON = new Image(); MON.src = '__MONITOR__';
  function monitorDown() {                        // zoom into the ultrasound monitor in the video; its screen becomes page 2
    var m = begin(), w = W.innerWidth, h = W.innerHeight, dpr = W.devicePixelRatio || 1;
    cv.width = w * dpr; cv.height = h * dpr; cv.style.opacity = '1'; cv.style.zIndex = '1';
    var top = layer(freezeHero(), 'z-index:2;');
    var iw = 848, ih = 480, s0 = Math.max(w / iw, h / ih), ox = (w - iw * s0) / 2, oy = (h - ih * s0) / 2;
    var R = { x: ox + 522 * s0, y: oy + 32 * s0, w: 306 * s0, h: 212 * s0 };      // the monitor's screen in the frame
    var Z = Math.max(w / R.w, h / R.h) * 1.04, rc = { x: R.x + R.w / 2, y: R.y + R.h / 2 };
    m.scrollTop = startY(); showInView();
    run(2200, function (k) {
      var fade = seg(k, 0.0, 0.22), z = ease(seg(k, 0.12, 1.0)), hole = ease(seg(k, 0.30, 0.62));
      top.style.opacity = String(1 - fade);
      var S = 1 + (Z - 1) * z, Cx = w / 2 + (rc.x - w / 2) * z, Cy = h / 2 + (rc.y - h / 2) * z;
      cx.setTransform(dpr, 0, 0, dpr, 0, 0); cx.clearRect(0, 0, w, h);
      cx.save(); cx.translate(w / 2, h / 2); cx.scale(S, S); cx.translate(-Cx, -Cy);
      if (MON.complete && MON.naturalWidth) cx.drawImage(MON, ox, oy, iw * s0, ih * s0);
      else { cx.fillStyle = '#15211d'; cx.fillRect(ox, oy, iw * s0, ih * s0); }
      // the screen turns into a window onto page 2
      var rr = 10 * s0;
      cx.globalCompositeOperation = 'destination-out'; cx.globalAlpha = hole;
      cx.beginPath(); if (cx.roundRect) cx.roundRect(R.x, R.y, R.w, R.h, rr); else cx.rect(R.x, R.y, R.w, R.h); cx.fill();
      cx.globalCompositeOperation = 'source-over'; cx.globalAlpha = 1;
      cx.strokeStyle = 'rgba(190,220,235,' + (0.55 * (1 - seg(k, 0.75, 1))) + ')'; cx.lineWidth = 2 / S * 2;
      cx.shadowColor = 'rgba(170,210,235,.9)'; cx.shadowBlur = 24;
      cx.beginPath(); if (cx.roundRect) cx.roundRect(R.x, R.y, R.w, R.h, rr); else cx.rect(R.x, R.y, R.w, R.h); cx.stroke();
      cx.restore();
      cv.style.opacity = String(1 - seg(k, 0.88, 1.0));
    });
  }

  var MODE = '__MODE__';
  try { var qm = new URLSearchParams(W.location.search).get('t'); if (qm) MODE = qm; } catch (e) {}
  function heroClip() {            // which part of the hero video is on screen: 'scan' (mother being scanned) or 'womb'
    var v = q('section.stMain .fs-hero video'); if (!v || !v.duration) return 'scan';
    var cut = v.duration / 2;        // hero_loop = scan clip, then (cross-fading) the womb clip
    return v.currentTime < cut ? 'scan' : 'womb';
  }
  function goDown() {
    if (MODE === 'auto') return heroClip() === 'womb' ? curtainDown() : monitorDown();
    if (MODE === 'curtain') return curtainDown();
    if (MODE === 'wipe') return wipeDown();
    if (MODE === 'monitor') return monitorDown();
    return fanDown();
  }

  function upToHero() { busy = true; lockUntil = Date.now() + 1800; smooth(0, 950, function () { setTimeout(function () { busy = false; }, 400); }); }
  function isBusy() { if (busy && Date.now() > lockUntil) busy = false; return busy; }
  D.addEventListener('wheel', function (e) {
    var m = main(), hero = q('section.stMain .fs-hero'); if (!m || !hero) return;
    var h = hero.offsetHeight, y = m.scrollTop;
    if (isBusy()) { if (y < h * 1.05) e.preventDefault(); return; }
    if (e.deltaY > 2 && y < h * 0.5) { e.preventDefault(); goDown(); }
    else if (e.deltaY < -2 && y > 2 && y <= h * 1.03) { e.preventDefault(); upToHero(); }
  }, { passive: false, capture: true });
  D.addEventListener('click', function (e) {
    var a = e.target.closest && e.target.closest('section.stMain .fs-hero .cue');
    if (a) { e.preventDefault(); if (!isBusy()) goDown(); }
  }, true);
  D.addEventListener('keydown', function (e) {
    var m = main(); if (!m) return; var t = e.target || {};
    if (/INPUT|TEXTAREA|SELECT/.test(t.tagName || '') || (t.getAttribute && t.getAttribute('role') === 'slider')) return;
    if ((e.key === 'PageDown' || e.key === 'ArrowDown' || e.key === ' ') && m.scrollTop < 5 && !isBusy()) { e.preventDefault(); goDown(); }
  });
  // steps "scan in" once, the first time they come into view (never again on updates)
  var seen = {};
  function keyOf(el) { var c = [].slice.call(el.classList).filter(function (x) { return x.indexOf('st-key-') === 0; })[0];
    return c || (el.className.indexOf('fs-chap') >= 0 ? 'chap' : null); }
  function showInView() {
    var m = main(); if (!m) return; var vh = m.clientHeight;
    D.querySelectorAll('section.stMain [class*="st-key-step_"], section.stMain .fs-chap').forEach(function (el) {
      var k = keyOf(el); if (!k) return; var r = el.getBoundingClientRect();
      if (r.top < vh * 0.92) { seen[k] = 1; el.classList.remove('fs-wait'); el.classList.remove('fs-show'); }
    });
  }
  function reveal() {
    var m = main(); if (!m || busy) return; var vh = m.clientHeight;
    D.querySelectorAll('section.stMain [class*="st-key-step_"], section.stMain .fs-chap').forEach(function (el) {
      var k = keyOf(el); if (!k) return;
      var r = el.getBoundingClientRect();
      if (seen[k]) { el.classList.remove('fs-wait'); return; }
      if (r.top > vh * 0.92) el.classList.add('fs-wait');
      else { seen[k] = 1; if (el.classList.contains('fs-wait')) { el.classList.remove('fs-wait'); el.classList.add('fs-show'); } }
    });
  }
  function onScroll() {
    var m = main(), e = q('section.stMain .fs-hero'); if (!m || !e) return;
    var p = Math.max(0, Math.min(1, m.scrollTop / e.offsetHeight)); e.style.setProperty('--p', p.toFixed(3));
    reveal();
  }
  D.addEventListener('scroll', onScroll, { passive: true, capture: true });
  W.addEventListener('resize', size);
  setInterval(size, 1000);
  size(); onScroll();
})();
"""

MON_URI = data_uri(str(ASSETS / 'ai' / 'monitor_frame.jpg'), 'image/jpeg',
                   (ASSETS / 'ai' / 'monitor_frame.jpg').stat().st_mtime) if (ASSETS / 'ai' / 'monitor_frame.jpg').exists() else ''
TRANSITION = 'monitor'   # zoom into the ultrasound monitor (chosen). Others: 'auto' | 'curtain' | 'wipe' | 'baby'
                      # others: 'curtain' | 'wipe' | 'monitor' | 'baby'   (or add ?t=curtain to the page address)
FX_JS = FX_JS.replace('__MONITOR__', MON_URI).replace('__MODE__', TRANSITION)
FX = ('<script>(function(){var D=window.parent.document;if(D.getElementById("fs-fx-js"))return;'
      'var st=D.createElement("style");st.id="fs-fx-css";st.textContent=' + json.dumps(FX_CSS) + ';D.head.appendChild(st);'
      'var sc=D.createElement("script");sc.id="fs-fx-js";sc.textContent=' + json.dumps(FX_JS) + ';D.head.appendChild(sc);'
      '})();</script>')
components.html(FX, height=0)

# =====================================================================  IMAGE PROCESSING
meta = pd.read_csv(SAMPLES / 'samples.csv')
if 'kind' not in meta.columns:
    def _kind(f):
        f = f.lower()
        for k_ in ('cervix', 'brain', 'abdomen', 'femur'):
            if k_ in f:
                return k_
        return 'head' if 'hc' in f else 'upload'
    meta['kind'] = meta['file'].map(_kind)
@st.fragment
def processing_section():
    """Only this part reruns when a value changes: the hero video and the rest of the page stay untouched."""
    with st.container(key='main'):
        st.markdown('<div class="fs-chap"><div><div class="k">Image processing</div><h2>Cleaning a real ultrasound scan</h2>'
                    '<p>Upload a scan or choose one of the real samples below. The same five steps are applied, and each '
                    'step tells you what it did to <b>this</b> image and by how much.</p></div></div>', unsafe_allow_html=True)

        # ---------------- input: upload first, then samples
        with st.container(key='step_in'):
            step_text('↑', 'Input', 'Upload your scan',
                      'Any 2D ultrasound image (PNG / JPG). Take a screenshot of a scan, or download one of the real samples '
                      'below and upload it here.')
            up = st.file_uploader('Upload an ultrasound image', type=['png', 'jpg', 'jpeg', 'bmp'], key='upl',
                                  label_visibility='collapsed')
            st.markdown('<div class="fs-step" style="margin-top:18px"><div class="n">or</div><div class="w">Real samples</div>'
                        '<div class="t">Pick a real scan</div><p>Three fetal heads from HC18 (with the expert head outline) '
                        'and four maternal–fetal planes from FETAL_PLANES_DB. All are real patient scans from open research '
                        'databases.</p></div>', unsafe_allow_html=True)
            if 'sample' not in st.session_state:
                st.session_state['sample'] = '225_2HC.png'
            with st.container(key='samples'):
                PER_ROW = 4                       # 4 per row: big enough thumbnails, labels on one line
                rows_ = [meta.iloc[i:i + PER_ROW] for i in range(0, len(meta), PER_ROW)]
                for chunk in rows_:
                    cols = st.columns(PER_ROW, gap='medium')
                    for c, (_, r) in zip(cols, chunk.iterrows()):
                        with c:
                            chosen = up is None and st.session_state['sample'] == r['file']
                            src = 'FETAL PLANES DB' if 'FETAL' in str(r['source']) else str(r['source'])
                            st.markdown(f'<img class="fs-thumb{" on" if chosen else ""}" '
                                        f'src="{thumb_uri(str(SAMPLES / r["file"]))}" alt="{r["label"]}">'
                                        f'<div class="fs-samp"><b>{r["label"]}</b><span>{src}</span></div>',
                                        unsafe_allow_html=True)
                            b1, b2 = st.columns([1.35, 1], gap='small')
                            with b1:
                                if st.button('Selected ✓' if chosen else 'Use this', key='use_' + r['file'],
                                             type='primary' if chosen else 'secondary', width='stretch'):
                                    st.session_state['sample'] = r['file']
                                    st.session_state['jump'] = True
                                    st.rerun(scope='fragment')
                            with b2:
                                st.download_button('↓ Save', (SAMPLES / r['file']).read_bytes(), file_name=r['file'],
                                                   mime='image/png', key='dl_' + r['file'], width='stretch',
                                                   help='Download this scan to try the upload')
                    st.write('')

        ann = None
        if up is not None:
            img = np.asarray(Image.open(up).convert('L'), dtype=np.float64) / 255.0
            label = up.name
            row = meta[meta['file'] == up.name]
            kind = row['kind'].iloc[0] if len(row) else 'upload'
            an = SAMPLES / up.name.replace('.png', '_Annotation.png')
            if an.exists():
                a_ = np.asarray(Image.open(an).convert('L'))
                ann = a_ if a_.shape == img.shape else None
        else:
            label = st.session_state['sample']
            row = meta[meta['file'] == label]
            kind = row['kind'].iloc[0] if len(row) else 'upload'
            img = np.asarray(Image.open(SAMPLES / label).convert('L'), dtype=np.float64) / 255.0
            an = SAMPLES / label.replace('.png', '_Annotation.png')
            if an.exists():
                ann = np.asarray(Image.open(an).convert('L'))
        if max(img.shape) > 1000:
            s_ = 1000 / max(img.shape); img = ndi.zoom(img, s_, order=1); ann = None
        Wd = WORDS[kind]
        pretty = row['label'].iloc[0] if len(row) else label
        src_id = (f'upload:{up.name}:{up.size}' if up is not None else 'sample:' + label)
        st.markdown('<div id="fs-go" style="height:18px"></div>', unsafe_allow_html=True)
        # always render this (empty when not needed) so the page layout never shifts between reruns
        _j = st.session_state.pop('jump', False)
        components.html(('<script>try{var e=window.parent.document.getElementById("fs-go");'
                         'if(e){e.scrollIntoView({behavior:"smooth",block:"start"});}}catch(err){}</script>') if _j else
                        '<!-- idle -->', height=0)
        with st.container(key='step_go'):
            a, b = st.columns([1, 1.35], gap='large')
            with a:
                step_text('▶', 'Ready', 'Process this scan',
                          f'Selected: <b>{pretty}</b> ({label}, {img.shape[1]} × {img.shape[0]} pixels'
                          + (', with the expert head outline' if ann is not None else '') + '). '
                          'Press the button to run the five cleaning steps on it. Choosing another sample or uploading a '
                          'new image brings you back here.')
                done = st.session_state.get('done_for') == src_id
                go_ = st.button('Processed ✓  (scroll down to see the steps)' if done else 'Process this scan  →',
                                type='primary', key='go', width='stretch')
                if go_ and not done:
                    st.session_state['done_for'] = src_id
                    st.session_state['jump_steps'] = True
                    st.rerun(scope='fragment')
            with b:
                c1, _ = st.columns([1.4, 1])
                pic(c1, img, 'Selected scan')
        if st.session_state.get('done_for') != src_id:
            return
        st.markdown('<div id="fs-steps"></div>', unsafe_allow_html=True)
        _js = st.session_state.pop('jump_steps', False)
        components.html(('<script>try{var e=window.parent.document.getElementById("fs-steps");'
                         'if(e){setTimeout(function(){e.scrollIntoView({behavior:"smooth",block:"start"});},300);}}'
                         'catch(err){}</script>') if _js else '<!-- idle -->', height=0)
        st.markdown(f'<div style="margin:18px 0 0 64px"><span class="fs-pill">PROCESSING</span>'
                    f'<span class="fs-pill">{pretty} · {label} · {img.shape[1]}×{img.shape[0]} px</span></div>',
                    unsafe_allow_html=True)
        rule()

        # common quantities
        sector = ndi.uniform_filter(img, 15) > 0.03
        rt, rb, re_, ell = ip.make_rois(img.shape, ann)
        rt &= sector; rb &= sector; re_ &= sector
        H_, W_ = img.shape
        if ell is not None:
            cy, cx = int(np.clip(ell[1] - ell[3] * 0.95, 0, H_ - 1)), int(ell[0])
        else:
            ys, xs = np.nonzero(re_) if re_.any() else (np.array([H_ // 2]), np.array([W_ // 2]))
            cy, cx = int(np.median(ys)), int(np.median(xs))
            if re_.any():
                k = int(np.argmin(ys)); cy, cx = int(ys[k]), int(xs[k])
        zs = min(H_, W_) // 4
        y0, x0 = int(np.clip(cy - zs // 2, 0, H_ - zs)), int(np.clip(cx - zs // 2, 0, W_ - zs))

        def zoom(im):
            return ndi.zoom(im[y0:y0 + zs, x0:x0 + zs], 2.5, order=0)

        def edge_strength(im):
            g = np.hypot(ndi.sobel(im, 0), ndi.sobel(im, 1))
            return float(g[re_].mean()) if re_.any() else float(g[sector].mean())

        def hist_fig(arrs, title='Grey-level histogram (scan area only)'):
            fig = go.Figure()
            for arr, nm, colr in arrs:
                h, e = np.histogram(np.clip(arr, 0, 1)[sector] * 255, 96, (0, 255))
                fig.add_trace(go.Scatter(x=e[:-1], y=h, name=nm, fill='tozeroy', line=dict(color=colr, width=1.2)))
            return lay(fig, title, 240, 'grey level (0 = black, 255 = white)', 'pixels')

        # ---------------- step 1: the image as numbers
        with st.container(key='step_1'):
            a, b = st.columns([1, 1.35], gap='large')
            p_dark = float((img < 0.08).mean() * 100); mean_g = float(img[sector].mean() * 255)
            with a:
                step_text(1, 'Read the image', 'The scan is a grid of numbers',
                          f'This {Wd["scan"]} is <b>{W_} × {H_} = {W_ * H_:,} pixels</b>. Each pixel is one number from '
                          f'0 (black) to 255 (white): the echo strength the scanner measured at that point. '
                          f'<b>{p_dark:.0f}%</b> of the pixels are almost black: the background outside the scan and '
                          f'{Wd["dark"]}. Inside the scan the average grey level is <b>{mean_g:.0f}</b>.',
                          'Colour images are converted to one grey channel first: grey = 0.299 R + 0.587 G + 0.114 B.')
            with b:
                n = 24
                yy, xx = int(np.clip(y0 + zs // 2 - n // 2, 0, H_ - n)), int(np.clip(x0 + zs // 2 - n // 2, 0, W_ - n))
                box = np.dstack([img] * 3).copy()
                for (r0, r1, q0, q1) in [(yy - 2, yy, xx - 2, xx + n + 2), (yy + n, yy + n + 2, xx - 2, xx + n + 2),
                                         (yy - 2, yy + n + 2, xx - 2, xx), (yy - 2, yy + n + 2, xx + n, xx + n + 2)]:
                    box[max(r0, 0):max(r1, 0), max(q0, 0):max(q1, 0)] = (0.88, 0.64, 0.35)
                c1, c2 = st.columns([1.15, 1])
                pic(c1, box, f'Original: {pretty} (gold box = zoomed area)')
                patch = img[yy:yy + n, xx:xx + n]
                big = np.kron(patch, np.ones((14, 14)))
                big[::14, :] = big[::14, :] * 0.6; big[:, ::14] = big[:, ::14] * 0.6
                pic(c2, big, f'The gold box enlarged: each square is one pixel near the {Wd["edge"]}, '
                             f'grey levels {int(patch.min() * 255)}–{int(patch.max() * 255)}')
        rule()

        # ---------------- step 2: scan area
        with st.container(key='step_2'):
            a, b = st.columns([1, 1.35], gap='large')
            sec_pct = float(sector.mean() * 100)
            with a:
                step_text(2, 'Find the scan area', 'Separate the scan from the background',
                          f'Only <b>{sec_pct:.0f}%</b> of this image is the fan-shaped scan; the rest is black border, '
                          'text and calliper marks. The image is smoothed with a 15 × 15 average and every pixel above 0.03 '
                          'is kept as "scan" (very dark fluid pockets inside the scan fall below the threshold too, as the gold outline shows). Later steps (histogram, contrast) use only these pixels, so the black border does '
                          f'not distort them. Two regions are also marked to measure quality: the <b>{Wd["inner"]}</b> (green) '
                          f'and the <b>{Wd["outer"]}</b> (clay)'
                          + (', taken from the expert head outline.' if ann is not None else '.'),
                          'Mean filter → threshold → binary mask', 'mask = mean₁₅ₓ₁₅(image) > 0.03')
            with b:
                rgb = np.dstack([img] * 3).copy()
                rgb[~sector] = rgb[~sector] * 0.25 + np.array([0.65, 0.38, 0.23]) * 0.75
                rgb[rt] = 0.6 * rgb[rt] + 0.4 * np.array([0.25, 0.55, 0.40])
                rgb[rb] = 0.6 * rgb[rb] + 0.4 * np.array([0.65, 0.38, 0.23])
                edge_m = sector ^ ndi.binary_erosion(sector, iterations=3)
                outl = np.dstack([img] * 3).copy(); outl[edge_m] = (0.88, 0.64, 0.35)
                c1, c2 = st.columns(2)
                pic(c1, outl, 'Scan area found (gold outline)')
                pic(c2, rgb, f'Outside the scan tinted; {Wd["inner"]} green, {Wd["outer"]} clay')
        rule()

        # ---------------- step 3: speckle removal
        with st.container(key='step_3'):
            a, b = st.columns([1, 1.35], gap='large')
            tbox = a.container()
            with a:
                nz = st.radio('Filter', ['Median', 'Gaussian', 'Mean', 'None'], horizontal=True, key='nz')
                k = st.select_slider('Window size k (pixels)', [3, 5, 7, 9], 5, key='nzk')
            if nz == 'Median':
                d1 = ip.median_filter(img, k)
            elif nz == 'Gaussian':
                d1 = ip.gaussian_filter(img, k / 3.3)
            elif nz == 'Mean':
                d1 = ip.mean_filter(img, k)
            else:
                d1 = img.copy()
            si0, si1 = ip.speckle_index(img), ip.speckle_index(d1)
            epi1 = ip.edge_preservation_index(img, d1, re_)
            ss1 = ssim(img, d1, data_range=1)
            how = {'Median': f'Each pixel is replaced by the <b>middle value</b> of its {k}×{k} neighbourhood ({k * k} pixels). '
                             'A single bright or dark speckle grain is never the middle value, so it disappears, while a real '
                             'edge (half bright, half dark) stays where it is.',
                   'Gaussian': f'Each pixel becomes a <b>weighted average</b> of its neighbours (σ = {k / 3.3:.1f} px); close '
                               'pixels count more. Smooth, but edges soften a little.',
                   'Mean': f'Each pixel becomes the <b>plain average</b> of its {k}×{k} neighbourhood. Strong smoothing, '
                           'softest edges.',
                   'None': 'No filtering: the image passes through unchanged.'}[nz]
            with tbox:
                step_text(3, 'Noise removal', 'Remove the speckle',
                          f'The grainy texture you see in {Wd["tissue"]} is <b>speckle</b>: interference between echoes from '
                          'tiny scatterers, not real anatomy. '
                          + (f'In this scan the {nz.lower()} filter changed the speckle index from <b>{si0:.3f}</b> to '
                             f'<b>{si1:.3f}</b> ({pct(si0, si1):+.0f}%), kept the {Wd["edge"]} in place (edge preservation '
                             f'<b>{epi1:.2f}</b> of 1) and kept the structure very close to the original (SSIM <b>{ss1:.3f}</b>).'
                             if nz != 'None' else 'Pick a filter to see what it does to this scan.'),
                          how,
                          {'Median': 'out(x,y) = median{ in(i,j) : (i,j) in the k×k window }',
                           'Gaussian': 'out = in ∗ G_σ,   G_σ(x,y) ∝ exp(−(x²+y²)/2σ²)',
                           'Mean': 'out(x,y) = (1/k²) Σ in(i,j) over the k×k window', 'None': None}[nz])
            with b:
                c1, c2 = st.columns(2)
                pic(c1, zoom(img), f'Before: zoom on the {Wd["edge"]}')
                pic(c2, zoom(d1), f'After {nz.lower()} filter')
                c1, c2 = st.columns(2)
                pic(c1, img, 'Whole scan before')
                pic(c2, d1, f'Whole scan after {nz.lower()} filter')
                k1, k2, k3 = st.columns(3)
                k1.metric('Speckle index', f'{si1:.3f}', f'{pct(si0, si1):+.0f}%', delta_color='inverse')
                k2.metric('Edge kept', f'{epi1:.2f}')
                k3.metric('SSIM vs original', f'{ss1:.3f}')
        rule()

        # ---------------- step 4: contrast
        with st.container(key='step_4'):
            a, b = st.columns([1, 1.35], gap='large')
            tbox = a.container()
            with a:
                ct = st.radio('Method', ['Stretch', 'Gamma', 'Hist. equalization', 'CLAHE', 'None'], horizontal=True, key='ct')
                lohi, gam, clip = (2, 98), 0.8, 0.01
                if ct == 'Stretch':
                    lohi = st.slider('Percentiles kept (inside the scan)', 0, 100, (2, 98), key='pct')
                elif ct == 'Gamma':
                    gam = st.slider('γ (below 1 brightens dark areas)', 0.4, 2.0, 0.8, 0.05, key='gam')
                elif ct == 'CLAHE':
                    clip = st.slider('Clip limit', 0.002, 0.04, 0.01, 0.002, format='%.3f', key='clip')
            if ct == 'Stretch':   # percentiles measured inside the scan area only (the black border would hide the change)
                _lo, _hi = np.percentile(d1[sector], lohi[0]), np.percentile(d1[sector], lohi[1])
                d2 = np.clip((d1 - _lo) / (_hi - _lo + 1e-12), 0, 1)
            elif ct == 'Gamma':
                d2 = ip.gamma_correction(d1, gam)
            elif ct == 'Hist. equalization':
                d2 = ip.histogram_equalization(d1, sector)
            elif ct == 'CLAHE':
                d2 = ip.clahe(d1, clip)
            else:
                d2 = d1.copy()
            p_lo, p_hi = np.percentile(d1[sector], lohi[0]) * 255, np.percentile(d1[sector], lohi[1]) * 255
            used0 = float(np.percentile(d1[sector], 99) - np.percentile(d1[sector], 1)) * 255
            used1 = float(np.percentile(d2[sector], 99) - np.percentile(d2[sector], 1)) * 255
            cnr0 = ip.cnr(d1, rt, rb); cnr1 = ip.cnr(d2, rt, rb)
            how = {'Stretch': f'In this scan the grey levels run from <b>{p_lo:.0f}</b> ({lohi[0]}th percentile) to '
                              f'<b>{p_hi:.0f}</b> ({lohi[1]}th percentile). Those two values are mapped to 0 and 255, so '
                              f'{Wd["dark"]} gets darker and {Wd["bright"]} gets brighter.',
                   'Gamma': f'Every grey level is raised to the power γ = {gam:.2f}. '
                            + ('Dark, hypoechoic areas are lifted most.' if gam < 1 else 'Bright areas are kept, dark areas get darker.'),
                   'Hist. equalization': 'Grey levels are re-spread so that every level is used equally often (inside the scan '
                                         'only). Strong contrast, but the speckle is stretched just as much as the anatomy.',
                   'CLAHE': f'Equalization is done separately in small tiles, with a clip limit ({clip:.3f}) that stops it '
                            'over-amplifying. Shows local detail, but boosts speckle.',
                   'None': 'No contrast change.'}[ct]
            with tbox:
                step_text(4, 'Contrast enhancement', 'Use the full grey range',
                          f'After step 3 this scan uses grey levels spread over <b>{used0:.0f}</b> of the 255 available '
                          f'(1st–99th percentile, scan area only). After {ct.lower()} it uses <b>{used1:.0f}</b>. '
                          f'The contrast between the {Wd["inner"]} and the {Wd["outer"]} (CNR) went from '
                          f'<b>{cnr0:.2f}</b> to <b>{cnr1:.2f}</b>.', how,
                          {'Stretch': 'out = (in − p_low) / (p_high − p_low), clipped to [0, 1]', 'Gamma': 'out = in^γ',
                           'Hist. equalization': 'out = CDF(in)  (cumulative histogram)',
                           'CLAHE': 'out = tile-wise clipped CDF, bilinearly blended', 'None': None}[ct])
            with b:
                c1, c2 = st.columns(2)
                pic(c1, d1, 'After step 3')
                pic(c2, d2, f'After {ct.lower()}')
                c1, c2 = st.columns(2)
                pic(c1, zoom(d1), f'Zoom before: {Wd["dark"]} and {Wd["bright"]}')
                pic(c2, zoom(d2), f'Zoom after {ct.lower()}')
        rule()

        # ---------------- step 5: sharpening
        with st.container(key='step_5'):
            a, b = st.columns([1, 1.35], gap='large')
            tbox = a.container()
            with a:
                sh = st.toggle('Unsharp masking', True, key='sh')
                sg = st.slider('Blur σ (pixels)', 0.5, 5.0, 2.0, 0.25, key='sg')
                amt = st.slider('Amount k', 0.0, 2.0, 0.6, 0.1, key='amt')
            d3 = np.clip(ip.unsharp_mask(d2, sg, amt), 0, 1) if sh else d2.copy()
            es0, es1 = edge_strength(d2), edge_strength(d3)
            with tbox:
                step_text(5, 'Sharpening', f'Make the {Wd["edge"]} crisp',
                          (f'Edge strength along the {Wd["edge"]} (average Sobel gradient) rose from <b>{es0:.3f}</b> to '
                           f'<b>{es1:.3f}</b> ({pct(es0, es1):+.0f}%). ' if sh else 'Sharpening is off. ')
                          + 'Sharpening comes last, after the speckle has been removed, so it sharpens the anatomy rather than '
                            'the grain.',
                          f'<b>Unsharp masking:</b> blur a copy (σ = {sg:.2f} px), subtract it to leave only the fine detail '
                          f'(edges), and add {amt:.1f} × that detail back.',
                          'out = in + k · (in − Gaussian_σ(in))')
            with b:
                c1, c2 = st.columns(2)
                pic(c1, zoom(d2), 'Before sharpening (zoom)')
                pic(c2, zoom(d3), 'After unsharp masking (zoom)')
                c1, c2 = st.columns(2)
                pic(c1, d2, 'Whole scan before sharpening')
                pic(c2, d3, 'Whole scan after unsharp masking')
        rule()

        # ---------------- result
        with st.container(key='step_res'):
            steps_used = ' → '.join([s_ for s_ in [nz if nz != 'None' else '', ct if ct != 'None' else '',
                                                   'unsharp mask' if sh else ''] if s_]) or 'no steps'
            st.markdown('<div class="fs-step"><div class="n">✓</div><div class="w">Result</div>'
                        f'<div class="t">Original vs cleaned</div><p>Pipeline used on this scan: <b>{steps_used}</b>.</p></div>',
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            pic(c1, img, 'Original · ' + pretty)
            pic(c2, d3, 'Cleaned · ' + steps_used)
            m0 = ip.all_measures(img, img, rt, rb, re_); m1 = ip.all_measures(img, d3, rt, rb, re_)
            names = {'Speckle index (lower better)': 'Speckle index (lower = less grain)',
                     'SNR in brain (higher better)': f'SNR in the {Wd["inner"]} (higher = smoother)',
                     'CNR brain vs fluid (higher better)': f'CNR {Wd["inner"]} vs {Wd["outer"]}',
                     'Entropy (bits)': 'Entropy (bits of grey-level information)',
                     'PSNR vs original (dB)': 'PSNR vs original (dB)', 'SSIM vs original': 'SSIM vs original (1 = same structure)',
                     'Edge preservation index (EPI)': f'Edge preservation of the {Wd["edge"]} (1 = kept)'}
            tbl = pd.DataFrame({'Measure': [names.get(k_, k_) for k_ in m0],
                                'Original': ['–' if v != v else f'{v:.3f}' for v in m0.values()],
                                'Cleaned': ['–' if v != v else f'{v:.3f}' for v in m1.values()]})
            c1, c2 = st.columns([1.6, 1], gap='large')
            with c1:
                st.markdown('<div class="fs-table">' + tbl.to_html(index=False, border=0) + '</div>', unsafe_allow_html=True)
            with c2:
                buf = io.BytesIO(); Image.fromarray(u8(d3)).save(buf, format='PNG')
                st.download_button('Download cleaned image', buf.getvalue(),
                                   file_name='cleaned_' + label.rsplit('.', 1)[0] + '.png', mime='image/png', type='primary',
                                   width='stretch')
                if ann is not None and len(row) and pd.notna(row['pixel_mm'].iloc[0]):
                    xc, yc, ea, eb, th = ip.fit_ellipse_from_annotation(ann)
                    pxm = float(row['pixel_mm'].iloc[0])
                    st.metric('Head circumference (expert outline)', f'{ip.ellipse_circumference(ea, eb) * pxm:.1f} mm',
                              f'dataset value {float(row["hc_mm"].iloc[0]):.1f} mm', delta_color='off')


processing_section()


with st.container(key='end'):
    st.markdown('<div style="font-family:Fraunces,serif;font-size:26px;color:#e8dcc4;margin-bottom:6px">FIRST SOUND</div>'
                '<div style="font-size:13px;line-height:1.7">Ultrasound Acoustic Mapping and Reconstruction of Maternal, '
                'Fetal and Placental Tissues · Medical Imaging mini project · SRM Institute of Science and Technology<br>'
                'Sample scans: HC18 (van den Heuvel et al., PLoS ONE 2018, CC BY 4.0) and FETAL_PLANES_DB (Burgos-Artizzu '
                'et al., Sci. Rep. 2020). Hero video: AI-generated. Teaching software, not a medical device.</div>',
                unsafe_allow_html=True)
