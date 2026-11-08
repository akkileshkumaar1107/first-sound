"""Full-screen hero for the FIRST SOUND site: looping background videos (scan first, then the
womb), the title and a scroll cue. Rendered in its own frame so the video and fonts work everywhere."""
import json

CSS = """
html,body{margin:0;height:100%;background:#15211d;overflow:hidden}
*{box-sizing:border-box}
.hero{position:relative;height:100vh;width:100%;overflow:hidden;color:#e8dcc4;
  font-family:Inter,system-ui,-apple-system,sans-serif;background:#15211d}
.bg{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;opacity:0;transition:opacity 1.2s ease}
.bg.on{opacity:1}
.veil{position:absolute;inset:0;background:
  linear-gradient(90deg,rgba(15,24,21,.92) 0%,rgba(15,24,21,.78) 34%,rgba(15,24,21,.25) 62%,rgba(15,24,21,.05) 100%),
  linear-gradient(0deg,rgba(15,24,21,.85) 0%,rgba(15,24,21,0) 38%)}
.content{position:absolute;left:clamp(24px,7vw,140px);top:50%;transform:translateY(-46%);width:min(680px,86vw);
  animation:up 1s cubic-bezier(.2,.7,.2,1) .2s both}
@keyframes up{from{opacity:0;transform:translateY(-40%)}to{opacity:1;transform:translateY(-46%)}}
.eyebrow{font-weight:700;font-size:12px;letter-spacing:.26em;text-transform:uppercase;color:#c9b58f;margin-bottom:18px}
h1{font-family:Fraunces,Georgia,serif;font-weight:600;font-size:clamp(54px,8.4vw,128px);line-height:.92;
  letter-spacing:-.02em;margin:0 0 14px;color:#f1e7d3}
.tag{font-family:Fraunces,Georgia,serif;font-style:italic;font-size:clamp(18px,1.9vw,26px);color:#d9b98a;margin-bottom:18px}
p{font-size:clamp(14px,1.15vw,16.5px);line-height:1.7;color:#ddd1b9;margin:0 0 28px;max-width:560px}
.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:22px;max-width:600px}
.stats div{border-top:1px solid rgba(232,220,196,.28);padding-top:12px;min-width:0}
.stats b{display:block;font-family:Fraunces,Georgia,serif;font-weight:600;font-size:clamp(22px,2.2vw,30px);color:#f1e7d3;line-height:1.1}
.stats span{display:block;margin-top:6px;font-size:10.5px;line-height:1.35;letter-spacing:.09em;text-transform:uppercase;color:#bfae8c}
.cue{position:absolute;left:50%;bottom:22px;transform:translateX(-50%);display:flex;flex-direction:column;align-items:center;gap:8px;
  font-size:11px;font-weight:600;letter-spacing:.24em;text-transform:uppercase;color:#d9c7a3;cursor:pointer;
  background:none;border:0;font-family:inherit}
.cue i{width:22px;height:36px;border:1.5px solid rgba(232,220,196,.6);border-radius:12px;position:relative}
.cue i:after{content:"";position:absolute;left:50%;top:7px;width:3px;height:7px;border-radius:2px;background:#e8dcc4;
  transform:translateX(-50%);animation:dot 1.6s ease-in-out infinite}
@keyframes dot{0%{opacity:0;transform:translate(-50%,0)}40%{opacity:1}100%{opacity:0;transform:translate(-50%,12px)}}
.credit{position:absolute;right:20px;bottom:16px;font-size:10.5px;color:rgba(232,220,196,.55);letter-spacing:.04em}
@media (max-width:700px){.stats{grid-template-columns:repeat(2,minmax(0,1fr))}.veil{background:rgba(15,24,21,.72)}}
@media (prefers-reduced-motion:reduce){.content,.cue i:after{animation:none}}
"""


def hero_html(videos, poster=None, n_images=0):
    """videos: list of URLs (data URIs) played in order and looped; poster: still image URL."""
    vids = ''
    if videos:
        vids = ''.join(
            '<video class="bg%s" muted playsinline preload="auto"%s src="%s"></video>'
            % (' on' if i == 0 else '', (' poster="%s"' % poster) if (poster and i == 0) else '', v)
            for i, v in enumerate(videos))
    elif poster:
        vids = '<div class="bg on" style="background:url(%s) center/cover"></div>' % poster
    js = """
<script>
(function(){ function fit(){ try{ var f=window.frameElement, h=window.parent.innerHeight;
    if(f && h){ f.style.height=h+'px'; var p=f.parentElement; for(var k=0;k<4&&p;k++){ p.style.height=h+'px'; p=p.parentElement; } } }catch(e){} }
  fit(); try{ window.parent.addEventListener('resize', fit); }catch(e){} })();
(function(){
  var v=[].slice.call(document.querySelectorAll('video.bg')); if(!v.length) return; var i=0;
  function play(k){ v.forEach(function(e,j){ e.classList.toggle('on', j===k); });
    try{ v[k].currentTime=0; var p=v[k].play(); if(p&&p.catch) p.catch(function(){}); }catch(e){} }
  v.forEach(function(e,j){ e.addEventListener('ended', function(){ i=(j+1)%v.length; play(i); }); });
  if(v.length===1) v[0].loop=true;
  play(0);
})();
function goDown(){
  try{ var d=window.parent.document, t=d.getElementById('fs-start');
       if(t){ t.scrollIntoView({behavior:'smooth', block:'start'}); return; } }catch(e){}
  try{ window.parent.scrollBy({top: window.innerHeight, behavior:'smooth'}); }catch(e){}
}
</script>"""
    body = """
<section class="hero">
  %s
  <div class="veil"></div>
  <div class="content">
    <div class="eyebrow">Medical imaging &middot; Obstetric ultrasound</div>
    <h1>FIRST<br>SOUND</h1>
    <div class="tag">Listening to Life Before Birth</div>
    <p>Ultrasound acoustic mapping and reconstruction of maternal, fetal and placental tissues:
       how raw echoes are reconstructed into an image, how each echo signal is cleaned, and how a real
       fetal scan is processed step by step.</p>
    <div class="stats">
      <div><b>%s</b><span>real scans</span></div>
      <div><b>8</b><span>reconstruction steps</span></div>
      <div><b>5</b><span>signal steps</span></div>
      <div><b>4</b><span>image cleaning steps</span></div>
    </div>
  </div>
  <button class="cue" onclick="goDown()"><i></i>Scroll</button>
  <div class="credit">AI-generated video</div>
</section>""" % (vids, '{:,}'.format(n_images) if n_images else '13,399')
    return ('<!doctype html><html><head><meta charset="utf-8">'
            '<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,600;1,9..144,400'
            '&family=Inter:wght@400;600;700&display=swap" rel="stylesheet">'
            '<style>' + CSS + '</style></head><body>' + body + js + '</body></html>')
