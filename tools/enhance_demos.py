#!/usr/bin/env python3
"""Post-process the copied demo HTMLs for the webpage (in place, webpage/ only).

For every demo trace under webpage/demos/{mental-state,steering}/:
  1. remove the "<h2>z W=1 K=64 seed=1 ...</h2>" / steering title
  2. remove the #live-info bar ("Cluster: ... | Group: ... | Tokens: ...
     | Segment: ...") and neutralise its JS so playback still works
  3. relabel the UMAP tab as the macrostate z_t trajectory
  4. replace the ~0.5 MB baked 2D-UMAP SVG with a self-contained,
     mouse-rotatable 3D viewer of z_t (coords from webpage/demos/_coords/)

Nothing outside webpage/ is touched. Idempotent: re-running detects the
already-enhanced marker and rebuilds from the original copies if present.

Usage (from repo root):  python webpage/tools/enhance_demos.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1]
COORDS = WEB / "demos" / "_coords"
TARGETS = sorted(
    list((WEB / "demos" / "mental-state").glob("*.html"))
    + list((WEB / "demos" / "steering").glob("*.html"))
)

MARKER = "<!--ret3d-enhanced-->"

# Runtime contrast-recompute: walks every .token span, reads its bg, picks
# light text on dark bg or vice versa (rec.601 luma). Mirrors the steering
# playground's txtColor formula so the two parts of the page agree.
CONTRAST_JS = """
<script>(function(){
  function pick(rgb){
    var m = rgb.match(/\\d+/g); if(!m || m.length<3) return null;
    var r=+m[0], g=+m[1], b=+m[2];
    return (0.299*r + 0.587*g + 0.114*b) > 140 ? '#111' : '#fff';
  }
  document.querySelectorAll('.token').forEach(function(el){
    var bg = el.style.background || getComputedStyle(el).backgroundColor;
    if(!bg) return;
    var fg = pick(bg); if(!fg) return;
    el.style.color = fg;
  });
})();</script>
"""

VIEWER_CSS = """
<style>
.ret3d-cap{flex:0 0 auto;padding:8px 12px 6px;
  font:12.5px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;color:#555;}
.ret3d-cap b{color:#1d1c1a;}
.ret3d-canvas{flex:1;width:100%;min-height:360px;display:block;background:#fff;
  cursor:grab;touch-action:none;}
.ret3d-canvas:active{cursor:grabbing;}
.ret3d-foot{display:flex;align-items:center;gap:14px;flex-wrap:wrap;
  padding:4px 10px 8px;}
.ret3d-hint{font:11px/1.4 ui-monospace,Menlo,Consolas,monospace;color:#8a8678;
  display:flex;gap:14px;flex-wrap:wrap;}
.ret3d-hint i{font-style:normal;}
/* legend icons that exactly mirror the canvas markers */
.ret3d-leg-s,.ret3d-leg-e{display:inline-block;vertical-align:-1px;
  margin-right:3px;background:transparent;}
.ret3d-leg-s{width:9px;height:9px;border-radius:50%;
  border:1.5px solid #1d3a8a;}
.ret3d-leg-e{width:10px;height:10px;border:2px solid #ffe000;}
.ret3d-mode{display:inline-flex;align-items:center;gap:6px;
  font:11px/1.4 ui-monospace,Menlo,Consolas,monospace;color:#8a8678;}
.ret3d-mode i{font-style:normal;}
#ret3d-proj{display:inline-flex;gap:4px;}
.ret3d-tg{font:11px/1 -apple-system,Segoe UI,Roboto,sans-serif;font-weight:600;
  padding:3px 9px;border-radius:999px;border:1px solid #ddd;background:#fff;
  color:#555;cursor:pointer;}
.ret3d-tg.active{background:#1d1c1a;color:#fff;border-color:#1d1c1a;}
.ret3d-scrub{display:flex;align-items:center;gap:9px;padding:5px 10px 1px;}
.ret3d-scrub-lab{font:9.5px/1 -apple-system,Segoe UI,Roboto,sans-serif;
  font-weight:700;letter-spacing:.05em;text-transform:uppercase;color:#8a8678;
  white-space:nowrap;}
.ret3d-scrub input[type=range]{flex:1;min-width:60px;height:15px;
  accent-color:#1481BA;cursor:pointer;}
.ret3d-scrub-val{font:10.5px/1 ui-monospace,Menlo,Consolas,monospace;
  font-weight:700;color:#1481BA;white-space:nowrap;min-width:62px;
  text-align:right;}
#ret3d-play{flex:0 0 auto;}
.token.ret3d-current{outline:3px solid #ffe000;outline-offset:1px;
  border-radius:3px;}
</style>
"""

# Mental-state only: wider scrolling text, smaller viz, no outer scroll.
# (steering demos share the template but live in their own section.)
LAYOUT_CSS = """
<style>
body{overflow:hidden;}
.container{height:calc(100vh - 168px)!important;min-height:0;}
.text-panel{flex:0 0 55%!important;min-height:0;}
.legend-tab-content{max-height:116px;overflow:auto;}
/* State Machine tab: two sub-tabs ("This sample" / "Global state machine").
   The FSM is sized by the (stable) container HEIGHT, never the width, so
   scrubbing the trajectory cannot resize it (no scrollbar feedback). */
#tab-live{overflow:hidden;background:#fff;}
/* match the Macrostate-Trajectory tab look: no enclosing box, no grey strip.
   Pill buttons sit directly on white, separated by a single thin rule. */
#tab-live .sm-subbar{display:flex;gap:6px;padding:8px 10px;flex:0 0 auto;
  background:transparent;border:none;border-bottom:1px solid #e5e1d8;}
#tab-live .sm-sub{font:12px/1 -apple-system,Segoe UI,Roboto,sans-serif;
  padding:4px 11px;border-radius:999px;border:1px solid #ddd;background:#fff;
  color:#555;cursor:pointer;}
#tab-live .sm-sub.active{background:#1d1c1a;color:#fff;border-color:#1d1c1a;}
#tab-live .sm-subpanel{display:flex;flex-direction:column;flex:1;
  min-height:0;overflow:hidden;}
/* short explanation above each FSM */
#tab-live .sm-cap{order:0;flex:0 0 auto;padding:8px 12px 8px;
  font:12.5px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;color:#555;}
/* the dashed group "bubbles" looked unfinished — make their edges solid */
#tab-live .sm-bubble{stroke-dasharray:none;}
/* FSM frame: no inset border or rounded box (matches the Macrostate-Trajectory
   tab where the canvas sits flush on white). Identical treatment in both
   sub-panels, so Global is visually the same as Sample trajectory. */
#tab-live #sub-this > .sm-container,
#tab-live #sub-glob > .sm-container{order:1;flex:1;min-height:0;
  overflow:hidden;display:flex;justify-content:center;
  background:transparent;border:none;border-radius:0;}
/* Sample-trajectory sub-panel: FSM fills the top, the 3D-style scrubber
   sits at the very bottom (same .ret3d-scrub / .ret3d-tg styling as the
   Macrostate-Trajectory tab). */
#tab-live #sub-this > #live-controls{order:2;flex:0 0 auto;
  background:transparent;border:none;}
#tab-live .sm-svg{height:100%;width:auto;max-width:100%;margin:0 auto;}
</style>
"""

# Mental-state only: phone layout. The desktop view is a two-column
# text | visualization flex; on a narrow screen that squishes both columns
# into unreadable slivers. Below 640px we stack them (text on top, the 3D /
# state-machine view below) and keep each panel scrolling inside the frame.
MOBILE_CSS = """
<style>
/* ---- phone layout (<=640px) ---- */
@media (max-width:640px){
  html,body{overflow-x:hidden;}
  body{padding:10px!important;}
  /* Stack: reasoning text on top, the 3D / state-machine view below. Natural
     flow with fixed panel heights (the parent page makes the frame tall
     enough) so the 3D view keeps a proper near-square size instead of being
     squeezed into a thin strip. */
  .container{flex-direction:column!important;height:auto!important;
    gap:10px!important;}
  .text-panel{flex:none!important;width:100%;height:230px;min-height:0!important;
    overflow:auto;}
  .right-panel{flex:none;width:100%;}
  /* The desktop caption is long and says "on the left" (now above); hide it
     on phones so it doesn't consume the visualization's height. */
  .ret3d-cap,.sm-cap{display:none!important;}
  .tab-content{min-height:0;}
  /* Real, near-square 3D canvas (like the steering demo). flex:none avoids the
     flex-grow-in-auto-height collapse that made it a sliver. */
  .ret3d-canvas{flex:none!important;width:100%;height:290px;min-height:290px!important;}
  /* Let long LaTeX-ish token runs wrap instead of overflowing sideways. */
  .token{font-size:14px;overflow-wrap:anywhere;}
  .text-content{overflow-x:hidden;}
  /* Keep the legend compact so it doesn't dominate the top of the frame, and
     tighten the chips so the rows wrap within the width instead of clipping. */
  .legend-tab-content{max-height:76px!important;overflow:auto;}
  .legend{gap:3px 4px;}
  .legend-group{gap:2px;}
  .legend-item{font-size:10px;padding:1px 4px;margin:1px 2px;}
  .tab-bar{flex-wrap:wrap;}
  .tab-btn{padding:6px 12px;font-size:12px;}
  /* Control rows wrap rather than overflow the narrow column. */
  .ret3d-scrub,#live-controls{flex-wrap:wrap;}
  .ret3d-scrub input[type=range],#live-pos-slider{min-width:120px;width:auto;}
  /* State Machine tab parity: give the FSM a matching fixed height. */
  #tab-live .sm-subpanel{min-height:0;}
  #tab-live .sm-container{height:290px;}
  #tab-live .sm-svg{height:290px;max-height:290px;width:auto;}
}
</style>
"""

# Mental-state only: (a) sub-tab switching inside the consolidated State
# Machine tab; (b) clicking a token in the text jumps BOTH visualizations
# (3D scrubber + State-Machine position) to that token by driving their
# existing slider 'input' handlers.
MENTAL_JS = """
<script>(function(){
  var subs=document.querySelectorAll('#tab-live .sm-sub');
  var pT=document.getElementById('sub-this');
  var pG=document.getElementById('sub-glob');
  if(subs.length&&pT&&pG){
    subs.forEach(function(b){ b.addEventListener('click',function(){
      subs.forEach(function(x){ x.classList.remove('active'); });
      b.classList.add('active');
      var glob = b.getAttribute('data-sub')==='glob';
      pG.style.display = glob ? 'flex' : 'none';
      pT.style.display = glob ? 'none' : 'flex';
    }); });
  }
  // Legend-hover tooltip: same playground-style HTML as the token tooltip
  // (bold C{id} / G{id} header + name + desc), driven by mouseover delegation.
  var legendArea = document.getElementById('ltab-legend')
                 || document.querySelector('.legend-tab-content');
  var tip = document.getElementById('tooltip');
  function legendHTML(el){
    var it = el.closest && el.closest('.legend-item');
    var gp = el.closest && el.closest('.legend-group');
    if(it){
      var cid = it.dataset.cluster, ci = window.CLUSTER_NAMES &&
                                          window.CLUSTER_NAMES[cid];
      var L = ['<b>C'+cid+'</b>'];
      if(ci) L.push(ci.name +
        '<br><span style="color:#c9c6bd">'+ci.desc+'</span>');
      return L.join(' &middot; ');
    }
    if(gp){
      var gid = gp.dataset.group, gi = window.GROUP_NAMES &&
                                       window.GROUP_NAMES[gid];
      var L2 = ['<b>G'+gid+'</b>'];
      if(gi) L2.push(gi.name +
        '<br><span style="color:#c9c6bd">'+gi.desc+'</span>');
      return L2.join(' &middot; ');
    }
    return null;
  }
  if(legendArea && tip && typeof showTip === 'function'){
    legendArea.addEventListener('mouseover', function(ev){
      var html = legendHTML(ev.target);
      if(html) showTip(ev, html);
    });
    legendArea.addEventListener('mousemove', function(ev){
      if(tip.style.display!=='block') return;
      tip.style.left = (ev.clientX + 14) + 'px';
      tip.style.top  = (ev.clientY + 14) + 'px';
    });
    legendArea.addEventListener('mouseout', function(ev){
      if(!ev.relatedTarget || !legendArea.contains(ev.relatedTarget)) hideTip();
      else if(!legendHTML(ev.relatedTarget)) hideTip();
    });
  }
  var tp=document.querySelector('.text-content')
       ||document.querySelector('.text-panel');
  if(!tp) return;
  tp.addEventListener('click',function(e){
    var t=e.target.closest&&e.target.closest('.token');
    if(!t||!/^tok-\\d+$/.test(t.id)) return;
    var n=parseInt(t.id.slice(4),10);
    // Only drive the scrubber for the currently-active right-hand tab.
    var umapTab=document.getElementById('tab-umap');
    var liveTab=document.getElementById('tab-live');
    var umapOn = umapTab && umapTab.classList.contains('active');
    var liveOn = liveTab && liveTab.classList.contains('active');
    if(umapOn){
      // 3D Macrostate-Trajectory scrubber -> nearest downsampled step
      var sc=document.getElementById('ret3d-scrub'), idx=window.__retIDX;
      if(sc&&idx&&idx.length){
        var lo=0,hi=idx.length-1;
        while(lo<hi){ var mid=(lo+hi)>>1;
          if(idx[mid]<n) lo=mid+1; else hi=mid; }
        if(lo>0 && Math.abs(idx[lo-1]-n)<=Math.abs(idx[lo]-n)) lo--;
        sc.value=lo+1;
        sc.dispatchEvent(new Event('input',{bubbles:true}));
      }
    } else if(liveOn){
      // State-Machine position -> the segment containing token n
      var ps=document.getElementById('live-pos-slider'), seq=window.LIVE_SEQ;
      if(ps&&seq&&seq.length){
        var si=seq.length-1;
        for(var k=0;k<seq.length;k++){
          if(n<=seq[k][2]){ si=k; break; }
        }
        ps.value=si;
        ps.dispatchEvent(new Event('input',{bubbles:true}));
      }
    }
  });
})();</script>
"""

# __DATA__ is replaced with the per-demo coords JSON.
VIEWER_JS = r"""
<script>
(function(){
  var RET3D = __DATA__;
  var cv = document.getElementById('ret3d');
  if(!cv || !RET3D || RET3D.n < 2) return;
  var ctx = cv.getContext('2d');
  var MODE='pca', COL = RET3D.color || [], ST = RET3D.steered || [];
  var IDX = RET3D.idx || null;  // downsampled point -> original token position
  window.__retIDX = IDX;        // read by the text-token click handler
  window.__retDATA = RET3D;     // read by the new mental-state tooltip
  function getP(){ return RET3D['points_'+MODE] || RET3D.points_umap || []; }
  var P = getP();
  var N = Math.min(P.length, COL.length);
  var VN = N;
  var scrub = document.getElementById('ret3d-scrub');
  var scrubVal = document.getElementById('ret3d-scrubval');
  var playBtn = document.getElementById('ret3d-play');
  var playTimer = null, curTok = null;
  // highlight the token at the path's current (final shown) step in the text
  function markTok(){
    if(curTok){ curTok.classList.remove('ret3d-current'); curTok=null; }
    if(!IDX) return;
    var k = Math.max(0, Math.min(VN, IDX.length) - 1);
    var el = document.getElementById('tok-' + IDX[k]);
    if(el){ el.classList.add('ret3d-current'); curTok = el;
      el.scrollIntoView({block:'nearest', inline:'nearest'}); }
  }
  // total = real token count for this sample (idx[-1]+1); fall back to point count
  var TOT = IDX ? (IDX[IDX.length-1] + 1) : N;
  function setStep(v){
    VN = Math.max(2, Math.min(v|0, N));
    if(scrub) scrub.value = VN;
    if(scrubVal){
      var k = Math.max(0, Math.min(VN, IDX ? IDX.length : N) - 1);
      var cur = IDX ? (IDX[k] + 1) : VN;
      scrubVal.textContent = cur + ' / ' + TOT;
    }
    markTok();
  }
  function stopPlay(){
    if(playTimer){ clearInterval(playTimer); playTimer=null; }
    if(playBtn) playBtn.innerHTML = '&#9654;&nbsp;Live';
  }
  function syncScrub(){ if(!scrub) return; scrub.max = N;
    stopPlay(); setStep(N); }
  if(scrub){ scrub.addEventListener('input', function(){
    stopPlay(); setStep(+scrub.value); }); }
  if(playBtn){ playBtn.addEventListener('click', function(){
    if(playTimer){ stopPlay(); return; }
    if(VN >= N) setStep(2);
    playBtn.innerHTML = '&#10073;&#10073;&nbsp;Pause';
    var stepMs = Math.max(50, Math.round(25000 / N));
    playTimer = setInterval(function(){
      if(VN >= N){ stopPlay(); return; }
      setStep(VN + 1);
    }, stepMs);
  }); }
  syncScrub();
  var anySteer = false; for(var i=0;i<N;i++){ if(ST[i]){anySteer=true;break;} }
  var hint = document.getElementById('ret3d-hint');
  function setHint(){ if(hint){ hint.innerHTML =
    '<i>● coloured by computational phase</i>' +
    '<i><span class="ret3d-leg-s"></span>start' +
    '<span class="ret3d-leg-e" style="margin-left:8px;"></span>end</i>' +
    (anySteer ? '<i style="color:#1481BA;">◯ steered tokens</i>' : ''); } }
  setHint();
  document.querySelectorAll('#ret3d-proj .ret3d-tg').forEach(function(b){
    b.addEventListener('click', function(){
      document.querySelectorAll('#ret3d-proj .ret3d-tg').forEach(function(x){
        x.classList.remove('active'); });
      b.classList.add('active'); MODE=b.getAttribute('data-m');
      P=getP(); N=Math.min(P.length,COL.length); VN=N; refit(); setHint();
      syncScrub();
    });
  });

  var yaw = 0.6, pitch = -0.45, zoom = 1.0;
  var drag = false, lx = 0, ly = 0, auto = true, dpr = 1;
  var MU=[0,0,0], SC=1;

  function fit(){
    dpr = window.devicePixelRatio || 1;
    var w = cv.clientWidth || cv.parentNode.clientWidth || 600;
    var h = cv.clientHeight || 420;
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  }
  // auto-fit displayed trace (keeps global rotation axes; recenter+rescale)
  function refit(){
    if(N<2){ MU=[0,0,0]; SC=1; return; }
    MU=[0,0,0];
    for(var i=0;i<N;i++){ MU[0]+=P[i][0]; MU[1]+=P[i][1]; MU[2]+=P[i][2]; }
    MU[0]/=N; MU[1]/=N; MU[2]/=N;
    var d=[];
    for(var i=0;i<N;i++){ var a=P[i][0]-MU[0],b=P[i][1]-MU[1],c=P[i][2]-MU[2];
      d.push(Math.sqrt(a*a+b*b+c*c)); }
    d.sort(function(x,y){return x-y;});
    SC = d[Math.floor(0.9*(d.length-1))] || d[d.length-1] || 1;
  }
  function cl(v){ return v<-1.9?-1.9:(v>1.9?1.9:v); }
  function rgba(hex,a){ var r=parseInt(hex.substr(1,2),16),
    g=parseInt(hex.substr(3,2),16), b=parseInt(hex.substr(5,2),16);
    return 'rgba('+r+','+g+','+b+','+a+')'; }
  function proj(p){
    var x0=cl((p[0]-MU[0])/SC), y0=cl((p[1]-MU[1])/SC), z0=cl((p[2]-MU[2])/SC);
    var cx=Math.cos(yaw), sx=Math.sin(yaw);
    var x = x0*cx + z0*sx;
    var z = -x0*sx + z0*cx;
    var cy=Math.cos(pitch), sy=Math.sin(pitch);
    return [x, y0*cy - z*sy, y0*sy + z*cy];
  }
  function render(){
    if(cv.clientWidth && Math.abs(cv.width-Math.round(cv.clientWidth*dpr))>2) fit();
    if(SC===1 && N>1) refit();
    var W=cv.width, H=cv.height;
    ctx.clearRect(0,0,W,H);
    var R = Math.min(W,H)*0.42*zoom;
    var ox = W/2, oy = H/2;
    var pr = [];
    for(var i=0;i<N;i++){
      var q = proj(P[i]);
      pr.push([ox + q[0]*R, oy - q[1]*R, q[2]]);
    }
    // path coloured by computational phase — same palette as the dots
    var nv = Math.max(2, Math.min(VN||N, N));
    ctx.lineWidth = Math.max(1.6, dpr*1.7); ctx.lineJoin='round';
    for(var i=1;i<nv;i++){
      ctx.strokeStyle = rgba(COL[i]||'#bdbdbd', 0.6);
      ctx.beginPath(); ctx.moveTo(pr[i-1][0],pr[i-1][1]);
      ctx.lineTo(pr[i][0],pr[i][1]); ctx.stroke();
    }
    var ord = [];
    for(i=0;i<nv;i++) ord.push(i);
    ord.sort(function(a,b){return pr[a][2]-pr[b][2];});
    for(var k=0;k<ord.length;k++){
      var idx = ord[k], a = pr[idx];
      var depth = (a[2]+1.4)/2.8;
      var rad = (1.9 + 2.1*depth) * dpr * Math.sqrt(zoom);
      var al = 0.78 + 0.22*depth;  // more solid balls (was 0.34 + 0.5*depth)
      ctx.beginPath(); ctx.arc(a[0],a[1],rad,0,6.2832);
      ctx.globalAlpha = al; ctx.fillStyle = COL[idx]; ctx.fill();
      if(ST[idx]){
        ctx.globalAlpha = 0.95; ctx.lineWidth = 1.6*dpr;
        ctx.strokeStyle = '#1481BA';
        ctx.beginPath(); ctx.arc(a[0],a[1],rad+2*dpr,0,6.2832); ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
    var s = pr[0], e = pr[nv-1];
    ctx.lineWidth = 2.4*dpr; ctx.strokeStyle = '#1d3a8a';
    ctx.beginPath(); ctx.arc(s[0],s[1],6*dpr,0,6.2832); ctx.stroke();
    // current point box — same bright yellow as the text-token highlight
    // (outlined to match the □ end legend glyph)
    ctx.lineWidth = 3*dpr; ctx.strokeStyle = '#ffe000';
    ctx.strokeRect(e[0]-6*dpr, e[1]-6*dpr, 12*dpr, 12*dpr);
  }
  function loop(){
    if(auto){ yaw += 0.0024; }
    render();
    requestAnimationFrame(loop);
  }
  function pt(ev){ var r=cv.getBoundingClientRect();
    var t = ev.touches ? ev.touches[0] : ev;
    return [(t.clientX-r.left), (t.clientY-r.top)]; }
  cv.addEventListener('mousedown', function(e){ drag=true; auto=false;
    var p=pt(e); lx=p[0]; ly=p[1]; });
  window.addEventListener('mouseup', function(){ drag=false; });
  window.addEventListener('mousemove', function(e){ if(!drag) return;
    var p=pt(e); yaw += (p[0]-lx)*0.01; pitch += (p[1]-ly)*0.01;
    pitch = Math.max(-1.5, Math.min(1.5, pitch)); lx=p[0]; ly=p[1]; });
  cv.addEventListener('touchstart', function(e){ drag=true; auto=false;
    var p=pt(e); lx=p[0]; ly=p[1]; }, {passive:true});
  cv.addEventListener('touchmove', function(e){ if(!drag) return;
    var p=pt(e); yaw += (p[0]-lx)*0.01; pitch += (p[1]-ly)*0.01;
    pitch = Math.max(-1.5, Math.min(1.5, pitch)); lx=p[0]; ly=p[1];
    e.preventDefault(); }, {passive:false});
  window.addEventListener('touchend', function(){ drag=false; });
  cv.addEventListener('wheel', function(e){ e.preventDefault();
    zoom *= (e.deltaY < 0 ? 1.1 : 0.9);
    zoom = Math.max(0.3, Math.min(5, zoom)); }, {passive:false});
  cv.addEventListener('dblclick', function(){ yaw=0.6; pitch=-0.45; zoom=1; auto=true; });

  function boot(){ fit(); }
  if('ResizeObserver' in window){ new ResizeObserver(fit).observe(cv); }
  window.addEventListener('resize', fit);
  // the UMAP tab is active by default; refit when any tab is (re)shown
  document.querySelectorAll('.tab-btn').forEach(function(b){
    b.addEventListener('click', function(){ setTimeout(fit, 30); }); });
  boot();
  requestAnimationFrame(loop);
})();
</script>
"""


def balanced_div_end(html: str, open_idx: int) -> int:
    """Return index just past the </div> that closes the div opened at open_idx."""
    i = html.index(">", open_idx) + 1
    depth = 1
    pat = re.compile(r"<div\b|</div>")
    for m in pat.finditer(html, i):
        if m.group(0) == "</div>":
            depth -= 1
            if depth == 0:
                return m.end()
        else:
            depth += 1
    raise RuntimeError("unbalanced #tab-umap div")


def enhance(path: Path) -> str:
    raw = path.read_text()
    if MARKER in raw:
        return "skip (already enhanced)"

    stem = path.stem
    cj = COORDS / f"{stem}.json"
    if not cj.exists():
        raise FileNotFoundError(f"missing coords for {stem}: {cj}")
    data = cj.read_text()

    h = raw

    # (0) responsive viewport meta — without it, mobile browsers lay the
    # iframe out at a ~980px virtual width and scale it down (tiny, unusable),
    # and the mobile @media queries below never trigger.
    if "name=\"viewport\"" not in h and "name='viewport'" not in h:
        h = re.sub(
            r"(<meta charset=['\"]utf-8['\"]\s*/?>)",
            r'\1<meta name="viewport" content="width=device-width, '
            r'initial-scale=1">',
            h,
            count=1,
        )

    # (1) drop the title h2
    h = re.sub(r"<h2>.*?</h2>", "", h, count=1, flags=re.S)

    # (3) relabel the UMAP tab button
    h = re.sub(
        r'(<button class="tab-btn[^"]*" data-tab="umap">)UMAP Trajectory(</button>)',
        r"\1Macrostate Trajectory\2",
        h,
        count=1,
    )

    # (4) replace the whole #tab-umap div (baked 2D SVG) with the 3D viewer div
    m = re.search(r'<div class="tab-content[^"]*" id="tab-umap">', h)
    if not m:
        raise RuntimeError(f"no #tab-umap div in {path.name}")
    end = balanced_div_end(h, m.start())
    new_div = (
        '<div class="tab-content active" id="tab-umap">'
        '<div class="ret3d-cap">This sample’s macrostate '
        "<b>z<sub>t</sub></b> trajectory in a global 3D space shared across "
        "all examples. <i>Tip:</i> hover or click tokens in the text on the "
        "left to drive this view, and try the playback controls below the "
        "figure (drag to rotate, scroll to zoom, double-click to reset).</div>"
        '<canvas id="ret3d" class="ret3d-canvas"></canvas>'
        '<div class="ret3d-scrub">'
        '<span class="ret3d-scrub-lab">token</span>'
        '<button type="button" id="ret3d-play" class="ret3d-tg">'
        "&#9654;&nbsp;Live</button>"
        '<input type="range" id="ret3d-scrub" min="2" value="2" step="1" '
        'aria-label="Reveal the macrostate path up to this step">'
        '<span id="ret3d-scrubval" class="ret3d-scrub-val"></span></div>'
        '<div class="ret3d-foot">'
        '<span class="ret3d-hint" id="ret3d-hint"></span>'
        '<span class="ret3d-mode"><i>global</i>'
        '<span id="ret3d-proj">'
        '<button class="ret3d-tg active" data-m="pca">PCA</button>'
        '<button class="ret3d-tg" data-m="umap">UMAP</button></span>'
        '<i>space</i></span>'
        "</div>"
        "</div>"
    )
    h = h[: m.start()] + new_div + h[end:]

    # (2) remove the live-info element + neutralise its JS (keep playback alive)
    h = re.sub(r'<div id="live-info">.*?</div>', "", h, count=1, flags=re.S)
    h = h.replace(
        "var infoBar = document.getElementById('live-info');",
        "var infoBar = {};",
    )

    # (2b) state-machine Live: drop the Speed control, hardcode 700 ms playback
    h = h.replace(
        '  <label>Speed: <input type="range" id="live-speed" min="100" '
        'max="2000" value="500" step="100">  '
        '<span id="live-speed-val">500</span>ms</label>',
        "",
        1,
    )
    h = h.replace(
        "var speedSlider = document.getElementById('live-speed');",
        "var speedSlider = null;",
        1,
    )
    h = h.replace(
        "  speedSlider.addEventListener('input', function() {\n"
        "    speedVal.textContent = speedSlider.value;\n"
        "    if (playTimer !== null) {\n"
        "      clearInterval(playTimer);\n"
        "      playTimer = setInterval(advance, parseInt(speedSlider.value));\n"
        "    }\n"
        "  });\n",
        "",
        1,
    )
    # match the 3D-viewer playback feel: ~25 s total regardless of length
    h = h.replace(
        "parseInt(speedSlider.value)",
        "Math.max(50, Math.round(25000 / "
        "((window.LIVE_SEQ&&window.LIVE_SEQ.length)||36)))",
    )

    # (2c) Live playback: loop at the end instead of stopping, and auto-pause
    #      when the user drags the position slider by hand.
    h = h.replace(
        "  function advance() {\n"
        "    var v = parseInt(posSlider.value);\n"
        "    if (v >= parseInt(posSlider.max)) {\n"
        "      clearInterval(playTimer);\n"
        "      playTimer = null;\n"
        "      playBtn.textContent = '\\u25B6 Play';\n"
        "      return;\n"
        "    }\n"
        "    posSlider.value = v + 1;\n"
        "    updateLive();\n"
        "  }\n",
        "  function advance() {\n"
        "    var v = parseInt(posSlider.value);\n"
        "    if (v >= parseInt(posSlider.max)) {\n"
        "      posSlider.value = 0; updateLive(); return;\n"
        "    }\n"
        "    posSlider.value = v + 1;\n"
        "    updateLive();\n"
        "  }\n",
        1,
    )
    h = h.replace(
        "  posSlider.addEventListener('input', updateLive);\n",
        "  posSlider.addEventListener('input', function() {\n"
        "    if (playTimer !== null) {\n"
        "      clearInterval(playTimer);\n"
        "      playTimer = null;\n"
        "      playBtn.textContent = '\\u25B6 Play';\n"
        "    }\n"
        "    updateLive();\n"
        "  });\n",
        1,
    )

    # (2d) Mental-state only: collapse State Machine / Trajectory / Live into
    #      ONE "State Machine" tab with two sub-tabs — "This sample" (the live
    #      trajectory) and "Global state machine". The global FSM <svg> is
    #      MOVED (not cloned) out of tab-sm so there are no duplicate IDs.
    if "mental-state" in path.parts:
        h = h.replace(
            '  <button class="tab-btn" data-tab="sm">State Machine</button>'
            '  <button class="tab-btn" data-tab="traj">Trajectory</button>'
            '  <button class="tab-btn" data-tab="live">Live</button>',
            '  <button class="tab-btn" data-tab="live">State Machine</button>',
            1,
        )
        # cut the global FSM <svg> out of tab-sm (svgs aren't nested here)
        sm_i = h.index('id="tab-sm"')
        gs = h.index('<svg class="sm-svg"', sm_i)
        ge = h.index("</svg>", gs) + len("</svg>")
        global_svg = h[gs:ge]
        h = h[:gs] + h[ge:]  # remove it from tab-sm (now unreachable anyway)
        # wrap tab-live's existing content as the "This sample" sub-panel and
        # append a "Global state machine" sub-panel holding the moved svg
        m_tl = re.search(r'<div class="tab-content[^"]*" id="tab-live">', h)
        tl_open_end = h.index(">", m_tl.start()) + 1
        tl_close = balanced_div_end(h, m_tl.start())  # past closing </div>
        open_tag = h[m_tl.start():tl_open_end]
        inner = h[tl_open_end:tl_close - len("</div>")]
        subbar = (
            '<div class="sm-subbar">'
            '<button type="button" class="sm-sub active" data-sub="this">'
            "Sample trajectory</button>"
            '<button type="button" class="sm-sub" data-sub="glob">'
            "Global state machine</button></div>"
        )
        # Per-trace paper-figure reference for the Sample-trajectory caption.
        # Each test doc maps to a different held-out NuminaMath sample shown
        # in the paper (Section 4.1, Fig 1 + Appendix F, Figs 8, 10–12).
        fig_ref = {
            "trace-1-function-range":
                "Figure 1 in the paper (full legend: Figure 8, Appendix F)",
            "trace-2-hyperbola":
                "Figure 11, Appendix F in the paper",
            "trace-3-statistics":
                "Figure 12, Appendix F in the paper",
            "trace-4-word-problem":
                "Figure 10, Appendix F in the paper",
        }.get(path.stem, "Figure 1 in the paper")
        cap_this = (
            '<div class="sm-cap">This sample’s computational-phase '
            "trajectory through the learned state machine. The current "
            "phase and the transition into it are highlighted; the slider "
            "below steps through phases. <i>Tip:</i> hover or click tokens "
            "in the text on the left to drive this view. See "
            + fig_ref + ".</div>"
        )
        cap_glob = (
            '<div class="sm-cap">The learned state machine across the test '
            "set — each node is a cluster (computational phase), "
            "edges are the most common transitions between them. The "
            "clustering and state-machine pipeline is described in "
            "Appendix E of the paper.</div>"
        )
        new_tab_live = (
            open_tag
            + subbar
            + '<div id="sub-this" class="sm-subpanel">'
            + cap_this + inner + "</div>"
            + '<div id="sub-glob" class="sm-subpanel" style="display:none">'
            + cap_glob
            + '<div class="sm-container">' + global_svg + "</div>"
            + "</div></div>"
        )
        h = h[:m_tl.start()] + new_tab_live + h[tl_close:]

        # (2e) "This sample" view: no window; rebuild the position bar to
        #      match the 3D scrubber (same .ret3d-scrub / .ret3d-tg look,
        #      pinned to the bottom); show only the current token + the
        #      single arrow INTO the current cluster.
        pos_max = re.search(
            r'id="live-pos-slider" min="0" max="(\d+)"', h
        ).group(1)
        h = re.sub(
            r'<div id="live-controls">.*?</div>',
            '<div id="live-controls" class="ret3d-scrub">'
            '<span class="ret3d-scrub-lab">position</span>'
            '<button type="button" id="live-play-btn" class="ret3d-tg">'
            "&#9654;&nbsp;Live</button>"
            '<input type="range" id="live-pos-slider" min="0" max="'
            + pos_max + '" value="0" step="1">'
            '<span id="live-pos-val" class="ret3d-scrub-val">0 / '
            + pos_max + "</span></div>",
            h,
            count=1,
            flags=re.S,
        )
        # the window slider is gone — neutralise its JS references
        h = h.replace(
            "var wSlider = document.getElementById('live-w-slider');",
            "var wSlider = null;", 1,
        )
        h = h.replace(
            "var wVal = document.getElementById('live-w-val');",
            "var wVal = {};", 1,
        )
        h = h.replace(
            "    var w = parseInt(wSlider.value);\n", "    var w = 0;\n", 1,
        )
        h = h.replace(
            "  wSlider.addEventListener('input', updateLive);\n", "", 1,
        )
        # drop the ±window token highlighting entirely
        ws = h.index("// Window segments")
        h = h[:ws] + h[h.index("// Scroll current into view", ws):]
        # replace the whole arrow section with a single incoming arrow
        a7 = h.index("// 7. Draw transition arrows")
        a8 = h.index("// 8. Update info bar", a7)
        h = h[:a7] + (
            "// 7. Arrow toward the current token (incoming transition)\n"
            "    if (pos >= 1) {\n"
            "      var _ps=String(LIVE_SEQ[pos-1][0]),"
            " _pd=String(LIVE_SEQ[pos][0]);\n"
            "      if (_ps !== _pd) {\n"
            "        var _d = liveBezier(_ps, _pd);\n"
            "        if (_d) {\n"
            "          var _pa = document.createElementNS("
            "'http://www.w3.org/2000/svg','path');\n"
            "          _pa.setAttribute('d', _d);\n"
            "          _pa.setAttribute('stroke', '#1481BA');\n"
            "          _pa.setAttribute('stroke-width', '2.5');\n"
            "          _pa.setAttribute('fill', 'none');\n"
            "          _pa.setAttribute('marker-end',"
            " 'url(#live-arrow-past)');\n"
            "          edgesG.appendChild(_pa);\n"
            "        }\n"
            "      }\n"
            "    }\n"
            "    "
        ) + h[a8:]
        # expose LIVE_SEQ so the text-token click handler can map tokens
        h = h.replace(
            "var LIVE_SEQ=", "var LIVE_SEQ=window.LIVE_SEQ=", 1,
        )
        # match the 3D play button wording ("Live" instead of "Play")
        h = h.replace("'\\u25B6 Play'", "'\\u25B6 Live'")
        # counter format "pos / max" — matches the Macrostate-Trajectory counter
        h = h.replace(
            "posVal.textContent = pos;",
            "posVal.textContent = pos + ' / ' + (LIVE_SEQ.length-1);",
            1,
        )
        # token backgrounds: drop the per-token alpha (was margin-shaded) so
        # each token gets the full cluster colour. The pristine inline style
        # is the only rgba() in the doc; legend pills use solid hex.
        h = re.sub(
            r"background:rgba\((\d+),(\d+),(\d+),[0-9.]+\)",
            r"background:rgb(\1,\2,\3)",
            h,
        )

    # (mental-state) restyle the token-hover tooltip to the same look as the
    # steering playground: bold "tok N" header, C{id} · name + desc, G{id} ·
    # name + desc, + cos to assigned centre + margin to runner-up. The
    # existing demo built a plain-text "G7: name | C18: name" suffix; we
    # rewrite the build block + flip showTip to innerHTML.
    if "mental-state" in path.parts:
        h = h.replace(
            "tooltip.textContent = text;",
            "tooltip.innerHTML = text;",
            1,
        )
        new_block = (
            "// Build tooltip — playground-style (HTML)\n"
            "    var cid = sp.dataset.cluster, gid = sp.dataset.group;\n"
            "    var ci = (cid!==undefined && CLUSTER_NAMES[cid]) || null;\n"
            "    var gi = (gid!==undefined && GROUP_NAMES[gid]) || null;\n"
            "    var D = window.__retDATA || {};\n"
            "    var tIdx = parseInt(sp.id.replace('tok-',''));\n"
            "    var co = D.tok_cos ? D.tok_cos[tIdx] : null;\n"
            "    var mg = D.tok_mg ? D.tok_mg[tIdx] : null;\n"
            "    var L = ['<b>tok '+tIdx+'</b>'];\n"
            "    if (ci) L.push('<b>C'+cid+'</b> &middot; '+ci.name+"
            "'<br><span style=\"color:#c9c6bd\">'+ci.desc+'</span>');\n"
            "    if (gi) L.push('<b>G'+gid+'</b> &middot; '+gi.name+"
            "'<br><span style=\"color:#c9c6bd\">'+gi.desc+'</span>');\n"
            "    if (co!=null && mg!=null) L.push("
            "'<span style=\"color:#c9c6bd\">cos '+(+co).toFixed(3)+"
            "' &middot; margin '+(+mg).toFixed(3)+'</span>');\n"
            "    showTip(ev, L.join('<br>'));"
        )
        h = re.sub(
            r"//\s*Build tooltip with optional cluster/group names"
            r".*?showTip\(ev,\s*tipText\);",
            new_block,
            h, count=1, flags=re.S,
        )

    # inject viewer CSS + JS (+ marker); layout overrides are mental-state only
    h = h.replace("</head>", VIEWER_CSS + "</head>", 1)
    if "mental-state" in path.parts:
        h = h.replace("</head>", LAYOUT_CSS + MOBILE_CSS + "</head>", 1)
    h = h.replace(
        "</body>", VIEWER_JS.replace("__DATA__", data) + MARKER + "</body>", 1
    )
    if "mental-state" in path.parts:
        h = h.replace("</body>", MENTAL_JS + "</body>", 1)

    # Recompute per-token foreground from bg luminance at load time. The
    # pristine HTMLs bake an inconsistent contrast picker (same cluster bg
    # ends up at both #fff and #000); this snippet enforces a uniform
    # rec.601-luma rule so identical bg always gets identical fg — same
    # formula the steering playground uses in JS.
    h = h.replace("</body>", CONTRAST_JS + "</body>", 1)

    path.write_text(h)
    return f"ok ({len(raw)//1024} KB -> {len(h)//1024} KB)"


def main() -> None:
    for p in TARGETS:
        print(f"  {p.relative_to(WEB)}: {enhance(p)}")
    print(f"\nEnhanced {len(TARGETS)} demo HTMLs.")


if __name__ == "__main__":
    main()
