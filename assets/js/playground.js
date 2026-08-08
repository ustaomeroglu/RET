/* ============================================================
   Steering playground — knob-driven grid viewer.
   file:// friendly: data via injected <script> JSONP.
   Multi-prompt · PCA/UMAP toggle · pale baseline overlay.
   ============================================================ */
(function () {
  "use strict";
  var ROOT = "demos/steering_grid/";
  var BASE = ROOT;                    // set per selected prompt
  var PROMPTS = null, MAN = null, curKey = null, cache = {}, viewer = null;
  var sel = { pid: null, target: null, strength: 0.5, start: 0, dur: 20,
              proj: "pca", overlay: true };
  var ctrl = {};

  window.__RETGRID_PROMPTS__ = function (list) {
    PROMPTS = list;
    var pe = document.getElementById("pg-prompt");
    pe.innerHTML = "";
    list.forEach(function (p) {
      var o = document.createElement("option");
      // user: keep only the headline title — drop the "· Fig 6 / Fig 24"-
      // style suffix AND any trailing parenthetical like "(Figs 27-29)" /
      // '("Rs.")' so the option list reads cleanly.
      o.value = p.pid;
      o.textContent = p.label
        .split("·")[0]
        .replace(/\s*\([^)]*\)\s*$/, "")
        .trim();
      pe.appendChild(o);
    });
    pe.addEventListener("change", function () { loadPrompt(pe.value); });
    loadPrompt(list[0].pid);
  };
  window.__RETGRID_MANIFEST__ = function (m) {
    if (m.pid !== sel.pid) return;
    MAN = m;
    // A prompt may declare an "open-on" recipe (manifest.defaults). This
    // manifest only (re)loads on a prompt switch, so a declared default
    // wins here (e.g. figs2729 always opens on the paper's Fig-27 run:
    // C18, s0.5, p0, d600), falling back to carry-over only when the
    // prompt declares no default for that knob.
    var df = m.defaults || {};
    function snap(v, arr, dflt) {
      return arr.indexOf(v) >= 0 ? v : (arr.indexOf(dflt) >= 0 ? dflt : arr[0]);
    }
    function pick(dfv, cur, arr, lit) {
      if (dfv != null && arr.indexOf(dfv) >= 0) return dfv;   // declared default
      return snap(cur, arr, lit);                             // else carry-over
    }
    if (df.target && m.targets.some(function (t) { return t.key === df.target; }))
      sel.target = df.target;
    else if (!m.targets.some(function (t) { return t.key === sel.target; }))
      sel.target = m.targets[0].key;
    sel.strength = pick(df.strength, sel.strength, strengthsFor(sel.target), 0.5);
    sel.start = pick(df.start, sel.start, m.starts, 0);
    sel.dur = pick(df.dur, sel.dur, m.durations, 20);
    buildUI(); select();
  };
  window.__RETGRID__ = function (k, d) {
    cache[sel.pid + "/" + k] = d;
    if (k === curKey) render();
    if (k === "baseline" && sel.overlay) render();   // overlay just arrived
  };

  function inject(src, onerr) {
    var s = document.createElement("script");
    s.src = src; s.async = false; if (onerr) s.onerror = onerr;
    document.head.appendChild(s);
  }
  function loadPrompt(pid) {
    sel.pid = pid; BASE = ROOT + pid + "/"; MAN = null;
    document.getElementById("pg-desc").textContent = "Loading prompt grid…";
    inject(BASE + "manifest.js");
  }
  function esc(t) {
    return t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
  function gstr(v) { return String(v).replace(".", "p"); }
  // Strength axis for one target: a target may restrict it (e.g. C18/C27
  // drop s=1). 0 == baseline is always prepended.
  function strengthsFor(tk) {
    var t = (MAN.targets || []).filter(function (x) { return x.key === tk; })[0];
    var base = (t && t.strengths) ? t.strengths
      : MAN.strengths.filter(function (v) { return v !== 0; });
    return [0].concat(base);
  }
  // Duration axis filtered by per-(target,strength) omissions
  // (manifest.cells_omit, e.g. drop C27_s0p5_*_d300 per the user spec).
  function durationsFor(tk, strength) {
    var all = MAN.durations || [];
    var omit = MAN.cells_omit || [];
    if (!omit.length || Number(strength) === 0) return all;
    return all.filter(function (d) {
      var k = tk + "_s" + gstr(strength) + "_p" + sel.start + "_d" + d;
      return omit.indexOf(k) < 0;
    });
  }
  function cellKey(strength) {
    if (Number(strength) === 0) return "baseline";
    return sel.target + "_s" + gstr(strength) + "_p" + sel.start + "_d" + sel.dur;
  }
  // Per-bg contrast picker: light backgrounds get dark text, dark
  // backgrounds get light text. Computed from the actual bg (rec. 601
  // luma) so identical-bg tokens always get identical fg — the same
  // recipe is mirrored at load time onto the mental-state demo HTMLs.
  function txtColor(hex) {
    var r = parseInt(hex.substr(1, 2), 16), g = parseInt(hex.substr(3, 2), 16),
        b = parseInt(hex.substr(5, 2), 16);
    return (0.299 * r + 0.587 * g + 0.114 * b) > 140 ? "#111" : "#fff";
  }

  /* shared hover tooltip */
  var tip = document.createElement("div");
  tip.className = "pg-tip"; tip.style.display = "none";
  document.body.appendChild(tip);
  function bindTip(el, htmlText) {
    el.addEventListener("mouseenter", function () {
      tip.innerHTML = htmlText; tip.style.display = "block";
    });
    el.addEventListener("mousemove", function (e) {
      var pad = 14;
      tip.style.left = Math.min(e.clientX + pad,
        window.innerWidth - tip.offsetWidth - 8) + "px";
      tip.style.top = (e.clientY + pad) + "px";
    });
    el.addEventListener("mouseleave", function () {
      tip.style.display = "none";
    });
  }

  /* compact square buttons (Target, Duration, Projection) */
  function sq(holder, items, cur, onpick) {
    holder.innerHTML = ""; var btns = [];
    items.forEach(function (it) {
      var b = document.createElement("button");
      b.className = "pg-sq" + (it.cls ? " " + it.cls : "") +
        (it.v === cur ? " active" : "");
      b.textContent = it.label;
      if (it.tip) bindTip(b, it.tip);
      b.addEventListener("click", function () {
        btns.forEach(function (x) { x.classList.remove("active"); });
        b.classList.add("active"); onpick(it.v);
      });
      btns.push(b); holder.appendChild(b);
    });
  }
  /* vertical snapping slider with label + live value */
  function vslider(holder, label, vals, cur, fmt, onpick) {
    holder.innerHTML = "";
    var lab = document.createElement("div");
    lab.className = "pg-vlab"; lab.textContent = label;
    var inp = document.createElement("input");
    inp.type = "range"; inp.min = 0; inp.max = vals.length - 1; inp.step = 1;
    inp.className = "pg-vrange";
    var i = vals.indexOf(cur); inp.value = i < 0 ? 0 : i;
    var out = document.createElement("div");
    out.className = "pg-vval"; out.textContent = fmt(vals[inp.value]);
    inp.addEventListener("input", function () {
      out.textContent = fmt(vals[inp.value]); });
    inp.addEventListener("change", function () { onpick(vals[inp.value]); });
    holder.appendChild(lab); holder.appendChild(inp); holder.appendChild(out);
  }

  function tgtTip(t) {
    var away = (t.kind === "avoid_C" || t.kind === "avoid_G");
    var name = t.label.replace(/^[→✕]\s*/, "");
    return "<b>Steering " + (away ? "away from" : "toward") + "</b> " +
      name + "<br><span>" + t.desc + "</span>";
  }
  function buildUI() {
    sq(document.getElementById("pg-target"),
      MAN.targets.map(function (t) {
        var away = (t.kind === "avoid_C" || t.kind === "avoid_G");
        return { v: t.key,
          label: (away ? "✕ " : "→ ") + t.key.replace(/^av/, ""),
          cls: away ? "avoid" : "toward", tip: tgtTip(t) };
      }), sel.target, function (v) {
        sel.target = v;
        var ss = strengthsFor(v);          // target may restrict strengths
        if (ss.indexOf(sel.strength) < 0)
          sel.strength = ss.indexOf(0.5) >= 0 ? 0.5 : ss[ss.length - 1];
        // and the new (target,strength) may omit the current duration
        var ds = durationsFor(v, sel.strength);
        if (ds.indexOf(sel.dur) < 0) sel.dur = ds[ds.length - 1];
        buildUI(); select();               // rebuild so the slider re-snaps
      });
    vslider(document.getElementById("pg-strength"), "Strength",
      strengthsFor(sel.target), sel.strength,
      function (v) { return Number(v) === 0 ? "0 · baseline" : String(v); },
      function (v) {
        sel.strength = v;
        // strength change may shrink the duration axis (cells_omit), so
        // rebuild the UI and re-snap sel.dur into what's still allowed.
        var ds = durationsFor(sel.target, sel.strength);
        if (ds.indexOf(sel.dur) < 0) sel.dur = ds[ds.length - 1];
        toggleKnobs(); buildUI(); select();
      });
    vslider(document.getElementById("pg-start"), "Start tok",
      MAN.starts, sel.start, function (v) { return "tok " + v; },
      function (v) { sel.start = v; select(); });
    vslider(document.getElementById("pg-dur"), "Duration",
      durationsFor(sel.target, sel.strength), sel.dur,
      function (v) { return v + " tok"; },
      function (v) { sel.dur = v; select(); });
    // mental-state-style projection toggle: "global [UMAP] [PCA] space"
    // with pill buttons (active is filled, inactive outlined).
    var projHolder = document.getElementById("pg-proj");
    projHolder.classList.add("ret3d-mode");
    projHolder.innerHTML =
      '<i class="ret3d-mode-lab">global</i>' +
      ['pca', 'umap'].map(function (m) {
        return '<button type="button" class="ret3d-tg' +
          (sel.proj === m ? ' active' : '') +
          '" data-m="' + m + '">' + m.toUpperCase() + '</button>';
      }).join('') +
      '<i class="ret3d-mode-lab">space</i>';
    [].forEach.call(projHolder.querySelectorAll('.ret3d-tg'), function (b) {
      b.addEventListener('click', function () {
        sel.proj = b.getAttribute('data-m');
        [].forEach.call(projHolder.querySelectorAll('.ret3d-tg'),
          function (x) { x.classList.toggle('active', x === b); });
        render();
      });
    });
    var ov = document.getElementById("pg-overlay");
    ov.checked = sel.overlay;
    ov.onchange = function () { sel.overlay = ov.checked; select(); };
    toggleKnobs();
    reorgLayout();
  }

  /* One-time DOM layout tweak: move the 3D-view toggle + baseline-overlay
     checkbox out of the right control column and under the canvas's scrub
     row (next to the play button), mirroring the mental-state demos. */
  function reorgLayout() {
    var scrub = document.querySelector(".pg-mid .ret3d-scrub");
    if (!scrub) return;
    // user: rename "step" to "tok" (matches the right sliders' "TOK" usage)
    var slab = scrub.querySelector(".ret3d-scrub-lab");
    if (slab) slab.textContent = "reveal to tok";
    // user: rewrite the canvas caption to drop the em dash
    var cap = document.querySelector(".pg-mid .ret3d-cap");
    if (cap) cap.innerHTML = "Macrostate <b>z<sub>t</sub></b> trajectory in " +
      "one global space. Drag to rotate, scroll to zoom, double-click to reset.";

    // user: move the Prompt selector to a top demo-bar styled like the
    // mental-state "Choose sample" picker.
    var stem = document.getElementById("steer-pg");
    var promptSel = document.getElementById("pg-prompt");
    if (stem && promptSel) {
      var topbar = stem.querySelector(".pg-topbar");
      if (!topbar) {
        topbar = document.createElement("div");
        topbar.className = "demo-bar pg-topbar";
        stem.insertBefore(topbar, stem.querySelector(".pg-main"));
      }
      var promptRow = promptSel.closest(".pg-row");
      if (promptRow && promptRow.parentNode !== topbar) {
        promptRow.classList.add("demo-pick");
        var pLab = promptRow.querySelector(".pg-lab");
        if (pLab) {
          pLab.classList.add("demo-pick-lab");
          pLab.textContent = "Choose sample";
        }
        promptSel.classList.add("demo-pick-sel");
        topbar.appendChild(promptRow);
      }
      // "Open fullscreen ↗" on the right of the topbar — wired below by
      // wireFullscreenLinks() so it behaves the same as the mental-state
      // link (Fullscreen API on its container, no new tab).
      if (!topbar.querySelector(".fs-link")) {
        var fs = document.createElement("a");
        fs.className = "fs-link"; fs.href = "#";
        fs.innerHTML = "Open fullscreen&nbsp;↗";
        topbar.appendChild(fs);
      }
    }
    // play / pause button (inject once into the scrub row)
    if (!document.getElementById("pg3d-play")) {
      var b = document.createElement("button");
      b.id = "pg3d-play"; b.type = "button"; b.className = "ret3d-tg";
      b.innerHTML = "&#9654;&nbsp;Live";
      scrub.insertBefore(b, scrub.querySelector("input[type=range]"));
      b.addEventListener("click", togglePlay);
    }
    // 3D-view + overlay row, anchored AFTER the scrub bar
    var bar = document.querySelector(".pg-mid .ret3d-bar");
    if (!bar) {
      bar = document.createElement("div");
      bar.className = "ret3d-bar";
      scrub.parentNode.insertBefore(bar, scrub.nextSibling);
    }
    var projRow = document.getElementById("pg-proj");
    if (projRow) projRow = projRow.closest(".pg-row");
    if (projRow) {
      // "global UMAP/PCA space" is self-labelling; drop the "3D view" tag
      var projLab = projRow.querySelector(".pg-lab");
      if (projLab) projLab.style.display = "none";
      if (projRow.parentNode !== bar) bar.appendChild(projRow);
    }
    var ovLab = document.querySelector(".pg-ov");
    if (ovLab && ovLab.parentNode !== bar) bar.appendChild(ovLab);
    wireFullscreenLinks();
  }

  /* Unify every ".fs-link" on the page (steering AND mental-state) to use
     the Fullscreen API on its natural container instead of opening the
     demo in a new tab — consistent UX across the two figures. */
  function wireFullscreenLinks() {
    [].forEach.call(document.querySelectorAll(".fs-link"), function (a) {
      if (a.__fsWired) return;
      var container = a.closest(".demo") || a.closest(".pg");
      if (!container) return;
      a.__fsWired = true;
      a.removeAttribute("target");
      a.setAttribute("href", "#");
      a.addEventListener("click", function (e) {
        e.preventDefault();
        if (document.fullscreenElement === container) {
          document.exitFullscreen().catch(function () { });
        } else {
          container.requestFullscreen().catch(function () { });
        }
      });
    });
    if (!document.__fsListener) {
      document.__fsListener = true;
      document.addEventListener("fullscreenchange", function () {
        [].forEach.call(document.querySelectorAll(".fs-link"), function (a) {
          var c = a.closest(".demo") || a.closest(".pg");
          a.innerHTML = (document.fullscreenElement === c)
            ? "Exit fullscreen&nbsp;↗" : "Open fullscreen&nbsp;↗";
        });
      });
    }
  }
  function toggleKnobs() {
    var base = Number(sel.strength) === 0;
    [].forEach.call(document.querySelectorAll("#steer-pg .pg-knob"),
      function (r) { r.style.opacity = base ? 0.4 : 1;
        r.style.pointerEvents = base ? "none" : "auto"; });
  }
  function descFor() {
    if (Number(sel.strength) === 0)
      return "Baseline. Strength 0: the prompt solved unsteered.";
    var t = MAN.targets.filter(function (x) { return x.key === sel.target; })[0];
    var verb = (t.kind === "avoid_C" || t.kind === "avoid_G")
      ? "Repelling away from" : "Attracting toward";
    return verb + " " + t.label.replace(/^[→✕]\s*/, "") + ". " + t.desc +
      " Strength " + sel.strength + ", from generation token " + sel.start +
      " for " + sel.dur + " tokens.";
  }

  function need(key, cb) {
    var ck = sel.pid + "/" + key;
    if (cache[ck]) { cb(cache[ck]); return; }
    inject(BASE + key + ".js", function () {
      if (key === curKey) document.getElementById("pg-text").innerHTML =
        '<div class="pg-msg">This combination was not generated. ' +
        'Try another knob setting.</div>';
    });
  }
  function select() {
    var k = cellKey(sel.strength); curKey = k;
    document.getElementById("pg-desc").textContent = descFor();
    if (sel.overlay && Number(sel.strength) !== 0) need("baseline", function () {});
    var ck = sel.pid + "/" + k;
    if (cache[ck]) { render(); return; }
    document.getElementById("pg-text").innerHTML =
      '<div class="pg-msg">Loading run…</div>';
    need(k, function () { render(); });
  }

  /* Shared tooltip body for one token (cluster + group name/desc + cos +
     margin). `head` = the first line (e.g. "tok 17 · steered" for gen
     tokens, or just "prompt token" for context tokens with no index). */
  function _tipBody(head, c, g, cos, mg) {
    var cn = MAN.cluster_names || {}, gn = MAN.group_names || {};
    var cinfo = c != null ? cn[c] : null, ginfo = g != null ? gn[g] : null;
    var lines = [head];
    if (cinfo) lines.push("<b>C" + c + "</b> · " + cinfo.name +
      "<br><span>" + cinfo.desc + "</span>");
    if (ginfo) lines.push("<b>G" + g + "</b> · " + ginfo.name +
      "<br><span>" + ginfo.desc + "</span>");
    if (cos != null && mg != null) {
      lines.push("<span class='pg-tip-num'>cos " + (+cos).toFixed(3) +
        " · margin " + (+mg).toFixed(3) + "</span>");
    }
    return lines.join("<br>");
  }
  function tokTip(d, i) {
    var head = "<b>tok " + i + "</b>" +
      (d.steered.indexOf(i) >= 0
        ? " · <span style='color:#ffb4b4;'>steered</span>" : "");
    var c = d.cl ? d.cl[i] : null, g = d.g[i];
    var cos = d.cos ? d.cos[i] : null, mg = d.mg ? d.mg[i] : null;
    return _tipBody(head, c, g, cos, mg);
  }
  function promptTip(p) {
    // prompt (context) token — no index per user's request
    var pc = MAN.prompt_cl || [], pg = MAN.prompt_g || [];
    var co = MAN.prompt_cos || [], mg = MAN.prompt_mg || [];
    return _tipBody("<b>prompt token</b>",
                    pc[p] != null ? pc[p] : null,
                    pg[p] != null ? pg[p] : null,
                    co[p] != null ? co[p] : null,
                    mg[p] != null ? mg[p] : null);
  }

  function render() {
    var d = cache[sel.pid + "/" + curKey];
    if (!d) return;
    var pal = MAN.palette, n = d.tokens.length;
    var col = new Array(n), stArr = new Array(n);
    for (var i = 0; i < n; i++) { col[i] = pal[d.g[i]] || "#bdbdbd"; stArr[i] = 0; }
    d.steered.forEach(function (idx) { if (idx < n) stArr[idx] = 1; });

    var html = "";
    if (MAN.prompt_tokens && MAN.prompt_tokens.length) {
      var pg = MAN.prompt_g || null;            // per-prompt-token macrostate
      for (var p = 0; p < MAN.prompt_tokens.length; p++) {
        if (pg && p < pg.length) {              // real cluster colour
          var pc = pal[pg[p]] || "#bdbdbd";
          html += '<span class="pg-tok pg-tok-ctx" data-st="0" style="background:'
            + pc + ";color:" + txtColor(pc) + '">' +
            esc(MAN.prompt_tokens[p]) + "</span>";
        } else {                                // fallback: muted run
          html += '<span class="pg-ctx">' + esc(MAN.prompt_tokens[p]) +
            "</span>";
        }
      }
      html += "<span class='pg-div'>↳ GENERATION"
        + (Number(sel.strength) === 0 ? "" : " (STEERED TOKENS UNDERLINED)")
        + "</span>";
    }
    for (i = 0; i < n; i++) {
      var c = col[i];
      // data-step on generation tokens => click-to-scrub + hover tooltip
      html += '<span class="pg-tok pg-tok-gen" data-step="' + i +
        '" data-st="' + (stArr[i] ? 1 : 0) + '" style="background:' +
        c + ";color:" + txtColor(c) + '">' + esc(d.tokens[i]) + "</span>";
    }
    document.getElementById("pg-text").innerHTML = html ||
      '<div class="pg-msg">(empty generation)</div>';
    // bind click + hover on each generation token
    var spans = document.querySelectorAll("#pg-text .pg-tok-gen");
    [].forEach.call(spans, function (el) {
      var i = +el.getAttribute("data-step");
      bindTip(el, tokTip(d, i));
      el.addEventListener("click", function () { applyStep(i + 1, true); });
    });
    // bind hover only (no click — prompt tokens are pre-generation) on each
    // context token, indexed by their position in MAN.prompt_tokens
    var ctx = document.querySelectorAll("#pg-text .pg-tok-ctx");
    [].forEach.call(ctx, function (el, k) { bindTip(el, promptTip(k)); });
    // user: drop the bottom "Model's final answer" section entirely
    var _ae = document.getElementById("pg-answer");
    if (_ae) { _ae.innerHTML = ""; _ae.style.display = "none"; }
    // 3D path (downsampled): colour each path point by its group
    var P3 = d["points_" + sel.proj] || d.points_umap;
    var pcol = d.pgi.map(function (g) { return pal[g] || "#bdbdbd"; });
    var pst = new Array(P3.length);
    for (i = 0; i < pst.length; i++) pst[i] = 0;
    d.pst.forEach(function (k) { if (k < pst.length) pst[k] = 1; });
    var ns = d.steered.length;
    // The proj toggle above already says "global UMAP / PCA space" and the
    // scrub bar reads "N / N", so the hint just carries the legend chips.
    document.getElementById("pg3d-hint").innerHTML =
      "<i>● phase colour</i>" +
      (ns ? '<i style="color:#b0122a;">◯ ' + ns + " steered</i>" : "") +
      (sel.overlay && Number(sel.strength) !== 0
        ? '<i style="color:#9a988f;">· baseline overlay</i>' : "");
    if (!viewer) viewer = make3d(document.getElementById("pg3d"));
    var ov = null, bc = cache[sel.pid + "/baseline"];
    if (sel.overlay && Number(sel.strength) !== 0 && bc)
      ov = bc["points_" + sel.proj] || bc.points_umap;
    viewer.set(P3, pcol, pst, ov, d.pidx || null);
    syncScrub(n);        // scrubber spans the FULL token sequence, not P3.length
    applyStep(n, false); // start at the end (full path revealed)
  }

  /* time scrubber: in token-step units. The 3D viewer maps step -> ovn
     via the cell's downsample-index map (d.pidx). */
  var scrub = document.getElementById("pg3d-scrub");
  var scrubVal = document.getElementById("pg3d-scrubval");
  var curTokEl = null, anim = null;
  function syncScrub(maxN) {
    scrub.min = 0; scrub.max = maxN; scrub.value = maxN;
    scrubVal.textContent = maxN + " / " + maxN;
  }
  function applyStep(step, scrollInto) {
    if (!scrub.max) return;
    step = Math.max(0, Math.min(step | 0, +scrub.max));
    scrub.value = step;
    scrubVal.textContent = step + " / " + scrub.max;
    if (viewer) viewer.setStep(step);
    // bright-yellow outline on the current generation token
    if (curTokEl) curTokEl.classList.remove("is-current");
    var sel1 = "#pg-text .pg-tok-gen[data-step='" +
      Math.max(0, step - 1) + "']";
    curTokEl = document.querySelector(sel1);
    if (curTokEl) {
      curTokEl.classList.add("is-current");
      if (scrollInto) {
        // keep the active token visible inside the trace pane
        var pt = document.getElementById("pg-text");
        var r = curTokEl.getBoundingClientRect();
        var pr = pt.getBoundingClientRect();
        if (r.top < pr.top + 40 || r.bottom > pr.bottom - 40)
          pt.scrollTop += (r.top - pr.top) - pr.height / 2;
      }
    }
  }
  scrub.addEventListener("input", function () { applyStep(+scrub.value, false); });
  function togglePlay() {
    if (anim) { clearInterval(anim); anim = null; setPlayBtn(false); return; }
    var s = +scrub.value;
    if (s >= +scrub.max) s = 0;
    setPlayBtn(true);
    anim = setInterval(function () {
      s += 1;
      if (s > +scrub.max) { clearInterval(anim); anim = null; setPlayBtn(false); return; }
      applyStep(s, true);
    }, 70);
  }
  function setPlayBtn(playing) {
    var b = document.getElementById("pg3d-play");
    if (b) b.innerHTML = playing ? "&#10073;&#10073;&nbsp;Pause"
      : "&#9654;&nbsp;Live";
  }

  /* --- shared 3D viewer (drag-rotate, wheel-zoom) --- */
  function make3d(cv) {
    var ctx = cv.getContext("2d"), P = [], COL = [], ST = [], OV = null,
        PIDX = null, N = 0, VN = 0, CUR = -1,
        yaw = 0.6, pitch = -0.45, zoom = 1, drag = false,
        lx = 0, ly = 0, auto = true, dpr = 1, MU = [0, 0, 0], SC = 1;
    // largest k with PIDX[k] <= step; if no PIDX, identity.
    function pidxUpTo(step) {
      if (!PIDX) return Math.max(2, Math.min(step, N));
      var lo = 0, hi = PIDX.length;
      while (lo < hi) {
        var m = (lo + hi) >> 1;
        if (PIDX[m] < step) lo = m + 1; else hi = m;
      }
      return Math.max(2, Math.min(lo, N));
    }
    function fit() {
      dpr = window.devicePixelRatio || 1;
      var w = cv.clientWidth || 460, h = cv.clientHeight || 340;
      cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
    }
    // auto-fit the *displayed* trace to the canvas (keeps global rotation
    // axes — only recenters/rescales so a short trace fills the view)
    function refit() {
      var pts = (OV && OV.length > 1) ? P.concat(OV) : P;
      if (pts.length < 2) { MU = [0, 0, 0]; SC = 1; return; }
      MU = [0, 0, 0];
      pts.forEach(function (p) {
        MU[0] += p[0]; MU[1] += p[1]; MU[2] += p[2]; });
      MU[0] /= pts.length; MU[1] /= pts.length; MU[2] /= pts.length;
      // robust radius: 90th-percentile distance (ignore UMAP outliers)
      var d = pts.map(function (p) {
        var a = p[0] - MU[0], b = p[1] - MU[1], c = p[2] - MU[2];
        return Math.sqrt(a * a + b * b + c * c);
      }).sort(function (x, y) { return x - y; });
      SC = d[Math.floor(0.9 * (d.length - 1))] || d[d.length - 1] || 1;
    }
    function cl(v) { return v < -1.9 ? -1.9 : v > 1.9 ? 1.9 : v; }
    function rgba(hex, a) {
      var r = parseInt(hex.substr(1, 2), 16),
          g = parseInt(hex.substr(3, 2), 16),
          b = parseInt(hex.substr(5, 2), 16);
      return "rgba(" + r + "," + g + "," + b + "," + a + ")";
    }
    function proj(p) {
      var x0 = cl((p[0] - MU[0]) / SC), y0 = cl((p[1] - MU[1]) / SC),
          z0 = cl((p[2] - MU[2]) / SC);
      var cx = Math.cos(yaw), sx = Math.sin(yaw);
      var x = x0 * cx + z0 * sx, z = -x0 * sx + z0 * cx;
      var cy = Math.cos(pitch), sy = Math.sin(pitch);
      return [x, y0 * cy - z * sy, y0 * sy + z * cy];
    }
    function scr(pts, W, H) {
      var R = Math.min(W, H) * 0.42 * zoom, ox = W / 2, oy = H / 2, o = [];
      for (var i = 0; i < pts.length; i++) { var q = proj(pts[i]);
        o.push([ox + q[0] * R, oy - q[1] * R, q[2]]); }
      return o;
    }
    function draw() {
      if (cv.clientWidth &&
          Math.abs(cv.width - Math.round(cv.clientWidth * dpr)) > 2) fit();
      var W = cv.width, H = cv.height; ctx.clearRect(0, 0, W, H);
      if (N < 2) { requestAnimationFrame(draw); return; }
      // points revealed by the time scrubber (default = whole path)
      var nv = Math.max(2, Math.min(VN || N, N));
      if (OV && OV.length > 1) {           // pale baseline overlay (behind)
        var ovn = Math.max(2, Math.round(nv / N * OV.length));
        var pb = scr(OV, W, H);
        // path
        ctx.lineWidth = Math.max(1.2, dpr * 1.0);
        ctx.strokeStyle = "rgba(150,148,140,0.5)"; ctx.beginPath();
        for (var j = 0; j < ovn; j++)
          j ? ctx.lineTo(pb[j][0], pb[j][1]) : ctx.moveTo(pb[j][0], pb[j][1]);
        ctx.stroke();
        // small grey dots per baseline step (user request) — kept low
        // contrast so they read as "behind" the coloured trajectory dots
        ctx.fillStyle = "rgba(140,138,130,0.55)";
        for (var jj = 0; jj < ovn; jj++) {
          ctx.beginPath();
          ctx.arc(pb[jj][0], pb[jj][1], 1.7 * dpr, 0, 6.2832);
          ctx.fill();
        }
      }
      var pr = scr(P, W, H);
      // path coloured by computational phase — same palette as the dots
      ctx.lineWidth = Math.max(1.6, dpr * 1.7); ctx.lineJoin = "round";
      for (var i = 1; i < nv; i++) {
        ctx.strokeStyle = rgba(COL[i] || "#bdbdbd", 0.6);
        ctx.beginPath(); ctx.moveTo(pr[i - 1][0], pr[i - 1][1]);
        ctx.lineTo(pr[i][0], pr[i][1]); ctx.stroke();
      }
      var ord = [];
      for (i = 0; i < nv; i++) ord.push(i);
      ord.sort(function (a, b) { return pr[a][2] - pr[b][2]; });
      for (var k = 0; k < ord.length; k++) {
        var idx = ord[k], a = pr[idx];
        var dep = (a[2] + 1.4) / 2.8;
        var rad = (1.9 + 2.1 * dep) * dpr * Math.sqrt(zoom);
        ctx.beginPath(); ctx.arc(a[0], a[1], rad, 0, 6.2832);
        ctx.globalAlpha = 0.34 + 0.5 * dep; ctx.fillStyle = COL[idx]; ctx.fill();
        if (ST[idx]) { ctx.globalAlpha = 0.95; ctx.lineWidth = 1.6 * dpr;
          ctx.strokeStyle = "#b0122a"; ctx.beginPath();   // red — user spec
          ctx.arc(a[0], a[1], rad + 2 * dpr, 0, 6.2832); ctx.stroke(); }
        ctx.globalAlpha = 1;
      }
      // current/end marker (yellow square). No start ring — it could read
      // as a steered token at first glance.
      var e = pr[nv - 1];
      var cu = (CUR >= 0 && CUR < nv) ? pr[CUR] : e;
      ctx.lineWidth = 3 * dpr; ctx.strokeStyle = "#ffe000";
      ctx.strokeRect(cu[0] - 6 * dpr, cu[1] - 6 * dpr, 12 * dpr, 12 * dpr);
      if (auto) yaw += 0.0022;
      requestAnimationFrame(draw);
    }
    function pt(ev) { var r = cv.getBoundingClientRect();
      var t = ev.touches ? ev.touches[0] : ev;
      return [t.clientX - r.left, t.clientY - r.top]; }
    cv.addEventListener("mousedown", function (e) { drag = true; auto = false;
      var p = pt(e); lx = p[0]; ly = p[1]; });
    window.addEventListener("mouseup", function () { drag = false; });
    window.addEventListener("mousemove", function (e) { if (!drag) return;
      var p = pt(e); yaw += (p[0] - lx) * 0.01; pitch += (p[1] - ly) * 0.01;
      pitch = Math.max(-1.5, Math.min(1.5, pitch)); lx = p[0]; ly = p[1]; });
    cv.addEventListener("wheel", function (e) { e.preventDefault();
      zoom *= e.deltaY < 0 ? 1.1 : 0.9;
      zoom = Math.max(0.3, Math.min(5, zoom)); }, { passive: false });
    cv.addEventListener("dblclick", function () {
      yaw = 0.6; pitch = -0.45; zoom = 1; auto = true; });
    if ("ResizeObserver" in window) new ResizeObserver(fit).observe(cv);
    window.addEventListener("resize", fit);
    fit(); requestAnimationFrame(draw);
    return {
      set: function (pts, col, st, ov, pidx) {
        P = pts || []; COL = col || []; ST = st || []; OV = ov || null;
        PIDX = pidx || null; N = P.length; VN = N; CUR = N - 1;
        auto = true; refit(); fit();
      },
      // step is in TOKEN units; map to downsampled-path index via PIDX
      setStep: function (step) {
        VN = pidxUpTo(step); CUR = VN - 1;
      }
    };
  }

  inject(ROOT + "prompts.js", function () {
    document.getElementById("pg-desc").textContent =
      "Steering grid not built yet (run webpage/tools/build_grid_assets.py).";
  });
})();
