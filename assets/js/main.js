/* ============================================================
   Single-page behaviour: scroll reveal, lazy iframes,
   mental-state trace switching, steering baseline/steered toggle
   ============================================================ */
(function () {
  "use strict";

  /* ---------- scroll reveal ---------- */
  var reveals = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window) {
    var ro = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add("in"); ro.unobserve(e.target); }
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.06 });
    reveals.forEach(function (s) { ro.observe(s); });
  } else {
    reveals.forEach(function (s) { s.classList.add("in"); });
  }

  /* ---------- demo helpers ---------- */
  // Build an iframe inside a holder, pointing at `src`.
  function mountFrame(holder, src) {
    var existing = holder.querySelector("iframe");
    if (existing) { if (existing.src.indexOf(src) === -1) existing.src = src; return existing; }
    var ph = holder.querySelector(".frame-ph");
    var f = document.createElement("iframe");
    f.setAttribute("loading", "lazy");
    f.setAttribute("title", "Interactive RET trajectory");
    f.src = src;
    holder.appendChild(f);
    if (ph) ph.remove();
    return f;
  }

  function currentSrc(demo) {
    var holder = demo.querySelector(".frame-holder");
    if (demo.dataset.demo === "steer") {
      var mode = demo.querySelector(".tg.active").dataset.mode;
      return mode === "steer" ? holder.dataset.steer : holder.dataset.base;
    }
    return holder.dataset.first;
  }

  function syncFsLink(demo) {
    var link = demo.querySelector(".fs-link");
    if (!link) return;
    var src = currentSrc(demo);
    link.href = src; link.dataset.fs = src;
  }

  // Lazy-load each demo's frame when it nears the viewport (or on click).
  var demos = document.querySelectorAll(".demo");
  demos.forEach(function (demo) {
    var holder = demo.querySelector(".frame-holder");
    var loaded = false;
    function load() {
      if (loaded) return;
      loaded = true;
      mountFrame(holder, currentSrc(demo));
      syncFsLink(demo);
    }
    syncFsLink(demo); // correct fullscreen target before any scroll/interaction

    var btn = holder.querySelector(".loadbtn");
    if (btn) btn.addEventListener("click", load);

    if ("IntersectionObserver" in window) {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting) { load(); io.disconnect(); }
        });
      }, { rootMargin: "500px 0px" });
      io.observe(holder);
    } else {
      load();
    }

    /* ----- mental-state: pick a trace via the top-right dropdown ----- */
    if (demo.dataset.demo === "mental") {
      var pick = demo.querySelector(".demo-pick-sel");
      if (pick) {
        pick.addEventListener("change", function () {
          var src = pick.value;
          holder.dataset.first = src;
          if (loaded) mountFrame(holder, src);
          syncFsLink(demo);
        });
      }
    }

    /* ----- steering: pick a case + baseline/steered toggle ----- */
    if (demo.dataset.demo === "steer") {
      var caps = {
        0: "<strong>Sum 1…11.</strong> Baseline enumerates <span class='m'>1+2=3, +3=6, +4=10…</span>; steered toward C35 it writes <span class='m'>“arithmetic series: sum = n(n+1)/2 … 11·12/2 = 66”</span>. Both reach 66 — a different computational path to the same answer.",
        1: "<strong>Train word problem.</strong> Baseline sets up one algebraic equation <span class='m'>60t + 80(t−1) = 240</span>. Steered toward C30 it reframes via staged relative motion — 60&nbsp;mi covered by 10:00, then 180&nbsp;mi closing at 140&nbsp;mph → they meet 11:17&nbsp;AM. The solution strategy itself shifts.",
        2: "<strong>Means problem.</strong> A subtler shift: both solve the AM/GM/HM system, but steered toward C57 (“wording disambiguation”) the model methodically restates what each given mean <em>means</em> before using it."
      };
      var pills = demo.querySelectorAll(".pill");
      var tgs = demo.querySelectorAll(".tg");
      var cap = demo.querySelector("#steer-cap");
      var steerLab = demo.querySelector(".tg-steerlabel");

      pills.forEach(function (p, idx) {
        p.addEventListener("click", function () {
          pills.forEach(function (x) { x.classList.remove("active"); });
          p.classList.add("active");
          holder.dataset.base = p.dataset.base;
          holder.dataset.steer = p.dataset.steer;
          if (steerLab) steerLab.textContent = p.dataset.steerlabel;
          if (cap && caps[idx]) cap.innerHTML = caps[idx];
          if (loaded) mountFrame(holder, currentSrc(demo));
          syncFsLink(demo);
        });
      });

      tgs.forEach(function (t) {
        t.addEventListener("click", function () {
          tgs.forEach(function (x) { x.classList.remove("active"); });
          t.classList.add("active");
          if (loaded) mountFrame(holder, currentSrc(demo));
          syncFsLink(demo);
        });
      });
    }
  });

  /* ---------- active nav link on scroll ---------- */
  var sections = [].slice.call(document.querySelectorAll("main section[id]"));
  var navlinks = {};
  document.querySelectorAll("#topnav .navlinks a").forEach(function (a) {
    navlinks[a.getAttribute("href").slice(1)] = a;
  });
  if ("IntersectionObserver" in window && sections.length) {
    var no = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        var a = navlinks[e.target.id];
        if (a && e.isIntersecting) {
          Object.keys(navlinks).forEach(function (k) {
            navlinks[k].style.color = "";
          });
          a.style.color = "var(--accent)";
        }
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    sections.forEach(function (s) { no.observe(s); });
  }
})();
