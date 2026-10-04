(function () {
  "use strict";

  function init() {
    const overview = document.getElementById("forcing-article-toc");
    if (!overview || document.getElementById("forcing-floating-toc")) return;
    const content = overview.closest(".page__content");
    if (!content) return;
    const headings = Array.from(content.querySelectorAll("h1[id], h2[id], h3[id]"));
    if (!headings.length) return;

    const root = document.createElement("aside");
    root.id = "forcing-floating-toc";
    root.className = "forcing-toc";
    root.setAttribute("aria-label", "Article navigation");
    root.innerHTML = '<button class="forcing-toc__toggle" type="button" aria-controls="forcing-toc-panel" aria-expanded="false" hidden><span aria-hidden="true">☰</span> Contents</button>' +
      '<div class="forcing-toc__panel" id="forcing-toc-panel" hidden>' +
      '<div class="forcing-toc__header"><span class="forcing-toc__title">On this page</span><button class="forcing-toc__close" type="button" aria-label="Hide table of contents">Hide</button></div>' +
      '<nav class="forcing-toc__scroll" aria-label="Article table of contents" tabindex="-1"><ul class="forcing-toc__list"></ul></nav>' +
      '<div class="forcing-toc__footer"><a href="#forcing-article-toc">Back to overview ↑</a></div></div>';

    const panel = root.querySelector(".forcing-toc__panel");
    const toggle = root.querySelector(".forcing-toc__toggle");
    const close = root.querySelector(".forcing-toc__close");
    const scroller = root.querySelector(".forcing-toc__scroll");
    const list = root.querySelector(".forcing-toc__list");
    const links = headings.map(function (heading) {
      const item = document.createElement("li");
      const link = document.createElement("a");
      const label = heading.textContent.trim();
      link.className = "forcing-toc__link forcing-toc__link--level-" + heading.tagName.slice(1);
      link.setAttribute("href", "#" + encodeURIComponent(heading.id));
      link.textContent = label;
      link.title = label;
      item.appendChild(link);
      list.appendChild(item);
      heading.setAttribute("data-forcing-section", "");
      return link;
    });

    // Keep fixed navigation outside the theme's animated/transformed main wrapper.
    document.body.appendChild(root);
    const masthead = document.querySelector(".masthead");
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    let mode = "";
    let desktopOpen = true;
    let compactOpen = false;
    let current = -1;
    let positions = [];
    let frame = 0;
    let browsing = false;

    function sectionOffset() {
      if (!masthead) return 32;
      const position = window.getComputedStyle(masthead).position;
      return position === "fixed" || position === "sticky"
        ? Math.max(32, masthead.getBoundingClientRect().bottom + 16) : 32;
    }

    function measure() {
      positions = headings.map(function (heading) {
        return heading.getBoundingClientRect().top + window.scrollY;
      });
      content.style.setProperty("--forcing-section-offset", sectionOffset() + "px");
    }

    function revealCurrent() {
      if (current < 0 || panel.hidden || browsing || scroller.contains(document.activeElement)) return;
      const linkRect = links[current].getBoundingClientRect();
      const scrollRect = scroller.getBoundingClientRect();
      if (linkRect.top < scrollRect.top + 12 || linkRect.bottom > scrollRect.bottom - 12) {
        scroller.scrollTop += linkRect.top - scrollRect.top - scroller.clientHeight / 3;
      }
    }

    function renderOpen() {
      const open = mode === "docked" ? desktopOpen : compactOpen;
      panel.hidden = !open;
      toggle.hidden = open;
      toggle.setAttribute("aria-expanded", String(open));
      revealCurrent();
    }

    function place() {
      const articleRect = content.getBoundingClientRect();
      const viewportWidth = document.documentElement.clientWidth;
      const available = viewportWidth - articleRect.right - 28 - 20;
      // Dock only when a usable column fits entirely outside the original text width.
      const nextMode = viewportWidth >= 1100 && available >= 200 ? "docked" : "compact";
      if (mode !== nextMode) {
        mode = nextMode;
        compactOpen = false;
        root.dataset.mode = mode;
        renderOpen();
      }
      if (mode === "docked") {
        const top = Math.max(24, masthead ? masthead.getBoundingClientRect().bottom + 18 : 24);
        root.style.left = articleRect.right + 28 + "px";
        root.style.top = top + "px";
        root.style.width = Math.min(300, available) + "px";
        root.style.setProperty("--forcing-toc-top", top + "px");
      } else {
        root.style.removeProperty("left");
        root.style.removeProperty("top");
        root.style.removeProperty("width");
      }
    }

    function update() {
      frame = 0;
      place();
      const readingLine = window.scrollY + sectionOffset() + 12;
      let index = -1;
      // The final heading above the reading line owns the current passage.
      for (let i = 0; i < positions.length && positions[i] <= readingLine; i++) index = i;
      if (index === current) return;
      if (current >= 0) links[current].removeAttribute("aria-current");
      current = index;
      if (current >= 0) links[current].setAttribute("aria-current", "location");
      revealCurrent();
    }

    function schedule() {
      if (!frame) frame = window.requestAnimationFrame(update);
    }

    function hidePanel() {
      if (mode === "docked") desktopOpen = false;
      else compactOpen = false;
      renderOpen();
      toggle.focus({ preventScroll: true });
    }

    toggle.addEventListener("click", function () {
      if (mode === "docked") desktopOpen = true;
      else compactOpen = true;
      renderOpen();
      close.focus({ preventScroll: true });
    });
    close.addEventListener("click", hidePanel);
    root.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && !panel.hidden) {
        event.preventDefault();
        hidePanel();
      }
    });
    scroller.addEventListener("pointerenter", function () { browsing = true; });
    scroller.addEventListener("pointerleave", function () { browsing = false; revealCurrent(); });

    // Capture before the theme's general smooth-scroll handler, including dotted IDs.
    root.addEventListener("click", function (event) {
      const link = event.target.closest("a");
      if (!link || !root.contains(link) || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      const target = document.getElementById(decodeURIComponent(link.hash.slice(1)));
      if (!target) return;
      event.preventDefault();
      event.stopPropagation();
      if (window.location.hash !== link.hash) window.history.pushState(null, "", link.hash);
      if (mode === "compact") {
        compactOpen = false;
        renderOpen();
      }
      target.setAttribute("tabindex", "-1");
      target.focus({ preventScroll: true });
      window.scrollTo({
        top: target.getBoundingClientRect().top + window.scrollY - sectionOffset(),
        behavior: reducedMotion.matches ? "auto" : "smooth"
      });
    }, true);

    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", function () {
      measure();
      place();
      revealCurrent();
      schedule();
    });
    window.addEventListener("load", function () { measure(); schedule(); });
    if (window.ResizeObserver) {
      new ResizeObserver(function () { measure(); schedule(); }).observe(content);
    }
    measure();
    update();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
}());
