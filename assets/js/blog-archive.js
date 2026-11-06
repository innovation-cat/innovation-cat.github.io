(function () {
  "use strict";

  var root = document.getElementById("blog-archive");
  if (!root) return;

  var controls = root.querySelector("[data-blog-controls]");
  var topicButtons = Array.from(root.querySelectorAll("[data-blog-topic]"));
  var search = root.querySelector("[data-blog-search]");
  var year = root.querySelector("[data-blog-year]");
  var count = root.querySelector("[data-blog-count]");
  var period = root.querySelector("[data-blog-period]");
  var pagination = root.querySelector("[data-blog-pagination]");
  var pageSummary = root.querySelector("[data-blog-page-summary]");
  var pageButtons = root.querySelector("[data-blog-page-buttons]");
  var empty = root.querySelector("[data-blog-empty]");
  var pageSize = Math.max(1, parseInt(root.dataset.pageSize, 10) || 4);
  var topics = topicButtons.map(function (button) { return button.dataset.blogTopic; });
  var years = Array.from(year.options).map(function (option) { return option.value; });
  var state = { topic: "all", year: "all", query: "", page: 1 };

  function normalize(text) {
    return String(text || "").toLowerCase().replace(/\s+/g, " ").trim();
  }

  // Search and pagination use the posts rendered by Jekyll's site.posts.
  var records = Array.from(root.querySelectorAll("[data-blog-card]")).map(function (card) {
    var recordTopics = (card.dataset.topics || card.dataset.topic || "").split(/\s+/).filter(Boolean);
    return {
      element: card,
      topics: recordTopics,
      year: card.dataset.year,
      searchText: normalize(card.querySelector(".blog-card-title").textContent + " " +
        (card.querySelector("[data-blog-description]") || { textContent: "" }).textContent + " " +
        card.querySelector(".blog-card-topic").textContent + " " + recordTopics.join(" "))
    };
  });

  function readLocation() {
    var params = new URL(window.location.href).searchParams;
    var selectedTopic = params.get("topic") || "all";
    var selectedYear = params.get("year") || "all";
    state.topic = topics.indexOf(selectedTopic) !== -1 ? selectedTopic : "all";
    state.year = years.indexOf(selectedYear) !== -1 ? selectedYear : "all";
    state.query = (params.get("q") || "").slice(0, 150);
    state.page = Math.max(1, parseInt(params.get("page"), 10) || 1);
  }

  function updateLocation() {
    var url = new URL(window.location.href);
    var values = {
      topic: state.topic === "all" ? "" : state.topic,
      year: state.year === "all" ? "" : state.year,
      q: state.query,
      page: state.page === 1 ? "" : String(state.page)
    };
    Object.keys(values).forEach(function (key) {
      if (values[key]) url.searchParams.set(key, values[key]);
      else url.searchParams.delete(key);
    });
    window.history.replaceState(null, "", url.pathname + url.search + url.hash);
  }

  function makePageButton(number, label, direction, totalPages) {
    var button = document.createElement("button");
    button.type = "button";
    button.className = "blog-page" + (direction ? " blog-page-direction" : "");
    button.dataset.blogPage = String(number);
    button.textContent = label;
    button.disabled = number < 1 || number > totalPages;
    if (direction) button.setAttribute("aria-label", direction + " page");
    else if (number === state.page) button.setAttribute("aria-current", "page");
    return button;
  }

  function render() {
    var terms = normalize(state.query).split(" ").filter(Boolean);
    var matches = records.filter(function (record) {
      return (state.topic === "all" || record.topics.indexOf(state.topic) !== -1) &&
        (state.year === "all" || record.year === state.year) &&
        terms.every(function (term) { return record.searchText.indexOf(term) !== -1; });
    });
    var totalPages = Math.max(1, Math.ceil(matches.length / pageSize));
    state.page = Math.min(Math.max(1, state.page), totalPages);
    var start = (state.page - 1) * pageSize;
    var visible = matches.slice(start, start + pageSize);
    var visibleElements = new Set(visible.map(function (record) { return record.element; }));

    records.forEach(function (record) { record.element.hidden = !visibleElements.has(record.element); });
    empty.hidden = matches.length !== 0;
    count.textContent = matches.length + (matches.length === 1 ? " essay" : " essays") + " · Newest first";
    period.textContent = visible.length ? Array.from(new Set(visible.map(function (record) { return record.year; }))).join(" / ") : "Archive";

    topicButtons.forEach(function (button) {
      button.setAttribute("aria-pressed", String(button.dataset.blogTopic === state.topic));
    });
    if (document.activeElement !== search) search.value = state.query;
    year.value = state.year;

    pageButtons.textContent = "";
    pagination.hidden = matches.length === 0;
    if (matches.length) {
      pageSummary.textContent = "Showing " + (start + 1) + "–" + Math.min(start + pageSize, matches.length) + " of " + matches.length + " essays";
      var fragment = document.createDocumentFragment();
      fragment.appendChild(makePageButton(state.page - 1, "←", "Previous", totalPages));
      for (var page = 1; page <= totalPages; page += 1) {
        fragment.appendChild(makePageButton(page, String(page), "", totalPages));
      }
      fragment.appendChild(makePageButton(state.page + 1, "→", "Next", totalPages));
      pageButtons.appendChild(fragment);
    } else {
      pageSummary.textContent = "0 essays";
    }
  }

  topicButtons.forEach(function (button) {
    button.addEventListener("click", function () {
      state.topic = button.dataset.blogTopic;
      state.page = 1;
      render();
      updateLocation();
    });
  });

  search.addEventListener("input", function () {
    state.query = search.value;
    state.page = 1;
    render();
    updateLocation();
  });

  year.addEventListener("change", function () {
    state.year = year.value;
    state.page = 1;
    render();
    updateLocation();
  });

  pageButtons.addEventListener("click", function (event) {
    var button = event.target.closest("[data-blog-page]");
    if (!button || button.disabled) return;
    state.page = parseInt(button.dataset.blogPage, 10);
    render();
    updateLocation();
    period.focus({ preventScroll: true });
    period.scrollIntoView({ block: "start", behavior: "auto" });
  });

  window.addEventListener("popstate", function () { readLocation(); render(); });

  var activeLink = document.querySelector('.masthead a[href$="/year-archive/"]');
  if (activeLink) activeLink.setAttribute("aria-current", "page");
  var menuToggle = document.querySelector("#site-nav > button");
  if (menuToggle) {
    var navigation = document.getElementById("site-nav");
    var visibleLinks = navigation.querySelector(".visible-links");
    var hiddenLinks = navigation.querySelector(".hidden-links");
    var navigationItems = Array.from(visibleLinks.children).concat(Array.from(hiddenLinks.children));
    var brandLink = navigationItems[0] && navigationItems[0].querySelector("a");

    // Adapt the shared priority menu on this page only. The legacy recursive
    // algorithm can hide the brand or recurse while a viewport is resizing.
    function layoutBlogNavigation() {
      var expanded = menuToggle.classList.contains("close");
      navigationItems.forEach(function (item) { visibleLinks.appendChild(item); });
      menuToggle.classList.add("hidden");
      hiddenLinks.classList.add("hidden");
      if (brandLink) brandLink.style.maxWidth = "";

      var available = navigation.getBoundingClientRect().width;
      if (available > 0 && visibleLinks.getBoundingClientRect().width > available) {
        menuToggle.classList.remove("hidden");
        available = Math.max(1, available - menuToggle.getBoundingClientRect().width - 16);
        if (brandLink) brandLink.style.maxWidth = available + "px";
        while (visibleLinks.children.length > 1 && visibleLinks.getBoundingClientRect().width > available) {
          hiddenLinks.insertBefore(visibleLinks.lastElementChild, hiddenLinks.firstChild);
        }
      }

      var hasHiddenLinks = hiddenLinks.children.length > 0;
      menuToggle.classList.toggle("hidden", !hasHiddenLinks);
      menuToggle.classList.toggle("close", expanded && hasHiddenLinks);
      hiddenLinks.classList.toggle("hidden", !expanded || !hasHiddenLinks);
      menuToggle.setAttribute("count", String(hiddenLinks.children.length));
      menuToggle.setAttribute("aria-expanded", String(expanded && hasHiddenLinks));
    }

    if (typeof window.updateNav === "function") window.updateNav = layoutBlogNavigation;
    window.addEventListener("resize", layoutBlogNavigation);
    layoutBlogNavigation();
    menuToggle.setAttribute("aria-label", "Toggle navigation");
    menuToggle.setAttribute("aria-expanded", String(menuToggle.classList.contains("close")));
    menuToggle.addEventListener("click", function () {
      menuToggle.setAttribute("aria-expanded", String(menuToggle.classList.contains("close")));
    });
  }

  readLocation();
  render();
  controls.hidden = false;
  root.dataset.enhanced = "true";
})();
