(function () {
  "use strict";

  var root = document.getElementById("publications-page");
  if (!root) return;

  var topicButtons = Array.from(root.querySelectorAll("[data-pub-topic]"));
  var search = root.querySelector("[data-pub-search]");
  var year = root.querySelector("[data-pub-year]");
  var count = root.querySelector("[data-pub-count]");
  var summary = root.querySelector("[data-pub-summary]");
  var pagination = root.querySelector("[data-pub-pagination]");
  var more = root.querySelector("[data-pub-more]");
  var empty = root.querySelector("[data-pub-empty]");
  var groups = Array.from(root.querySelectorAll("[data-pub-group]"));
  var pageSize = Math.max(1, parseInt(root.dataset.pageSize, 10) || 4);
  var topics = topicButtons.map(function (button) { return button.dataset.pubTopic; });
  var years = Array.from(year.options).map(function (option) { return option.value; });
  var state = { topic: "all", year: "all", query: "", limit: pageSize };

  function normalize(text) {
    return String(text || "").normalize("NFKC").toLowerCase()
      .replace(/[\u2010-\u2015]/g, "-").replace(/\s+/g, " ").trim();
  }

  // Jekyll renders every publication and link; JavaScript only filters visibility.
  var records = Array.from(root.querySelectorAll("[data-pub-entry]")).map(function (entry) {
    return {
      element: entry,
      topic: entry.dataset.topic,
      year: entry.dataset.year,
      searchText: normalize(entry.querySelector(".pub-title").textContent + " " +
        entry.querySelector(".pub-authors").textContent + " " +
        entry.querySelector(".pub-venue").textContent + " " + entry.dataset.topic)
    };
  });

  function readLocation() {
    var params = new URL(window.location.href).searchParams;
    var selectedTopic = params.get("topic") || "all";
    var selectedYear = params.get("year") || "all";
    state.topic = topics.indexOf(selectedTopic) !== -1 ? selectedTopic : "all";
    state.year = years.indexOf(selectedYear) !== -1 ? selectedYear : "all";
    state.query = (params.get("q") || "").slice(0, 150);
    state.limit = Math.min(records.length, Math.max(pageSize, parseInt(params.get("shown"), 10) || pageSize));
  }

  function updateLocation() {
    var url = new URL(window.location.href);
    var values = {
      topic: state.topic === "all" ? "" : state.topic,
      year: state.year === "all" ? "" : state.year,
      q: state.query,
      shown: state.limit === pageSize ? "" : String(state.limit)
    };
    Object.keys(values).forEach(function (key) {
      if (values[key]) url.searchParams.set(key, values[key]);
      else url.searchParams.delete(key);
    });
    window.history.replaceState(null, "", url.pathname + url.search + url.hash);
  }

  function render() {
    var terms = normalize(state.query).split(" ").filter(Boolean);
    var matches = records.filter(function (record) {
      return (state.topic === "all" || record.topic === state.topic) &&
        (state.year === "all" || record.year === state.year) &&
        terms.every(function (term) { return record.searchText.indexOf(term) !== -1; });
    });
    var visible = new Set(matches.slice(0, state.limit).map(function (record) { return record.element; }));

    records.forEach(function (record) { record.element.hidden = !visible.has(record.element); });
    groups.forEach(function (group) { group.hidden = !group.querySelector("[data-pub-entry]:not([hidden])"); });
    topicButtons.forEach(function (button) { button.setAttribute("aria-pressed", String(button.dataset.pubTopic === state.topic)); });
    if (document.activeElement !== search) search.value = state.query;
    year.value = state.year;
    count.textContent = matches.length + (matches.length === 1 ? " publication" : " publications");
    summary.textContent = "Showing " + visible.size + " of " + matches.length + (matches.length === 1 ? " entry" : " entries");
    more.hidden = visible.size >= matches.length;
    empty.hidden = matches.length !== 0;
    pagination.hidden = matches.length === 0;
    return matches;
  }

  function applyFilters() {
    state.limit = pageSize;
    render();
    updateLocation();
  }

  topicButtons.forEach(function (button) {
    button.addEventListener("click", function () { state.topic = button.dataset.pubTopic; applyFilters(); });
  });
  search.addEventListener("input", function () { state.query = search.value; applyFilters(); });
  year.addEventListener("change", function () { state.year = year.value; applyFilters(); });
  more.addEventListener("click", function () {
    var previousLimit = state.limit;
    state.limit = Math.min(records.length, state.limit + pageSize);
    var matches = render();
    updateLocation();
    if (matches[previousLimit]) matches[previousLimit].element.querySelector(".pub-title a").focus();
  });
  root.querySelector("[data-pub-reset]").addEventListener("click", function () {
    state.topic = "all";
    state.year = "all";
    state.query = "";
    applyFilters();
    search.focus();
  });
  window.addEventListener("popstate", function () { readLocation(); render(); });

  var menuToggle = document.querySelector("#site-nav > button");
  if (menuToggle) {
    var navigation = document.getElementById("site-nav");
    var visibleLinks = navigation.querySelector(".visible-links");
    var hiddenLinks = navigation.querySelector(".hidden-links");
    var navigationItems = Array.from(visibleLinks.children).concat(Array.from(hiddenLinks.children));
    var brandLink = navigationItems[0].querySelector("a");

    // The legacy priority menu restores only one item per resize. Recompute
    // its bounded list on this page so a phone-to-desktop resize restores all links.
    function layoutPublicationNavigation() {
      var expanded = menuToggle.classList.contains("close");
      navigationItems.forEach(function (item) { visibleLinks.appendChild(item); });
      menuToggle.classList.add("hidden");
      hiddenLinks.classList.add("hidden");
      brandLink.style.maxWidth = "";

      var available = navigation.getBoundingClientRect().width;
      if (available > 0 && visibleLinks.getBoundingClientRect().width > available) {
        menuToggle.classList.remove("hidden");
        available = Math.max(1, available - menuToggle.getBoundingClientRect().width - 16);
        brandLink.style.maxWidth = available + "px";
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

    if (typeof window.updateNav === "function") window.updateNav = layoutPublicationNavigation;
    window.addEventListener("resize", layoutPublicationNavigation);
    layoutPublicationNavigation();
    hiddenLinks.id = "publications-site-menu";
    menuToggle.setAttribute("aria-controls", hiddenLinks.id);
    menuToggle.addEventListener("click", function () {
      menuToggle.setAttribute("aria-expanded", String(menuToggle.classList.contains("close")));
    });
  }

  readLocation();
  render();
  root.querySelectorAll("[data-pub-controls]").forEach(function (control) { control.hidden = false; });
  root.dataset.enhanced = "true";
})();
