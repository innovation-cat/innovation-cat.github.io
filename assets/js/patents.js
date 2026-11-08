(function () {
  "use strict";

  var root = document.getElementById("patents-page");
  if (!root) return;

  var officeButtons = Array.from(root.querySelectorAll("[data-pt-office]"));
  var search = root.querySelector("[data-pt-search]");
  var status = root.querySelector("[data-pt-status]");
  var count = root.querySelector("[data-pt-count]");
  var summary = root.querySelector("[data-pt-summary]");
  var pagination = root.querySelector("[data-pt-pagination]");
  var more = root.querySelector("[data-pt-more]");
  var empty = root.querySelector("[data-pt-empty]");
  var groups = Array.from(root.querySelectorAll("[data-pt-group]"));
  var pageSize = Math.max(1, parseInt(root.dataset.pageSize, 10) || 8);
  var offices = officeButtons.map(function (button) { return button.dataset.ptOffice; });
  var statuses = Array.from(status.options).map(function (option) { return option.value; });
  var state = { office: "all", status: "all", query: "", limit: pageSize };

  function normalize(text) {
    return String(text || "").normalize("NFKC").toLowerCase()
      .replace(/[\u2010-\u2015]/g, "-").replace(/\s+/g, " ").trim();
  }

  // The complete catalogue and its PDF links are rendered by Jekyll.
  // Enhancement changes visibility only, leaving the archive usable without JS.
  var records = Array.from(root.querySelectorAll("[data-pt-entry]")).map(function (entry) {
    return {
      element: entry,
      office: entry.dataset.office,
      status: entry.dataset.status,
      searchText: normalize(entry.querySelector(".pt-title").textContent + " " +
        entry.dataset.number + " " + entry.dataset.status.replace(/-/g, " ") + " " +
        entry.closest("[data-pt-group]").querySelector(".pt-office-name").textContent)
    };
  });

  function readLocation() {
    var params = new URL(window.location.href).searchParams;
    var selectedOffice = params.get("office") || "all";
    var selectedStatus = params.get("status") || "all";
    state.office = offices.indexOf(selectedOffice) !== -1 ? selectedOffice : "all";
    state.status = statuses.indexOf(selectedStatus) !== -1 ? selectedStatus : "all";
    state.query = (params.get("q") || "").slice(0, 150);
    state.limit = Math.min(records.length, Math.max(pageSize, parseInt(params.get("shown"), 10) || pageSize));
  }

  function updateLocation() {
    var url = new URL(window.location.href);
    var values = {
      office: state.office === "all" ? "" : state.office,
      status: state.status === "all" ? "" : state.status,
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
      return (state.office === "all" || record.office === state.office) &&
        (state.status === "all" || record.status === state.status) &&
        terms.every(function (term) { return record.searchText.indexOf(term) !== -1; });
    });
    var visible = new Set(matches.slice(0, state.limit).map(function (record) { return record.element; }));

    records.forEach(function (record) { record.element.hidden = !visible.has(record.element); });
    groups.forEach(function (group) {
      group.hidden = !group.querySelector("[data-pt-entry]:not([hidden])");
      var groupCount = matches.filter(function (record) { return record.office === group.dataset.ptGroup; }).length;
      group.querySelector("[data-pt-group-count]").textContent = groupCount + (groupCount === 1 ? " record" : " records");
    });
    officeButtons.forEach(function (button) { button.setAttribute("aria-pressed", String(button.dataset.ptOffice === state.office)); });
    if (search.value !== state.query) search.value = state.query;
    status.value = state.status;
    count.textContent = matches.length + (matches.length === 1 ? " record" : " records");
    summary.textContent = "Showing " + visible.size + " of " + matches.length + (matches.length === 1 ? " record" : " records");
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

  officeButtons.forEach(function (button) {
    button.addEventListener("click", function () { state.office = button.dataset.ptOffice; applyFilters(); });
  });
  search.addEventListener("input", function () { state.query = search.value; applyFilters(); });
  status.addEventListener("change", function () { state.status = status.value; applyFilters(); });
  more.addEventListener("click", function () {
    var previousLimit = state.limit;
    state.limit = Math.min(records.length, state.limit + pageSize);
    var matches = render();
    updateLocation();
    if (matches[previousLimit]) matches[previousLimit].element.querySelector(".pt-title a").focus();
  });
  root.querySelector("[data-pt-reset]").addEventListener("click", function () {
    state.office = "all";
    state.status = "all";
    state.query = "";
    applyFilters();
    search.focus();
  });
  window.addEventListener("popstate", function () { readLocation(); render(); });

  var navigation = document.getElementById("site-nav");
  var menuToggle = navigation && navigation.querySelector("button");
  if (menuToggle) {
    var visibleLinks = navigation.querySelector(".visible-links");
    var hiddenLinks = navigation.querySelector(".hidden-links");
    var navigationItems = Array.from(visibleLinks.children).concat(Array.from(hiddenLinks.children));
    var brandLink = navigationItems[0].querySelector("a");

    // Keep this page's legacy priority navigation stable across phone/desktop
    // resizes without changing the shared navigation or another page's scripts.
    function layoutPatentNavigation() {
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

    menuToggle.setAttribute("type", "button");
    menuToggle.setAttribute("aria-label", "Toggle navigation");
    hiddenLinks.id = "patents-site-menu";
    menuToggle.setAttribute("aria-controls", hiddenLinks.id);
    navigation.querySelectorAll("a").forEach(function (link) {
      if (new URL(link.href, window.location.href).pathname === window.location.pathname) link.setAttribute("aria-current", "page");
    });
    if (typeof window.updateNav === "function") window.updateNav = layoutPatentNavigation;
    window.addEventListener("resize", layoutPatentNavigation);
    layoutPatentNavigation();
    if (document.fonts) document.fonts.ready.then(layoutPatentNavigation);
    menuToggle.addEventListener("click", function () {
      menuToggle.setAttribute("aria-expanded", String(menuToggle.classList.contains("close")));
    });
    navigation.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && menuToggle.classList.contains("close")) {
        menuToggle.classList.remove("close");
        hiddenLinks.classList.add("hidden");
        menuToggle.setAttribute("aria-expanded", "false");
        menuToggle.focus();
      }
    });
  }

  readLocation();
  render();
  root.querySelectorAll("[data-pt-controls]").forEach(function (control) { control.hidden = false; });
  root.dataset.enhanced = "true";
})();
