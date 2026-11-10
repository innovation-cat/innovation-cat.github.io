/* Apply the saved appearance before first paint, then connect the shared panel. */
(function () {
  'use strict';

  var storageKey = 'innovation-cat.appearance.v1';
  var root = document.documentElement;
  var labels = { forest: 'Forest', parchment: 'Parchment', ocean: 'Ocean Mist', lavender: 'Dusk Lavender', midnight: 'Midnight', white: 'Pure White' };
  var media = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
  var state = { mode: 'light', lightTheme: 'forest' };
  var panel, trigger, container;
  var canSave = true;

  function readState(value) {
    var parsed;
    try { parsed = JSON.parse(value); } catch (error) { return { mode: 'light', lightTheme: 'forest' }; }
    return {
      mode: parsed && typeof parsed.mode === 'string' && /^(light|dark|system)$/.test(parsed.mode) ? parsed.mode : 'light',
      lightTheme: parsed && typeof parsed.lightTheme === 'string' && Object.prototype.hasOwnProperty.call(labels, parsed.lightTheme) && parsed.lightTheme !== 'midnight' ? parsed.lightTheme : 'forest'
    };
  }

  try { state = readState(window.localStorage.getItem(storageKey)); } catch (error) { canSave = false; }

  function activeTheme() {
    return state.mode === 'dark' || (state.mode === 'system' && media && media.matches) ? 'midnight' : state.lightTheme;
  }

  function apply() {
    var theme = activeTheme();
    root.setAttribute('data-theme', theme);
    root.setAttribute('data-color-mode', state.mode);
    if (!panel) return;
    panel.querySelectorAll('[data-theme-mode]').forEach(function (button) {
      button.setAttribute('aria-pressed', String(button.getAttribute('data-theme-mode') === state.mode));
    });
    panel.querySelectorAll('[data-theme-choice]').forEach(function (button) {
      button.setAttribute('aria-pressed', String(button.getAttribute('data-theme-choice') === theme));
    });
    panel.querySelector('[data-theme-active]').textContent = labels[theme];
    panel.querySelector('[data-theme-save-status]').textContent = canSave ? 'Your preference is saved' : 'Applied for this visit';
    var meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute('content', window.getComputedStyle(root).getPropertyValue('--theme-page').trim());
  }

  function save() {
    try { window.localStorage.setItem(storageKey, JSON.stringify(state)); canSave = true; } catch (error) { canSave = false; }
    apply();
    panel.querySelector('[data-theme-announcement]').textContent = labels[activeTheme()] + ' theme applied' + (state.mode === 'system' ? '. Following your system appearance.' : '.');
  }

  function close(restoreFocus) {
    panel.hidden = true;
    trigger.setAttribute('aria-expanded', 'false');
    if (restoreFocus) trigger.focus();
  }

  function initialize() {
    container = document.querySelector('[data-appearance]');
    if (!container) return;
    panel = container.querySelector('#appearance-panel');
    trigger = container.querySelector('[data-theme-toggle]');
    trigger.hidden = false;
    apply();
    // Revealing the trigger changes the space available to the priority menu.
    window.dispatchEvent(new Event('resize'));

    trigger.addEventListener('click', function () {
      if (!panel.hidden) { close(false); return; }
      // The appearance panel and the priority navigation are separate controls.
      var navigation = document.querySelector('#site-nav .hidden-links');
      if (navigation) navigation.classList.add('hidden');
      var navigationToggle = document.querySelector('#site-nav > button');
      if (navigationToggle) navigationToggle.classList.remove('close');
      panel.hidden = false;
      panel.scrollTop = 0;
      trigger.setAttribute('aria-expanded', 'true');
      panel.querySelector('[data-theme-mode][aria-pressed="true"]').focus();
    });
    panel.querySelector('[data-theme-close]').addEventListener('click', function () { close(true); });
    panel.querySelectorAll('[data-theme-choice]').forEach(function (button) {
      button.addEventListener('click', function () {
        var theme = button.getAttribute('data-theme-choice');
        state.mode = theme === 'midnight' ? 'dark' : 'light';
        if (theme !== 'midnight') state.lightTheme = theme;
        save();
      });
    });
    panel.querySelectorAll('[data-theme-mode]').forEach(function (button) {
      button.addEventListener('click', function () { state.mode = button.getAttribute('data-theme-mode'); save(); });
    });
    panel.querySelector('[data-theme-reset]').addEventListener('click', function () {
      state = { mode: 'light', lightTheme: 'forest' };
      try { window.localStorage.removeItem(storageKey); canSave = true; } catch (error) { canSave = false; }
      apply();
      panel.querySelector('[data-theme-announcement]').textContent = 'Default appearance restored.';
    });
    document.addEventListener('click', function (event) { if (!panel.hidden && !container.contains(event.target)) close(false); });
    document.addEventListener('keydown', function (event) { if (event.key === 'Escape' && !panel.hidden) { event.preventDefault(); close(true); } });
    container.addEventListener('focusout', function () {
      window.setTimeout(function () { if (!panel.hidden && !container.contains(document.activeElement)) close(false); }, 0);
    });
    window.addEventListener('storage', function (event) {
      if (event.key === storageKey || event.key === null) { state = readState(event.newValue); apply(); }
    });
  }

  apply();
  if (media) {
    var systemChanged = function () { if (state.mode === 'system') apply(); };
    if (media.addEventListener) media.addEventListener('change', systemChanged);
    else if (media.addListener) media.addListener(systemChanged);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initialize);
  else initialize();
}());
