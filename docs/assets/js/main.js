/**
 * Stoolap — Main JavaScript
 */
(function () {
  'use strict';

  document.addEventListener('DOMContentLoaded', function () {
    document.body.classList.remove('preload');

    // Homepage: fallback class for browsers without :has()
    if (document.querySelector('.hero')) {
      var sc = document.querySelector('.site-content');
      if (sc) sc.classList.add('homepage-content');
    }

    initHeaderScroll();
    initThemeToggle();
    initMobileMenu();
    initSidebarNav();
    addCodeCopyButtons();
    addLanguageBadges();
    initSvgThemeSync();
    initCodeTabs();

    if (document.querySelector('.doc-article')) {
      setupTableOfContents();
    }
  });

  /* ── Header Scroll (transparent → solid on homepage) ── */
  function initHeaderScroll() {
    var header = document.querySelector('.site-header.header-transparent');
    if (!header) return;

    var threshold = 40;
    var scrolled = false;

    function check() {
      var past = window.scrollY > threshold;
      if (past !== scrolled) {
        scrolled = past;
        if (scrolled) {
          header.classList.add('header-scrolled');
        } else {
          header.classList.remove('header-scrolled');
        }
      }
    }

    window.addEventListener('scroll', check, { passive: true });
    check();
  }

  /* ── Theme Toggle ── */
  function initThemeToggle() {
    var toggle = document.getElementById('theme-toggle');
    if (!toggle) return;
    toggle.addEventListener('click', function () {
      var current = document.documentElement.getAttribute('data-theme') || 'light';
      var next = current === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('theme', next);
    });
  }

  /* ── Mobile Menu ── */
  function initMobileMenu() {
    var btn = document.querySelector('.mobile-menu-toggle');
    var nav = document.querySelector('.main-nav');
    if (!btn || !nav) return;

    function close() {
      nav.classList.remove('active');
      btn.setAttribute('aria-expanded', 'false');
      btn.classList.remove('active');
      document.body.style.overflow = '';
    }

    btn.addEventListener('click', function () {
      var open = nav.classList.toggle('active');
      btn.setAttribute('aria-expanded', open);
      btn.classList.toggle('active', open);
      document.body.style.overflow = open ? 'hidden' : '';
    });

    document.addEventListener('click', function (e) {
      if (nav.classList.contains('active') &&
          !nav.contains(e.target) &&
          !btn.contains(e.target)) {
        close();
      }
    });

    window.addEventListener('resize', function () {
      if (window.innerWidth > 768 && nav.classList.contains('active')) close();
    });
  }

  /* ── Sidebar Navigation ── */
  function initSidebarNav() {
    var toggles = document.querySelectorAll('.docs-nav-toggle');
    toggles.forEach(function (toggle) {
      toggle.addEventListener('click', function () {
        var expanded = this.getAttribute('aria-expanded') === 'true';
        this.setAttribute('aria-expanded', !expanded);
      });
    });

    if (window.innerWidth <= 768) {
      var current = document.querySelector('.docs-nav a[aria-current="page"]');
      if (current) {
        var section = current.closest('.docs-nav-section');
        toggles.forEach(function (t) {
          t.setAttribute('aria-expanded', t.parentElement === section);
        });
      }
    }
  }

  /* ── Copy Buttons ── */
  var COPY_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16"><path d="M6.9998 6V3C6.9998 2.44772 7.44752 2 7.9998 2H19.9998C20.5521 2 20.9998 2.44772 20.9998 3V17C20.9998 17.5523 20.5521 18 19.9998 18H16.9998V20.9991C16.9998 21.5519 16.5499 22 15.993 22H4.00666C3.45059 22 3 21.5554 3 20.9991L3.0026 7.00087C3.0027 6.44811 3.45264 6 4.00942 6H6.9998ZM5.00242 8L5.00019 20H14.9998V8H5.00242ZM8.9998 6H16.9998V16H18.9998V4H8.9998V6Z"/></svg>';
  var CHECK_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16"><path d="M9.9997 15.1709L19.1921 5.97852L20.6063 7.39273L9.9997 17.9993L3.63574 11.6354L5.04996 10.2212L9.9997 15.1709Z"/></svg>';

  function addCodeCopyButtons() {
    document.querySelectorAll('pre').forEach(function (block) {
      if (block.querySelector('.copy-button')) return;
      // Skip homepage terminal and code example pre elements
      if (block.classList.contains('term-body') || block.classList.contains('code-pre')) return;
      var code = block.querySelector('code');
      var text = code ? code.innerText : block.innerText;

      var btn = document.createElement('button');
      btn.className = 'copy-button';
      btn.type = 'button';
      btn.setAttribute('aria-label', 'Copy code to clipboard');
      btn.innerHTML = COPY_SVG;

      btn.addEventListener('click', function () {
        navigator.clipboard.writeText(text).then(function () {
          btn.innerHTML = CHECK_SVG;
          setTimeout(function () { btn.innerHTML = COPY_SVG; }, 2000);
        });
      });

      // Append to outer div.highlighter-rouge if it exists (for toolbar alignment with language badge)
      var wrapper = block.closest('div.highlighter-rouge');
      if (wrapper) {
        wrapper.appendChild(btn);
        // Position copy button to the left of the language badge
        var badgeWidth = window.getComputedStyle(wrapper, '::before').width;
        var w = parseFloat(badgeWidth) || 0;
        btn.style.right = w + 'px';
      } else {
        btn.style.right = '0px';
        block.appendChild(btn);
      }
    });
  }

  /* ── Language Badges ── */
  function addLanguageBadges() {
    document.querySelectorAll('div.highlighter-rouge').forEach(function (block) {
      var classes = block.className.split(' ');
      for (var i = 0; i < classes.length; i++) {
        if (classes[i].indexOf('language-') === 0) {
          block.setAttribute('data-lang', classes[i].replace('language-', '').toUpperCase());
          break;
        }
      }
    });
  }

  /* ── Table of Contents ── */
  function setupTableOfContents() {
    var content = document.querySelector('.doc-content');
    if (!content) return;

    var headings = content.querySelectorAll('h2, h3');
    if (headings.length < 3) return;

    var toc = document.createElement('nav');
    toc.className = 'table-of-contents';
    toc.setAttribute('aria-label', 'Table of Contents');

    var title = document.createElement('h2');
    title.textContent = 'On this page';
    title.className = 'toc-title';

    var list = document.createElement('ul');

    headings.forEach(function (h, i) {
      if (!h.id) h.id = 'heading-' + i;
      var li = document.createElement('li');
      li.className = 'toc-' + h.tagName.toLowerCase();
      var a = document.createElement('a');
      a.href = '#' + h.id;
      a.textContent = h.textContent;
      li.appendChild(a);
      list.appendChild(li);
    });

    toc.appendChild(title);
    toc.appendChild(list);

    var header = document.querySelector('.doc-header');
    if (header) {
      header.insertAdjacentElement('afterend', toc);
    } else {
      content.insertAdjacentElement('beforebegin', toc);
    }
  }

  /* ── SVG Theme Sync ── */
  function initSvgThemeSync() {
    syncAllSvgs();
    var observer = new MutationObserver(function (mutations) {
      for (var i = 0; i < mutations.length; i++) {
        if (mutations[i].attributeName === 'data-theme') { syncAllSvgs(); break; }
      }
    });
    observer.observe(document.documentElement, { attributes: true });
  }

  function syncAllSvgs() {
    var theme = document.documentElement.getAttribute('data-theme') || 'light';
    var isDark = theme === 'dark';
    document.querySelectorAll('object[type="image/svg+xml"]').forEach(function (obj) {
      applyThemeToSvgObject(obj, theme, isDark);
      obj.addEventListener('load', function handler() {
        applyThemeToSvgObject(obj, document.documentElement.getAttribute('data-theme') || 'light',
          document.documentElement.getAttribute('data-theme') === 'dark');
        obj.removeEventListener('load', handler);
      });
    });
  }

  function applyThemeToSvgObject(obj, theme, isDark) {
    try {
      var doc = obj.contentDocument;
      if (!doc || !doc.documentElement) return;
      var svg = doc.querySelector('svg');
      if (!svg) return;
      if (isDark) {
        svg.setAttribute('data-theme', 'dark');
        svg.classList.add('dark-theme');
      } else {
        svg.removeAttribute('data-theme');
        svg.classList.remove('dark-theme');
      }
    } catch (e) { /* cross-origin */ }
  }

  /* ── Code Tabs ── */
  function initCodeTabs() {
    document.querySelectorAll('.code-tabs').forEach(function (tabs) {
      var container = tabs.closest('.code-window');
      if (!container) return;

      var buttons = tabs.querySelectorAll('.code-tab');
      var panels = container.querySelectorAll('.code-panel');

      buttons.forEach(function (btn) {
        btn.addEventListener('click', function () {
          var target = this.getAttribute('data-tab');
          buttons.forEach(function (b) { b.classList.remove('active'); });
          panels.forEach(function (p) { p.classList.remove('active'); });
          this.classList.add('active');
          var panel = container.querySelector('.code-panel[data-tab="' + target + '"]');
          if (panel) panel.classList.add('active');
        });
      });
    });
  }

})();
