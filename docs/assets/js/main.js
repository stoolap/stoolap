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
    initHeroTerminal();

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

  /* ── Hero Terminal Typing Animation ── */
  function initHeroTerminal() {
    var output = document.getElementById('heroTermOutput');
    var typed = document.getElementById('heroTyped');
    var cursor = document.querySelector('.hero-cursor');
    var terminal = document.getElementById('heroTerminal');
    if (!output || !typed || !terminal) return;

    var scenes = [
      {
        queryText: "SELECT name, salary, RANK() OVER (ORDER BY salary DESC) FROM employees LIMIT 3;",
        queryHtml: '<span class="tk">SELECT</span> name, salary, <span class="tf">RANK</span>() <span class="tk">OVER</span> (<span class="tk">ORDER BY</span> salary <span class="tk">DESC</span>) <span class="tk">FROM</span> employees <span class="tk">LIMIT</span> <span class="tn">3</span>;',
        result: [
          '+-------+--------+------+',
          '| name  | salary | rank |',
          '+-------+--------+------+',
          '| Alice |  95000 |    1 |',
          '| Bob   |  87000 |    2 |',
          '| Carol |  82000 |    3 |',
          '+-------+--------+------+',
          '(3 rows, 0.3ms)'
        ]
      },
      {
        queryText: "SELECT * FROM accounts AS OF '2024-01-15 10:30:00' WHERE balance > 5000;",
        queryHtml: '<span class="tk">SELECT</span> * <span class="tk">FROM</span> accounts <span class="tk">AS OF</span> <span class="ts">\'2024-01-15 10:30:00\'</span> <span class="tk">WHERE</span> balance &gt; <span class="tn">5000</span>;',
        result: [
          '+----+-------+---------+',
          '| id | owner | balance |',
          '+----+-------+---------+',
          '|  3 | Carol |    8750 |',
          '|  1 | Alice |    5200 |',
          '+----+-------+---------+',
          '(2 rows, 0.1ms)'
        ]
      },
      {
        queryText: "EXPLAIN ANALYZE SELECT u.name, SUM(o.amount) FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.name;",
        queryHtml: '<span class="tk">EXPLAIN ANALYZE</span> <span class="tk">SELECT</span> u.name, <span class="tf">SUM</span>(o.amount) <span class="tk">FROM</span> users u <span class="tk">JOIN</span> orders o <span class="tk">ON</span> u.id = o.user_id <span class="tk">GROUP BY</span> u.name;',
        result: [
          'Hash Join  (cost=45.2, rows=1150) (actual: 0.4ms)',
          '  -> Parallel Seq Scan on orders  (workers=4)',
          '  -> Hash Build on users  (rows=50)',
          'Planning: 0.05ms  Execution: 0.6ms'
        ]
      },
      {
        queryText: "SELECT dept, COUNT(*), AVG(salary) FROM employees GROUP BY dept HAVING AVG(salary) > 60000;",
        queryHtml: '<span class="tk">SELECT</span> dept, <span class="tf">COUNT</span>(*), <span class="tf">AVG</span>(salary) <span class="tk">FROM</span> employees <span class="tk">GROUP BY</span> dept <span class="tk">HAVING</span> <span class="tf">AVG</span>(salary) &gt; <span class="tn">60000</span>;',
        result: [
          '+-------------+-------+----------+',
          '| dept        | count |      avg |',
          '+-------------+-------+----------+',
          '| Engineering |    12 | 89500.00 |',
          '| Marketing   |     8 | 72300.00 |',
          '+-------------+-------+----------+',
          '(2 rows, 0.2ms)'
        ]
      },
      {
        queryText: "WITH RECURSIVE org(id, name, lvl) AS (SELECT id, name, 0 FROM employees WHERE manager_id IS NULL UNION ALL SELECT e.id, e.name, o.lvl+1 FROM employees e JOIN org o ON e.manager_id = o.id) SELECT * FROM org;",
        queryHtml: '<span class="tk">WITH RECURSIVE</span> org(id, name, lvl) <span class="tk">AS</span> (<span class="tk">SELECT</span> id, name, <span class="tn">0</span> <span class="tk">FROM</span> employees <span class="tk">WHERE</span> manager_id <span class="tk">IS NULL</span> <span class="tk">UNION ALL</span> <span class="tk">SELECT</span> e.id, e.name, o.lvl+<span class="tn">1</span> <span class="tk">FROM</span> employees e <span class="tk">JOIN</span> org o <span class="tk">ON</span> e.manager_id = o.id) <span class="tk">SELECT</span> * <span class="tk">FROM</span> org;',
        result: [
          '+----+-------+-----+',
          '| id | name  | lvl |',
          '+----+-------+-----+',
          '|  1 | Alice |   0 |',
          '|  2 | Bob   |   1 |',
          '|  5 | Eve   |   2 |',
          '+----+-------+-----+',
          '(8 rows, 0.4ms)'
        ]
      },
      {
        queryText: "SELECT name, email FROM users WHERE id IN (SELECT user_id FROM orders WHERE amount > 500) ORDER BY name;",
        queryHtml: '<span class="tk">SELECT</span> name, email <span class="tk">FROM</span> users <span class="tk">WHERE</span> id <span class="tk">IN</span> (<span class="tk">SELECT</span> user_id <span class="tk">FROM</span> orders <span class="tk">WHERE</span> amount &gt; <span class="tn">500</span>) <span class="tk">ORDER BY</span> name;',
        result: [
          '+-------+-------------------+',
          '| name  | email             |',
          '+-------+-------------------+',
          '| Alice | alice@example.com |',
          '| Carol | carol@example.com |',
          '+-------+-------------------+',
          '(2 rows, 0.2ms)'
        ]
      },
      {
        queryText: "BEGIN; UPDATE accounts SET balance = balance - 200 WHERE id = 1; COMMIT;",
        queryHtml: '<span class="tk">BEGIN</span>; <span class="tk">UPDATE</span> accounts <span class="tk">SET</span> balance = balance - <span class="tn">200</span> <span class="tk">WHERE</span> id = <span class="tn">1</span>; <span class="tk">COMMIT</span>;',
        result: [
          'BEGIN',
          '1 row(s) affected',
          'COMMIT (0.1ms)'
        ]
      },
      {
        queryText: "SELECT category, ROUND(SUM(price), 2) AS total, COUNT(*) FROM products GROUP BY ROLLUP(category);",
        queryHtml: '<span class="tk">SELECT</span> category, <span class="tf">ROUND</span>(<span class="tf">SUM</span>(price), <span class="tn">2</span>) <span class="tk">AS</span> total, <span class="tf">COUNT</span>(*) <span class="tk">FROM</span> products <span class="tk">GROUP BY ROLLUP</span>(category);',
        result: [
          '+-------------+---------+-------+',
          '| category    |   total | count |',
          '+-------------+---------+-------+',
          '| Electronics | 2499.97 |     5 |',
          '| Clothing    |  389.85 |     4 |',
          '| Books       |   97.50 |     3 |',
          '| NULL        | 2987.32 |    12 |',
          '+-------------+---------+-------+',
          '(4 rows, 0.2ms)'
        ]
      },
      {
        queryText: "SELECT name, salary, salary - LAG(salary) OVER (ORDER BY salary DESC) AS gap FROM employees LIMIT 4;",
        queryHtml: '<span class="tk">SELECT</span> name, salary, salary - <span class="tf">LAG</span>(salary) <span class="tk">OVER</span> (<span class="tk">ORDER BY</span> salary <span class="tk">DESC</span>) <span class="tk">AS</span> gap <span class="tk">FROM</span> employees <span class="tk">LIMIT</span> <span class="tn">4</span>;',
        result: [
          '+-------+--------+-------+',
          '| name  | salary |   gap |',
          '+-------+--------+-------+',
          '| Alice |  95000 |  NULL |',
          '| Bob   |  87000 | -8000 |',
          '| Carol |  82000 | -5000 |',
          '| Dave  |  78000 | -4000 |',
          '+-------+--------+-------+',
          '(4 rows, 0.3ms)'
        ]
      },
      {
        queryText: "SELECT JSON_EXTRACT(metadata, '$.tags') AS tags FROM articles WHERE JSON_TYPE(metadata, '$.rating') = 'number';",
        queryHtml: '<span class="tk">SELECT</span> <span class="tf">JSON_EXTRACT</span>(metadata, <span class="ts">\'$.tags\'</span>) <span class="tk">AS</span> tags <span class="tk">FROM</span> articles <span class="tk">WHERE</span> <span class="tf">JSON_TYPE</span>(metadata, <span class="ts">\'$.rating\'</span>) = <span class="ts">\'number\'</span>;',
        result: [
          '+-------------------------+',
          '| tags                    |',
          '+-------------------------+',
          '| ["rust","database"]     |',
          '| ["sql","optimization"]  |',
          '| ["mvcc","transactions"] |',
          '+-------------------------+',
          '(3 rows, 0.1ms)'
        ]
      },
      {
        queryText: "SELECT LEFT(name, 1) AS initial, STRING_AGG(name, ', ') AS names FROM users GROUP BY LEFT(name, 1) ORDER BY initial;",
        queryHtml: '<span class="tk">SELECT</span> <span class="tf">LEFT</span>(name, <span class="tn">1</span>) <span class="tk">AS</span> initial, <span class="tf">STRING_AGG</span>(name, <span class="ts">\', \'</span>) <span class="tk">AS</span> names <span class="tk">FROM</span> users <span class="tk">GROUP BY</span> <span class="tf">LEFT</span>(name, <span class="tn">1</span>) <span class="tk">ORDER BY</span> initial;',
        result: [
          '+---------+-------------------+',
          '| initial | names             |',
          '+---------+-------------------+',
          '| A       | Alice, Alex       |',
          '| B       | Bob               |',
          '| C       | Carol, Charlie    |',
          '| D       | Dave, Diana       |',
          '+---------+-------------------+',
          '(4 rows, 0.2ms)'
        ]
      },
      {
        queryText: "SELECT * FROM orders WHERE amount BETWEEN 100 AND 500 AND created_at > CURRENT_DATE - INTERVAL '7 days';",
        queryHtml: '<span class="tk">SELECT</span> * <span class="tk">FROM</span> orders <span class="tk">WHERE</span> amount <span class="tk">BETWEEN</span> <span class="tn">100</span> <span class="tk">AND</span> <span class="tn">500</span> <span class="tk">AND</span> created_at &gt; <span class="tf">CURRENT_DATE</span> - <span class="tk">INTERVAL</span> <span class="ts">\'7 days\'</span>;',
        result: [
          '+----+---------+-----------+--------+---------------------+',
          '| id | user_id | product   | amount | created_at          |',
          '+----+---------+-----------+--------+---------------------+',
          '| 42 |       3 | Keyboard  | 149.99 | 2024-03-12 09:15:00 |',
          '| 47 |       1 | Headphone | 299.00 | 2024-03-14 14:30:00 |',
          '+----+---------+-----------+--------+---------------------+',
          '(2 rows, 0.1ms)'
        ]
      },
      {
        queryText: "CREATE TABLE metrics (id INTEGER PRIMARY KEY, ts TIMESTAMP, value FLOAT, CHECK(value >= 0));",
        queryHtml: '<span class="tk">CREATE TABLE</span> metrics (id <span class="tk">INTEGER PRIMARY KEY</span>, ts <span class="tk">TIMESTAMP</span>, value <span class="tk">FLOAT</span>, <span class="tk">CHECK</span>(value &gt;= <span class="tn">0</span>));',
        result: [
          'Table created.'
        ]
      },
      {
        queryText: "SELECT EXTRACT(MONTH FROM created_at) AS month, COUNT(*) AS orders FROM orders GROUP BY month ORDER BY orders DESC LIMIT 3;",
        queryHtml: '<span class="tk">SELECT</span> <span class="tf">EXTRACT</span>(<span class="tk">MONTH FROM</span> created_at) <span class="tk">AS</span> month, <span class="tf">COUNT</span>(*) <span class="tk">AS</span> orders <span class="tk">FROM</span> orders <span class="tk">GROUP BY</span> month <span class="tk">ORDER BY</span> orders <span class="tk">DESC LIMIT</span> <span class="tn">3</span>;',
        result: [
          '+-------+--------+',
          '| month | orders |',
          '+-------+--------+',
          '|    12 |    347 |',
          '|    11 |    312 |',
          '|     3 |    298 |',
          '+-------+--------+',
          '(3 rows, 0.4ms)'
        ]
      },
      {
        queryText: "SELECT name, CASE WHEN salary > 90000 THEN 'Senior' WHEN salary > 70000 THEN 'Mid' ELSE 'Junior' END AS level FROM employees;",
        queryHtml: '<span class="tk">SELECT</span> name, <span class="tk">CASE WHEN</span> salary &gt; <span class="tn">90000</span> <span class="tk">THEN</span> <span class="ts">\'Senior\'</span> <span class="tk">WHEN</span> salary &gt; <span class="tn">70000</span> <span class="tk">THEN</span> <span class="ts">\'Mid\'</span> <span class="tk">ELSE</span> <span class="ts">\'Junior\'</span> <span class="tk">END AS</span> level <span class="tk">FROM</span> employees;',
        result: [
          '+-------+--------+',
          '| name  | level  |',
          '+-------+--------+',
          '| Alice | Senior |',
          '| Bob   | Mid    |',
          '| Carol | Mid    |',
          '| Dave  | Mid    |',
          '| Eve   | Junior |',
          '+-------+--------+',
          '(5 rows, 0.1ms)'
        ]
      }
    ];

    var sceneIndex = 0;
    var charIndex = 0;
    var typingTimer = null;
    var paused = false;
    var CHAR_DELAY = 25;
    var RESULT_LINE_DELAY = 40;
    var SCENE_PAUSE = 2500;

    // Pause animation when hero is off-screen
    if ('IntersectionObserver' in window) {
      var observer = new IntersectionObserver(function (entries) {
        paused = !entries[0].isIntersecting;
      }, { threshold: 0.1 });
      observer.observe(terminal);
    }

    // Reduced motion: show static content
    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      showStatic();
      return;
    }

    function showStatic() {
      var scene = scenes[0];
      var q = document.createElement('div');
      q.className = 'hero-term-line is-query';
      q.innerHTML = '<span class="hero-prompt">stoolap&gt; </span>' + scene.queryHtml;
      output.appendChild(q);
      scene.result.forEach(function (line) {
        var el = document.createElement('div');
        el.className = 'hero-term-line is-result';
        el.textContent = line;
        output.appendChild(el);
      });
      if (cursor) cursor.style.display = 'none';
    }

    var MAX_OUTPUT_LINES = 40;

    function scrollToBottom() {
      terminal.scrollTop = terminal.scrollHeight;
    }

    function trimOldLines() {
      while (output.children.length > MAX_OUTPUT_LINES) {
        output.removeChild(output.firstChild);
      }
    }

    function typeNextChar() {
      if (paused) { typingTimer = setTimeout(typeNextChar, 200); return; }
      var scene = scenes[sceneIndex];
      if (charIndex < scene.queryText.length) {
        typed.textContent += scene.queryText[charIndex];
        charIndex++;
        scrollToBottom();
        typingTimer = setTimeout(typeNextChar, CHAR_DELAY);
      } else {
        if (cursor) cursor.classList.remove('typing');
        setTimeout(showResults, 300);
      }
    }

    function showResults() {
      var scene = scenes[sceneIndex];

      // Add colorized query to output
      var queryEl = document.createElement('div');
      queryEl.className = 'hero-term-line is-query';
      queryEl.innerHTML = '<span class="hero-prompt">stoolap&gt; </span>' + scene.queryHtml;
      output.appendChild(queryEl);

      // Clear typed text
      typed.textContent = '';
      scrollToBottom();

      // Show result lines one by one
      var ri = 0;
      function nextLine() {
        if (ri < scene.result.length) {
          var el = document.createElement('div');
          el.className = 'hero-term-line is-result';
          el.textContent = scene.result[ri];
          output.appendChild(el);
          ri++;
          scrollToBottom();
          setTimeout(nextLine, RESULT_LINE_DELAY);
        } else {
          // Blank line between scenes
          var blank = document.createElement('div');
          blank.className = 'hero-term-line';
          blank.innerHTML = '\u00a0';
          output.appendChild(blank);
          scrollToBottom();
          trimOldLines();
          setTimeout(nextScene, SCENE_PAUSE);
        }
      }
      nextLine();
    }

    function nextScene() {
      sceneIndex = (sceneIndex + 1) % scenes.length;
      charIndex = 0;
      if (cursor) cursor.classList.add('typing');
      typeNextChar();
    }

    // Start after page settles
    setTimeout(function () {
      if (cursor) cursor.classList.add('typing');
      typeNextChar();
    }, 1000);
  }

})();
