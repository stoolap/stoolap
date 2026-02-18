# stoolap.io

The [stoolap.io](https://stoolap.io) website — documentation, interactive playground, and blog for the Stoolap database engine.

Built with [Jekyll](https://jekyllrb.com/) 4.4 and deployed automatically via GitHub Pages.

## Local Development

```bash
cd docs
gem install jekyll bundler   # one-time setup
bundle install               # install dependencies
bundle exec jekyll serve     # http://localhost:4000
```

## Structure

```
docs/
├── index.html               # Homepage (hero, features, comparison table)
├── playground.html           # Interactive SQL playground (WebAssembly)
├── blog/index.html           # Blog listing
├── docs/index.html           # Documentation hub
├── 404.html                  # Custom error page
├── _config.yml               # Jekyll configuration (domain, collections, plugins)
├── _data/
│   └── doc_categories.yml    # Documentation category definitions
├── _layouts/
│   ├── default.html          # Base layout (nav, footer, theme toggle)
│   ├── doc.html              # Documentation page (sidebar, prev/next nav)
│   └── post.html             # Blog post
├── _docs/                    # 51 documentation pages in 7 categories
│   ├── getting-started/      # Installation, quickstart, API reference
│   ├── architecture/         # MVCC, storage engine, indexes, persistence
│   ├── data-types/           # Types overview, date/time, JSON
│   ├── sql-commands/         # SQL reference, SHOW, PRAGMA, DESCRIBE
│   ├── functions/            # Scalar, aggregate, window functions
│   ├── sql-features/         # Transactions, CTEs, joins, window funcs, etc.
│   └── performance/          # Optimizer, parallel execution, semantic cache
├── assets/
│   ├── css/                  # main.css, home.css, code.css, playground.css
│   ├── js/                   # main.js, playground.js
│   ├── img/                  # logo.svg, stoolap_logo.png, architecture-diagram.svg
│   └── wasm/                 # WebAssembly build (stoolap_bg.wasm, JS bindings)
├── Gemfile                   # Ruby dependencies
└── CNAME                     # stoolap.io domain
```

## Adding Documentation

1. Create a Markdown file in the appropriate `_docs/<category>/` folder.
2. Add front matter:

```yaml
---
title: Your Page Title
category: Category Name    # must match a name in _data/doc_categories.yml
order: 5                   # controls sort order within the category
---
```

3. Write content in Markdown. Use `sql` fenced code blocks for SQL examples.

Categories are defined in `_data/doc_categories.yml`. The sidebar and prev/next navigation are generated automatically from the category and order fields.

## Updating the Playground

The playground runs Stoolap compiled to WebAssembly entirely in the browser. To update the WASM build:

1. Build the WASM package from the project root (requires `wasm-pack`).
2. Copy the output into `assets/wasm/`.

## Deployment

Pushing to the `main` branch triggers automatic deployment via GitHub Pages. The `CNAME` file maps to [stoolap.io](https://stoolap.io).

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Static site generator | Jekyll 4.4 |
| Hosting | GitHub Pages |
| Fonts | Rubik (body), JetBrains Mono (code) |
| Theme | Light/dark mode with system preference detection |
| Playground | WebAssembly (Rust compiled via `wasm-pack`) |
| Plugins | jekyll-sitemap, jekyll-seo-tag |