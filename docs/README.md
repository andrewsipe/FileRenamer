# Docs scaffold for FileRenamer

Starter GitHub Pages site from the monorepo **DocsTemplate** package.

| | |
|---|---|
| Pages URL (once enabled) | https://andrewsipe.github.io/FileRenamer/ |
| Repo | https://github.com/andrewsipe/FileRenamer |

## Next steps

1. Replace placeholder copy in `index.html` with a short product intro + install.
2. Duplicate `page.html` for real topics (concepts, flags, recipes, …); wire them in the nav.
3. Tune accent tokens at the top of `styles.css` if the assigned theme doesn’t fit.
4. Slim the README to install + quick start and link here.
5. Enable Pages: Settings → Pages → `main` / `/docs`.
6. Set the repo About website field to the Pages URL.

Do not expand this into a large site — README stays the front door; this holds advanced detail.

To refresh styles from the monorepo template (keeps your HTML):

```bash
python "../DocsTemplate/scaffold.py" "FileRenamer" --styles-only
```

(Path relative to this project; adjust if your layout differs.)
