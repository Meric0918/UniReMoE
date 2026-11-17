#!/usr/bin/env bash
# Compile from any working directory; keep source paths relative to main.tex.
set -euo pipefail
cd -- "$(dirname -- "$0")"

tectonic_bin="${CODEX_TECTONIC_PATH:-}"
if [[ ! -x "$tectonic_bin" ]]; then
  tectonic_bin="$(command -v tectonic || true)"
fi
if [[ ! -x "$tectonic_bin" ]]; then
  for candidate in /Applications/ChatGPT.app/Contents/Resources/tectonic/tectonic /Applications/Codex.app/Contents/Resources/tectonic/tectonic; do
    if [[ -x "$candidate" ]]; then
      tectonic_bin="$candidate"
      break
    fi
  done
fi

if [[ -x "$tectonic_bin" ]]; then
  # Reuse the desktop preview cache when using its bundled engine.
  if [[ "$tectonic_bin" == /Applications/ChatGPT.app/* && -z "${TECTONIC_CACHE_DIR:-}" && -d "$HOME/Library/Application Support/ChatGPT/latex/tectonic" ]]; then
    export TECTONIC_CACHE_DIR="$HOME/Library/Application Support/ChatGPT/latex/tectonic"
  fi
  exec "$tectonic_bin" -X compile --untrusted --keep-logs main.tex
fi

if [[ -d /Library/TeX/texbin ]]; then
  export PATH="/Library/TeX/texbin:$PATH"
fi
if command -v pdflatex >/dev/null && command -v bibtex >/dev/null; then
  pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
else
  echo 'No usable compiler: install Tectonic or TeX Live/MacTeX, or set CODEX_TECTONIC_PATH.' >&2
  exit 127
fi
