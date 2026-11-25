#!/usr/bin/env bash
# Render a report's markdown to PDF. It needs no conda environment, because `pandoc` and `xelatex`
# are system binaries and DejaVu is a system font.
#
#     bash experiments/analysis/render_pdf.sh <run>/report/REPORT.md
#
# The default is portrait, the layout a reader expects of a report. The figures are drawn at aspect
# ratios that fit a portrait column, and `page_layout.lua` gives the widest of them a rotated page
# of their own instead of shrinking them past reading size. That filter also starts each appendix
# on a fresh page, so the report itself says nothing about where a page ends.
#
# `REPORT_ORIENTATION=landscape` renders the whole document landscape instead, for a dataset whose
# tables are too wide for portrait.
#
# `--columns` is set explicitly, because pandoc gives a pipe table explicit column widths only when
# the table's source is wider than this value, and otherwise leaves it at its natural width. A table
# at its natural width runs off the page when its cells hold long monospace names, because a
# character of `feature_name_like_this` is far wider than a character of `0.998`. So the value sits
# below the widest table known to fit unaided: every wider table is scaled to the text block and
# wraps inside its cells, and only narrow tables stay compact. Too low a value only stretches a
# narrow table to the full width, whereas too high a value lets a table run into the margin, so the
# value errs low.
set -euo pipefail

ORIENTATION="${REPORT_ORIENTATION:-portrait}"
if [ "$ORIENTATION" = "landscape" ]; then
    MARGIN=1.5cm
    GEOMETRY=(-V geometry:a4paper -V geometry:landscape -V "geometry:margin=$MARGIN")
    COLUMNS=155
else
    MARGIN=1.9cm
    GEOMETRY=(-V geometry:a4paper -V "geometry:margin=$MARGIN")
    COLUMNS=100
fi

# pandoc takes the header by its content, so the file needs no `.tex` suffix, which macOS's mktemp
# cannot add.
HEADER="$(mktemp "${TMPDIR:-/tmp}/seqmaestro_header.XXXXXX")"
trap 'rm -f "$HEADER"' EXIT
cat > "$HEADER" <<'TEX'
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{ragged2e}
% Tables hold much of the report's substance, and there are many of them, so they are set one step
% down in size with a little more leading between rows, which keeps a page of them readable.
\AtBeginEnvironment{longtable}{\footnotesize}
\renewcommand{\arraystretch}{1.18}
\setlength{\tabcolsep}{3.5pt}
\setlength{\LTpre}{6pt}
\setlength{\LTpost}{10pt}
% Feature names are long snake_case identifiers sitting in narrow columns, and TeX has no break
% point inside one, so without this they run straight into the column beside them. A zero-width
% break opportunity after every underscore lets the name wrap and adds nothing to the page.
\let\origtextunderscore\_
\renewcommand{\_}{\origtextunderscore\allowbreak}
% Figures stay where the text puts them. Left to float, a full-width panel is deferred past the
% section that introduces it and sometimes past the end of the document.
\usepackage{float}
\floatplacement{figure}{H}
% A few panels are much wider than they are tall and cannot be read at a portrait column width.
% Those get a rotated page of their own, which `page_layout.lua` gives them.
\usepackage{pdflscape}
% A heading needs room for itself and the start of what follows. A table cannot split a row, so
% when its header and first row do not fit below a heading the whole table moves to the next page
% and leaves the heading alone at the foot of this one. Each heading therefore starts a new page
% when too few lines are left: six above prose, and twelve at the lowest level, which heads each
% configuration's tables, whose rows wrap to nine lines in the narrow description column. A
% lowest-level heading placed straight under another heading skips the check, because the check
% leaves a place where the page may break, which there would fall between the two headings, so on a
% page with no better break the upper heading would be left alone at its foot.
\usepackage{etoolbox}
\usepackage{needspace}
\pretocmd{\section}{\needspace{6\baselineskip}}{}{}
\pretocmd{\subsection}{\needspace{6\baselineskip}}{}{}
\makeatletter
\pretocmd{\subsubsection}{\if@nobreak\else\needspace{12\baselineskip}\fi}{}{}
\makeatother
% Block quotes hold the marked interpretation blocks, so they must look like a distinct object
% rather than like indented prose.
\usepackage{framed}
\usepackage{xcolor}
\definecolor{noteedge}{HTML}{2A78D6}
\definecolor{noteback}{HTML}{F4F7FC}
\renewenvironment{quote}{%
  \def\FrameCommand{\textcolor{noteedge}{\vrule width 2pt}\hspace{8pt}}%
  \MakeFramed{\advance\hsize-\width \FrameRestore}\small}%
  {\endMakeFramed}
\sloppy
\emergencystretch=3em
TEX

HERE="$(cd "$(dirname "$0")" && pwd)"

for source in "$@"; do
    target="${source%.md}.pdf"
    # Figures `build_report_figures.py` measured as too wide for a portrait column get a rotated
    # page. The list comes from the build rather than from the report, so nothing about page
    # geometry reaches the markdown.
    REGISTRY="$(dirname "$source")/figures.json"
    WIDE=""
    if [ -f "$REGISTRY" ]; then
        WIDE="$(python3 -c "
import json, sys
figures = json.load(open(sys.argv[1]))['figures']
print(','.join(v['path'].split('/')[-1] for v in figures.values() if v.get('rotate')))
" "$REGISTRY")"
    fi
    pandoc "$source" \
        --lua-filter="$HERE/page_layout.lua" \
        -M wide-figures="$WIDE" \
        -M resource-dir="$(cd "$(dirname "$source")" && pwd)" \
        -M wide-figure-width="\\dimexpr\\paperheight-2\\dimexpr${MARGIN}\\relax\\relax" \
        -M page-break-before="^Appendix" \
        --from=markdown+pipe_tables-raw_html \
        --to=pdf --pdf-engine=xelatex \
        --resource-path="$(dirname "$source")" \
        --columns="$COLUMNS" \
        --toc --toc-depth=2 \
        --include-in-header="$HEADER" \
        "${GEOMETRY[@]}" \
        -V documentclass=article \
        -V fontsize=10pt \
        -V mainfont="DejaVu Serif" \
        -V sansfont="DejaVu Sans" \
        -V monofont="DejaVu Sans Mono" \
        -V monofontoptions="Scale=0.80" \
        -V colorlinks=true \
        -V linkcolor=NavyBlue \
        -V urlcolor=NavyBlue \
        -o "$target"
    echo "$target"
done
