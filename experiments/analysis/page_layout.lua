--[[
Page layout for the PDF: where a page break falls, and which figures get a rotated page.

The report is plain markdown and says nothing about page geometry, because the same document
renders on a web page, in a portrait PDF and in a landscape one, and only the PDFs care how wide a
panel is. So the decision arrives as metadata: `wide-figures`, a comma-separated list of image file
names that `build_report_figures.py` measured as too wide.

`pdflscape` turns the page in the viewer and leaves the text block alone, so a figure inside it at
`\linewidth` comes out the width it would have been in portrait, only turned. The horizontal extent
of the turned page is the paper height less the margins, and `render_pdf.sh` passes that as
`wide-figure-width` so the margin is stated in one place.

`page-break-before` is a Lua pattern matched against level 1 and 2 headings, so the appendices start
on a fresh page without the report saying so.

Both happen only on the LaTeX path. Any other output format sees the document unchanged.
]]

local wide = {}
local rotating = false
local resource_dir = nil
local figure_width = "\\linewidth"
local break_before = nil

--- Characters LaTeX reads as syntax, in a caption that arrives as plain text.
local function escape(text)
  return (text:gsub("([&%%%$#_{}])", "\\%1"):gsub("~", "\\textasciitilde{}")
              :gsub("%^", "\\textasciicircum{}"))
end

function Meta(meta)
  if meta["wide-figures"] then
    for name in pandoc.utils.stringify(meta["wide-figures"]):gmatch("[^,]+") do
      wide[name:match("^%s*(.-)%s*$")] = true
    end
  end
  if meta["resource-dir"] then
    resource_dir = pandoc.utils.stringify(meta["resource-dir"])
  end
  if meta["wide-figure-width"] then
    figure_width = pandoc.utils.stringify(meta["wide-figure-width"])
  end
  if meta["page-break-before"] then
    local pattern = pandoc.utils.stringify(meta["page-break-before"])
    if pattern ~= "" then
      break_before = pattern
    end
  end
  rotating = FORMAT:match("latex") ~= nil
end

--- A raw LaTeX block bypasses the resource path pandoc resolves images against, so a relative
--- source has to be made absolute here or the compiler will not find it.
local function resolve(src)
  if resource_dir and not src:match("^/") then
    return resource_dir .. "/" .. src
  end
  return src
end


--- The image inside a figure, when that image is one the build marked as wide.
local function wide_image(block)
  local found = nil
  pandoc.walk_block(block, {
    Image = function(image)
      if wide[image.src:match("([^/]+)$") or ""] then
        found = image
      end
    end,
  })
  return found
end

function Figure(figure)
  if not rotating then
    return nil
  end
  local image = wide_image(figure)
  if not image then
    return nil
  end
  local caption = escape(pandoc.utils.stringify(figure.caption.long or figure.caption))
  return pandoc.RawBlock("latex", table.concat({
    "\\begin{landscape}",
    "\\begin{figure}[H]",
    "\\centering",
    "\\includegraphics[width=" .. figure_width .. "]{" .. resolve(image.src) .. "}",
    "\\caption{" .. caption .. "}",
    "\\end{figure}",
    "\\end{landscape}",
  }, "\n"))
end

function Header(header)
  if not rotating or not break_before or header.level > 2 then
    return nil
  end
  if not pandoc.utils.stringify(header.content):match(break_before) then
    return nil
  end
  -- `\clearpage` and not `\newpage`, because a figure held back by an earlier page has to come out
  -- before the new section starts, or it is placed under a heading it does not belong to.
  return { pandoc.RawBlock("latex", "\\clearpage"), header }
end

-- Meta runs after the body by default, so the settings have to be read first.
return {{ Meta = Meta }, { Figure = Figure, Header = Header }}
