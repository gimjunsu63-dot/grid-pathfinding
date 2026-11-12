"""Small SVG chart helpers for the report. Standard library only.

Three kinds of figures:
    LinePlot     lines / scatter on linear or log axes, optional error bars
    bar_chart()  grouped bars with error whiskers, linear or log y
    bar_panels() small multiples of single-color bars

Style rules shared by every figure (so the report reads as one system):
  - one color per algorithm, the same everywhere (ALGO_COLORS)
  - values and labels in ink colors, never in the series color
  - thin marks: 2px lines, 8px markers with a surface ring, bars with a
    small gap and rounded data ends
  - recessive grid and axes; a legend whenever there is more than one series
  - every mark carries a <title>, so hovering in a browser shows its value
"""

import math
from xml.sax.saxutils import escape

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

# Validated categorical set (CVD-safe for all pairs of the first three).
ALGO_COLORS = {
    "bfs": "#2a78d6",        # blue
    "astar": "#eb6834",      # orange
    "dijkstra": "#1baf7a",   # aqua
    "bidir": "#6250d6",      # violet
}


def tint(hex_color, amount):
    """Mix a color with white; amount 0 = same color, 1 = white."""
    h = hex_color.lstrip("#")
    rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    mixed = [round(c + (255 - c) * amount) for c in rgb]
    return "#" + "".join(f"{c:02x}" for c in mixed)


def _fmt(v):
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1_000_000:
        return f"{v / 1_000_000:g}M"
    if a >= 10000:
        return f"{v / 1000:g}k" if v % 1000 == 0 else f"{v:,.0f}"
    if a >= 100:
        return f"{v:,.0f}"
    if a >= 1:
        return f"{v:g}"
    return f"{v:.2g}"


def nice_ticks(lo, hi, n=5):
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    first = math.floor(lo / step + 1e-9)
    last = math.ceil(hi / step - 1e-9)
    return [round(k * step, 10) for k in range(first, last + 1)]


def log_ticks(lo, hi):
    a, b = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    majors = [10 ** k for k in range(a, b + 1)]
    if b - a <= 2:
        majors = sorted(m * f for m in majors for f in (1, 2, 5))
    return [t for t in majors if lo / 1.0001 <= t <= hi * 1.0001]


class _Axis:
    def __init__(self, lo, hi, log, pix_lo, pix_hi, ticks=None):
        if log:
            lo = 10 ** math.floor(math.log10(lo)) if ticks is None else lo
            hi = 10 ** math.ceil(math.log10(hi)) if ticks is None else hi
            self.ticks = ticks or log_ticks(lo, hi)
        else:
            self.ticks = ticks or nice_ticks(lo, hi)
            lo, hi = min(lo, self.ticks[0]), max(hi, self.ticks[-1])
        self.lo, self.hi, self.log = lo, hi, log
        self.pix_lo, self.pix_hi = pix_lo, pix_hi

    def __call__(self, v):
        if self.log:
            v = max(v, self.lo)
            t = (math.log10(v) - math.log10(self.lo)) / (math.log10(self.hi) - math.log10(self.lo))
        else:
            t = (v - self.lo) / ((self.hi - self.lo) or 1)
        return self.pix_lo + t * (self.pix_hi - self.pix_lo)


class _Svg:
    def __init__(self, width, height):
        self.w, self.h = width, height
        self.parts = []

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=12, color=INK_2, anchor="start", weight="normal", rotate=None):
        tr = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate else ""
        lines = str(s).split("\n")
        for i, line in enumerate(lines):
            self.add(f'<text x="{x:.1f}" y="{y + i * (size + 2):.1f}" font-size="{size}" fill="{color}" '
                     f'text-anchor="{anchor}" font-weight="{weight}"{tr}>{escape(line)}</text>')

    def render(self):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}" font-family="{FONT}">'
                f'<rect width="100%" height="100%" fill="{SURFACE}"/>')
        return head + "".join(self.parts) + "</svg>\n"


def _legend(svg, items, x, y):
    """items: list of (label, color, kind) with kind 'line' | 'box'. One row."""
    cx = x
    for label, color, kind in items:
        if kind == "line":
            svg.add(f'<line x1="{cx}" y1="{y - 4}" x2="{cx + 18}" y2="{y - 4}" stroke="{color}" stroke-width="2"/>')
            svg.add(f'<circle cx="{cx + 9}" cy="{y - 4}" r="4" fill="{color}" stroke="{SURFACE}" stroke-width="2"/>')
            cx += 24
        else:
            svg.add(f'<rect x="{cx}" y="{y - 10}" width="12" height="12" rx="3" fill="{color}"/>')
            cx += 17
        svg.text(cx, y, label, size=12, color=INK_2)
        cx += 7 * len(label) + 18


def _header(svg, title, subtitle, left):
    svg.text(left, 22, title, size=15, color=INK, weight="600")
    if subtitle:
        svg.text(left, 40, subtitle, size=12, color=INK_2)


def _y_axis(svg, ax, left, right, label, fmt=_fmt):
    for t in ax.ticks:
        y = ax(t)
        svg.add(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        svg.text(left - 6, y + 4, fmt(t), size=11, color=MUTED, anchor="end")
    if label:
        mid = (ax.pix_lo + ax.pix_hi) / 2
        svg.text(16, mid, label, size=12, color=INK_2, anchor="middle", rotate=-90)


# ---------------------------------------------------------------------------
# line / scatter plot
# ---------------------------------------------------------------------------


class LinePlot:
    def __init__(self, title, xlabel, ylabel, subtitle=None, width=680, height=420,
                 xlog=False, ylog=False, xticks=None, yticks=None, xfmt=_fmt, yfmt=_fmt,
                 xlim=None, ylim=None):
        self.title, self.subtitle = title, subtitle
        self.xlabel, self.ylabel = xlabel, ylabel
        self.width, self.height = width, height
        self.xlog, self.ylog = xlog, ylog
        self.xticks, self.yticks = xticks, yticks
        self.xfmt, self.yfmt = xfmt, yfmt
        self.xlim, self.ylim = xlim, ylim
        self.series = []
        self.notes = []
        self.refs = []

    def line(self, xs, ys, color, label, lo=None, hi=None, end_label=None, dashed=False):
        self.series.append(dict(kind="line", xs=xs, ys=ys, color=color, label=label,
                                lo=lo, hi=hi, end_label=end_label, dashed=dashed))

    def scatter(self, xs, ys, color, label=None, r=3.5, opacity=0.75, titles=None):
        self.series.append(dict(kind="scatter", xs=xs, ys=ys, color=color, label=label,
                                r=r, opacity=opacity, titles=titles))

    def note(self, x, y, s, anchor="start", dy=0):
        self.notes.append((x, y, s, anchor, dy))

    def reference(self, xs, ys, label):
        """Dashed gray guide line (for example y = x)."""
        self.refs.append((xs, ys, label))

    def render(self):
        left, right = 72, self.width - 120
        top, bottom = 70 if self.subtitle else 56, self.height - 52
        xs = [x for s in self.series for x in s["xs"]]
        ys = [y for s in self.series for y in s["ys"]]
        ys += [v for s in self.series for v in (s.get("lo") or []) + (s.get("hi") or [])]
        xlo, xhi = self.xlim or (min(xs), max(xs))
        ylo, yhi = self.ylim or (min(ys), max(ys))
        if not self.ylog and self.ylim is None:
            ylo = min(0, ylo)
        ax_x = _Axis(xlo, xhi, self.xlog, left, right, self.xticks)
        ax_y = _Axis(ylo, yhi, self.ylog, bottom, top, self.yticks)
        svg = _Svg(self.width, self.height)
        _header(svg, self.title, self.subtitle, left - 56)
        _y_axis(svg, ax_y, left, right, self.ylabel, self.yfmt)
        for t in ax_x.ticks:
            x = ax_x(t)
            svg.add(f'<line x1="{x:.1f}" y1="{bottom}" x2="{x:.1f}" y2="{bottom + 4}" stroke="{AXIS}"/>')
            svg.text(x, bottom + 18, self.xfmt(t), size=11, color=MUTED, anchor="middle")
        svg.add(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="{AXIS}" stroke-width="1"/>')
        svg.text((left + right) / 2, self.height - 12, self.xlabel, size=12, color=INK_2, anchor="middle")

        for rx, ry, label in self.refs:
            pts = " ".join(f"{ax_x(a):.1f},{ax_y(b):.1f}" for a, b in zip(rx, ry))
            svg.add(f'<polyline points="{pts}" fill="none" stroke="{MUTED}" stroke-width="1.5" stroke-dasharray="5 4"/>')
            if label:
                svg.text(ax_x(rx[-1]) + 6, ax_y(ry[-1]) + 4, label, size=11, color=MUTED)

        legend = []
        for s in self.series:
            c = s["color"]
            if s["kind"] == "scatter":
                for i, (a, b) in enumerate(zip(s["xs"], s["ys"])):
                    tip = s["titles"][i] if s.get("titles") else f"{a:g}, {b:g}"
                    svg.add(f'<circle cx="{ax_x(a):.1f}" cy="{ax_y(b):.1f}" r="{s["r"]}" fill="{c}" '
                            f'fill-opacity="{s["opacity"]}" stroke="{SURFACE}" stroke-width="1">'
                            f'<title>{escape(tip)}</title></circle>')
                if s["label"]:
                    legend.append((s["label"], c, "box"))
                continue
            if s["lo"] and s["hi"]:
                for a, l, h in zip(s["xs"], s["lo"], s["hi"]):
                    x = ax_x(a)
                    svg.add(f'<line x1="{x:.1f}" y1="{ax_y(l):.1f}" x2="{x:.1f}" y2="{ax_y(h):.1f}" '
                            f'stroke="{c}" stroke-width="1.5" stroke-opacity="0.6"/>')
            pts = " ".join(f"{ax_x(a):.1f},{ax_y(b):.1f}" for a, b in zip(s["xs"], s["ys"]))
            dash = ' stroke-dasharray="6 4"' if s["dashed"] else ""
            svg.add(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2"{dash}/>')
            for a, b in zip(s["xs"], s["ys"]):
                svg.add(f'<circle cx="{ax_x(a):.1f}" cy="{ax_y(b):.1f}" r="4" fill="{c}" stroke="{SURFACE}" '
                        f'stroke-width="2"><title>{escape(s["label"])}: {b:.4g} at {a:g}</title></circle>')
            if s["end_label"]:
                svg.text(ax_x(s["xs"][-1]) + 8, ax_y(s["ys"][-1]) + 4, s["end_label"], size=11, color=INK_2)
            legend.append((s["label"], c, "line"))

        for a, b, s, anchor, dy in self.notes:
            svg.text(ax_x(a), ax_y(b) + dy, s, size=11, color=INK_2, anchor=anchor)
        if len(legend) > 1:
            _legend(svg, legend, left, top - 12)
        return svg.render()


# ---------------------------------------------------------------------------
# grouped bars
# ---------------------------------------------------------------------------


def _bar_path(x, w, y_base, y_top, r=4):
    """Bar with rounded corners only at the data end (top)."""
    h = y_base - y_top
    if h <= 0:
        return ""
    r = min(r, w / 2, h)
    return (f"M{x:.1f},{y_base:.1f} L{x:.1f},{y_top + r:.1f} Q{x:.1f},{y_top:.1f} {x + r:.1f},{y_top:.1f} "
            f"L{x + w - r:.1f},{y_top:.1f} Q{x + w:.1f},{y_top:.1f} {x + w:.1f},{y_top + r:.1f} "
            f"L{x + w:.1f},{y_base:.1f} Z")


def bar_chart(title, categories, series, ylabel, subtitle=None, ylog=False, width=760, height=420,
              yfmt=_fmt, value_labels=False, value_fmt=None, ylim=None):
    """series: list of dicts {label, color, values, lo (opt), hi (opt)}; one value per category."""
    left, right = 72, width - 20
    top, bottom = 76 if subtitle else 62, height - 64
    vals = [v for s in series for v in s["values"] if v is not None]
    vals += [v for s in series for v in (s.get("hi") or []) if v is not None]
    if ylim:
        ylo, yhi = ylim
    elif ylog:
        ylo, yhi = min(v for v in vals if v > 0), max(vals)
    else:
        ylo, yhi = 0, max(vals) * 1.08
    ax_y = _Axis(ylo, yhi, ylog, bottom, top)
    svg = _Svg(width, height)
    _header(svg, title, subtitle, left - 56)
    _y_axis(svg, ax_y, left, right, ylabel, yfmt)

    n_cat, n_ser = len(categories), len(series)
    slot = (right - left) / n_cat
    group = slot * 0.78
    gap = 2
    bw = (group - gap * (n_ser - 1)) / n_ser
    base = ax_y(ax_y.lo)
    for i, cat in enumerate(categories):
        gx = left + i * slot + (slot - group) / 2
        for j, s in enumerate(series):
            v = s["values"][i]
            if v is None:
                continue
            x = gx + j * (bw + gap)
            y = ax_y(v)
            svg.add(f'<path d="{_bar_path(x, bw, base, y)}" fill="{s["color"]}">'
                    f'<title>{escape(s["label"])} | {escape(cat.replace(chr(10), " "))}: {v:.4g}</title></path>')
            lo, hi = s.get("lo"), s.get("hi")
            if lo and hi and lo[i] is not None and hi[i] is not None:
                xm = x + bw / 2
                svg.add(f'<line x1="{xm:.1f}" y1="{ax_y(lo[i]):.1f}" x2="{xm:.1f}" y2="{ax_y(hi[i]):.1f}" '
                        f'stroke="{INK_2}" stroke-width="1.2"/>')
            if value_labels:
                txt = (value_fmt or _fmt)(v)
                label_y = min(y, ax_y(hi[i])) if hi and hi[i] is not None else y
                svg.text(x + bw / 2, label_y - 5, txt, size=10, color=INK_2, anchor="middle")
        svg.text(left + i * slot + slot / 2, bottom + 18, cat, size=11, color=INK_2, anchor="middle")
    svg.add(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="{AXIS}" stroke-width="1"/>')
    if n_ser > 1:
        _legend(svg, [(s["label"], s["color"], "box") for s in series], left, top - 14)
    return svg.render()


def bar_panels(title, panels, ylabel, subtitle=None, color=ALGO_COLORS["astar"], width=900,
               height=330, ylim=None, yfmt=_fmt):
    """Small multiples. panels: list of {title, categories, values, labels (text above bars)}."""
    left_pad, top = 64, 86 if subtitle else 70
    bottom = height - 46
    n = len(panels)
    pw = (width - left_pad - 16) / n
    vals = [v for p in panels for v in p["values"]]
    ylo, yhi = ylim or (0, max(vals) * 1.15)
    svg = _Svg(width, height)
    _header(svg, title, subtitle, 8)
    ax_y = _Axis(ylo, yhi, False, bottom, top)
    for t in ax_y.ticks:
        y = ax_y(t)
        svg.add(f'<line x1="{left_pad}" y1="{y:.1f}" x2="{width - 16}" y2="{y:.1f}" stroke="{GRID}"/>')
        svg.text(left_pad - 6, y + 4, yfmt(t), size=11, color=MUTED, anchor="end")
    svg.text(16, (top + bottom) / 2, ylabel, size=12, color=INK_2, anchor="middle", rotate=-90)
    for k, p in enumerate(panels):
        x0 = left_pad + k * pw + 8
        inner = pw - 16
        svg.text(x0 + inner / 2, top - 12, p["title"], size=12, color=INK, anchor="middle", weight="600")
        m = len(p["values"])
        slot = inner / m
        bw = slot * 0.72
        for i, v in enumerate(p["values"]):
            x = x0 + i * slot + (slot - bw) / 2
            y = ax_y(v)
            svg.add(f'<path d="{_bar_path(x, bw, ax_y(ylo), y)}" fill="{p.get("color", color)}">'
                    f'<title>{escape(p["title"])} | {escape(p["categories"][i])}: {v:.4g}</title></path>')
            if p.get("labels"):
                svg.text(x + bw / 2, y - 5, p["labels"][i], size=10, color=INK_2, anchor="middle")
            svg.text(x + bw / 2, bottom + 16, p["categories"][i], size=11, color=MUTED, anchor="middle")
        svg.add(f'<line x1="{x0}" y1="{bottom}" x2="{x0 + inner}" y2="{bottom}" stroke="{AXIS}"/>')
    return svg.render()
