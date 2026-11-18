"""Images shown by the demo: overlay, probability map, disagreement map and error map.

Colours follow the report figures (src/plots.py): green for agreement with the ground truth,
orange for false positives, blue for missed polyp. Orange/blue stay distinguishable for the
common forms of colour blindness.
"""
import cv2
import numpy as np

CONTOUR = (61, 220, 151)
TP, FP, FN = (27, 175, 122), (235, 104, 52), (42, 120, 214)


def _hex(rgb):
    return "#%02x%02x%02x" % rgb


LEGEND_COLOURS = {"contour": _hex(CONTOUR), "tp": _hex(TP), "fp": _hex(FP), "fn": _hex(FN)}


def line_width(img):
    return max(2, round(min(img.shape[:2]) / 260))


def _muted(rgb, keep=0.45):
    """Desaturated, slightly darkened copy so a colour map on top stays readable."""
    gray = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    return (keep * gray).astype(np.float32)


def _colormap(values, cmap):
    """values in [0, 1] -> RGB float32."""
    v = (np.clip(values, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(v, cmap), cv2.COLOR_BGR2RGB).astype(np.float32)


def _contours(img, mask, colour, width):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, colour, width, lineType=cv2.LINE_AA)
    return img


def _label(img, text, x, y, width):
    scale = width * 0.32
    thick = max(1, width // 2)
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    pad = width * 2
    x0, y0 = max(0, x), max(0, y - th - 2 * pad)
    cv2.rectangle(img, (x0, y0), (x0 + tw + 2 * pad, y0 + th + 2 * pad), CONTOUR, -1)
    cv2.putText(img, text, (x0 + pad, y0 + th + pad), cv2.FONT_HERSHEY_SIMPLEX, scale, (16, 24, 20), thick, cv2.LINE_AA)


def overlay(rgb, pred):
    """Image with a translucent fill and outline on each predicted region, numbered when several."""
    out = rgb.astype(np.float32)
    m = pred.mask.astype(bool)
    out[m] = 0.72 * out[m] + 0.28 * np.array(CONTOUR, np.float32)
    out = out.clip(0, 255).astype(np.uint8)
    w = line_width(rgb)
    _contours(out, pred.mask, CONTOUR, w)
    if len(pred.regions) > 1:
        for i, r in enumerate(pred.regions, 1):
            _label(out, str(i), r.bbox[0], r.bbox[1], w)
    return out


def probability(rgb, pred):
    """Probability as a viridis map over the muted image, with the thresholded outline in white."""
    alpha = np.clip(pred.prob * 1.4, 0.15, 0.85)[..., None]
    out = (1 - alpha) * _muted(rgb) + alpha * _colormap(pred.prob, cv2.COLORMAP_VIRIDIS)
    out = out.clip(0, 255).astype(np.uint8)
    return _contours(out, pred.mask, (255, 255, 255), max(1, line_width(rgb) // 2))


SPREAD_MAX = 0.35  # std of probabilities shown at full colour


def disagreement(rgb, pred):
    """Standard deviation across the averaged predictions (models and/or flips)."""
    s = np.clip(pred.spread / SPREAD_MAX, 0, 1)
    alpha = np.clip(s * 1.5, 0.0, 0.9)[..., None]
    out = (1 - alpha) * _muted(rgb, 0.6) + alpha * _colormap(s, cv2.COLORMAP_MAGMA)
    out = out.clip(0, 255).astype(np.uint8)
    return _contours(out, pred.mask, (255, 255, 255), max(1, line_width(rgb) // 2))


def errors(rgb, pred, truth):
    """Pixels the model got right (green), predicted wrongly (orange) and missed (blue)."""
    p, t = pred.mask.astype(bool), truth.astype(bool)
    out = _muted(rgb, 0.55)
    for sel, colour in ((p & t, TP), (p & ~t, FP), (~p & t, FN)):
        out[sel] = 0.25 * out[sel] + 0.75 * np.array(colour, np.float32)
    out = out.clip(0, 255).astype(np.uint8)
    return _contours(out, truth, (255, 255, 255), max(1, line_width(rgb) // 2))
