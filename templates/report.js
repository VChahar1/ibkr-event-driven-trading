const FIGS = JSON.parse(document.getElementById('figs').textContent);
const css = n => getComputedStyle(document.documentElement).getPropertyValue('--' + n).trim();
function resolve(v) {
  if (Array.isArray(v)) return v.map(resolve);
  if (v && typeof v === 'object') { const o = {}; for (const k in v) o[k] = resolve(v[k]); return o; }
  if (v === '@diverging') return [[0, css('neg')], [0.5, css('mid')], [1, css('pos')]];
  if (v === '@sequential') return [[0, css('seq-lo')], [0.5, css('seq-mid')], [1, css('seq-hi')]];
  if (typeof v === 'string' && v[0] === '@') return css(v.slice(1));
  return v;
}
function merge(a, b) {
  const o = {...a};
  for (const k in b) o[k] = (b[k] && typeof b[k] === 'object' && !Array.isArray(b[k]) && a[k]) ? merge(a[k], b[k]) : b[k];
  return o;
}
function base() {
  const axis = {gridcolor: css('grid'), linecolor: css('baseline'), zerolinecolor: css('baseline'),
    tickfont: {color: css('muted')}, showspikes: true, spikemode: 'across', spikethickness: 1,
    spikecolor: css('muted'), spikedash: 'solid', automargin: true};
  return {paper_bgcolor: css('surface-1'), plot_bgcolor: css('surface-1'),
    font: {family: 'system-ui, -apple-system, "Segoe UI", sans-serif', color: css('text-secondary'), size: 12},
    margin: {l: 48, r: 16, t: 36, b: 32}                                   , xaxis: axis, yaxis: {...axis, showspikes: false},
    legend: {orientation: 'h', x: 0, y: 1.02, yanchor: 'bottom', font: {color: css('text-secondary')}},
    hoverlabel: {bgcolor: css('surface-1'), bordercolor: css('border'), font: {color: css('text-primary')}}};
}
function render() {
  for (const f of FIGS) {
    Plotly.react(f.id, resolve(f.data), merge(base(), resolve(f.layout)),
                 {displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d']});
  }
}
render();
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', render);
