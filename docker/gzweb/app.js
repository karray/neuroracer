import {SceneManager} from 'gzweb/src/SceneManager.ts';
import {Topic} from 'gzweb/src/Topic.ts';
// JSON that the training processes publish, in the format of docs/visualization.md; each topic gets its own section.
const TELEMETRY = ['/telemetry/car', '/telemetry/learner'];
const VERSION = 1;
const VALID = {
  text: value => value === null || ['string', 'number', 'boolean'].includes(typeof value),
  graph: value => typeof value === 'number',
  image: value => typeof value === 'string' && /^data:image\/(png|jpeg);base64,/.test(value),
};
const HISTORY = 300;
// A section disappears when its topic falls silent, e.g. the learner's while an agent only drives.
const STALE_MS = 5000;
const status = document.querySelector('#status');
const toggle = document.querySelector('#connect');
const panel = document.querySelector('#telemetry-panel');
const images = document.querySelector('#images');
const sections = new Map();
let manager, subscription, retry, connected = false, desired = true, telemetry = true, generation = 0;
function showTelemetry() {
  panel.hidden = images.hidden = false;
  for (const topic of TELEMETRY) {
    manager.subscribeToTopic(new Topic(topic, message => render(topic, message.data)));
  }
}
function element(tag, text, className) {
  return Object.assign(document.createElement(tag), {textContent: text ?? '', className: className ?? ''});
}
function parse(data) {
  const message = JSON.parse(data);
  if (message?.version !== VERSION) throw new Error(`unsupported telemetry version ${JSON.stringify(message?.version)}`);
  if (Object.keys(message).length !== 2 || !Array.isArray(message.fields)) throw new Error('message must have exactly version and fields');
  return message.fields;
}
function invalid(field, names) {
  if (field === null || typeof field !== 'object' || Object.keys(field).sort().join() !== 'name,type,value') {
    return 'field must have exactly name, type and value';
  }
  if (typeof field.name !== 'string') return 'name must be a string';
  if (names.has(field.name)) return 'duplicate name';
  if (!Object.hasOwn(VALID, field.type)) return `unknown type ${JSON.stringify(field.type)}`;
  if (!VALID[field.type](field.value)) return `invalid ${field.type} value`;
}
function createRow(section, name, type) {
  if (type === 'image') {
    const figure = element('figure');
    const image = figure.appendChild(Object.assign(document.createElement('img'), {alt: `${section.name} ${name}`}));
    figure.append(element('figcaption', `${section.name} · ${name}`));
    return {type, nodes: [figure], image};
  }
  const row = {type, value: element('dd'), history: []};
  row.nodes = [element('dt', name), row.value];
  if (type === 'graph') row.nodes.push(row.canvas = element('canvas'));
  return row;
}
function format(value) {
  if (value === null) return '—';
  return typeof value === 'number' && !Number.isInteger(value) ? value.toFixed(3) : String(value);
}
// The section shows exactly the fields of its topic's latest message, in their order.
function render(topic, data) {
  let section = sections.get(topic);
  if (!section) {
    const name = topic.split('/').pop();
    section = {name, heading: element('h2', name), list: element('dl'), tiles: element('div'), rows: new Map()};
    panel.append(section.heading, section.list);
    images.append(section.tiles);
    sections.set(topic, section);
  }
  clearTimeout(section.timer);
  section.timer = setTimeout(() => removeSection(topic), STALE_MS);
  let fields;
  try {
    fields = parse(data);
  } catch (error) {
    section.rows.clear();
    section.list.replaceChildren(element('dt', 'message'), element('dd', error.message, 'error'));
    section.tiles.replaceChildren();
    return;
  }
  const rows = new Map(), items = [], tiles = [];
  for (const field of fields) {
    const error = invalid(field, rows);
    if (error) {
      items.push(element('dt', typeof field?.name === 'string' ? field.name : '?'), element('dd', error, 'error'));
      continue;
    }
    let row = section.rows.get(field.name);
    if (row?.type !== field.type) row = createRow(section, field.name, field.type);
    rows.set(field.name, row);
    if (row.type === 'image') {
      row.image.src = field.value;
      tiles.push(...row.nodes);
      continue;
    }
    row.value.textContent = format(field.value);
    if (row.type === 'graph') {
      row.history.push(field.value);
      if (row.history.length > HISTORY) row.history.shift();
    }
    items.push(...row.nodes);
  }
  section.rows = rows;
  section.list.replaceChildren(...items);
  section.tiles.replaceChildren(...tiles);
  rows.forEach(row => row.canvas && draw(row.canvas, row.history));
  // The panel grows to fit its longest value and keeps that width, instead of jittering.
  panel.style.minWidth = `${Math.max(parseFloat(panel.style.minWidth) || 0, panel.offsetWidth)}px`;
}
function removeSection(topic) {
  const section = sections.get(topic);
  clearTimeout(section.timer);
  [section.heading, section.list, section.tiles].forEach(node => node.remove());
  sections.delete(topic);
}
// Newest sample at the right edge, each value held until the next (actions are discrete).
function draw(canvas, values) {
  const ratio = devicePixelRatio, width = canvas.clientWidth, height = canvas.clientHeight;
  if (canvas.width !== Math.round(width * ratio)) canvas.width = Math.round(width * ratio);
  if (canvas.height !== Math.round(height * ratio)) canvas.height = Math.round(height * ratio);
  const context = canvas.getContext('2d');
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.shadowBlur = 0;
  context.clearRect(0, 0, width, height);
  const low = Math.min(...values), high = Math.max(...values), span = high - low || 1;
  const x = i => width - (values.length - 1 - i) * width / (HISTORY - 1);
  const y = value => height - 2 - (value - low) / span * (height - 4);
  context.beginPath();
  context.moveTo(x(0), y(values[0]));
  for (let i = 1; i < values.length; i++) {
    context.lineTo(x(i), y(values[i - 1]));
    context.lineTo(x(i), y(values[i]));
  }
  context.strokeStyle = '#8fd694';
  context.lineWidth = 1.5;
  context.stroke();
  context.lineTo(x(values.length - 1), height);
  context.lineTo(x(0), height);
  context.fillStyle = '#8fd69430';
  context.fill();
  const label = value => Number.isInteger(value) ? String(value) : value.toFixed(2);
  context.font = '10px ui-monospace,monospace';
  context.fillStyle = '#b1c3d9';
  context.shadowColor = '#000';
  context.shadowBlur = 2;
  context.fillText(label(high), 2, 10);
  context.fillText(label(low), 2, height - 2);
}
function hideTelemetry() {
  panel.hidden = images.hidden = true;
  [...sections.keys()].forEach(removeSection);
  panel.style.minWidth = '';
}
function disconnect() {
  generation++;
  clearTimeout(retry);
  subscription?.unsubscribe();
  subscription = undefined;
  manager?.destroy();
  manager = undefined;
  document.querySelector('#gz-scene').replaceChildren();
  hideTelemetry();
  connected = false;
}
function connect() {
  disconnect();
  if (!desired || document.hidden) return;
  const current = generation;
  status.textContent = 'Connecting…';
  manager = new SceneManager({elementId: 'gz-scene'});
  manager.connect(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  subscription = manager.getConnectionStatusAsObservable().subscribe(ready => {
    if (current !== generation) return;
    connected = ready;
    if (ready) {
      clearTimeout(retry);
      status.textContent = 'Connected';
      if (telemetry) showTelemetry();
    } else {
      status.textContent = 'Waiting for Gazebo…';
      clearTimeout(retry);
      retry = setTimeout(connect, 5000);
    }
  });
}
document.querySelector('#telemetry').onclick = event => {
  telemetry = !telemetry;
  event.target.setAttribute('aria-pressed', telemetry);
  if (!connected) return;
  if (telemetry) showTelemetry();
  else { TELEMETRY.forEach(topic => manager.unsubscribeFromTopic(topic)); hideTelemetry(); }
};
toggle.onclick = () => {
  desired = !desired;
  toggle.textContent = desired ? 'Disconnect' : 'Connect';
  if (desired) connect();
  else { disconnect(); status.textContent = 'Disconnected'; }
};
window.addEventListener('resize', () => {
  manager?.resize();
  panel.style.minWidth = '';
});
document.addEventListener('visibilitychange', () => {
  if (document.hidden) { disconnect(); status.textContent = 'Preview suspended'; }
  else if (desired) connect();
});
window.addEventListener('pagehide', disconnect);
connect();
