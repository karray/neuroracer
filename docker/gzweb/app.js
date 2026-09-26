import {SceneManager} from 'gzweb/src/SceneManager.ts';
import {Topic} from 'gzweb/src/Topic.ts';
const CAMERA = '/camera/image_raw';
// JSON that the training processes publish; each topic gets its own section.
const TELEMETRY = ['/telemetry/car', '/telemetry/learner'];
const status = document.querySelector('#status');
const toggle = document.querySelector('#connect');
const controls = ['car', 'overview'].map(id => document.getElementById(id));
const pip = document.querySelector('#pip');
const frame = pip.querySelector('img');
const caption = pip.querySelector('figcaption');
const panel = document.querySelector('#telemetry-panel');
let manager, subscription, retry, desired = true, camera = true, telemetry = false, generation = 0;
function enabled(value) { controls.forEach(button => button.disabled = !value); }
function showCamera() {
  pip.hidden = false;
  // The robot's own sensor, streamed as PNG while the world runs; paused worlds send no frames.
  manager.subscribeToTopic(new Topic(CAMERA, png => {
    URL.revokeObjectURL(frame.src);
    frame.src = URL.createObjectURL(new Blob([png], {type: 'image/png'}));
    frame.hidden = false;
    caption.hidden = true;
  }));
}
function hideCamera() {
  pip.hidden = frame.hidden = true;
  caption.hidden = false;
  URL.revokeObjectURL(frame.src);
  frame.removeAttribute('src');
}
function showTelemetry() {
  panel.hidden = false;
  for (const topic of TELEMETRY) {
    manager.subscribeToTopic(new Topic(topic, message => render(topic, JSON.parse(message.data))));
  }
}
// Fields that move within bounds, rather than count up, start as graphs; clicking a field switches it.
const HISTORY = 300;
let graphs = new Set(['/telemetry/car reward', '/telemetry/car action', '/telemetry/learner loss',
                      '/telemetry/learner q', '/telemetry/learner updates_per_s']);
try {
  const saved = JSON.parse(localStorage.getItem('telemetry-graphs'));
  if (Array.isArray(saved)) graphs = new Set(saved);
} catch {}
// A section disappears when its topic falls silent, e.g. the learner's while an agent only drives.
const STALE_MS = 5000;
const sections = new Map();
function render(topic, values) {
  let section = sections.get(topic);
  if (!section) {
    const heading = Object.assign(document.createElement('h2'), {textContent: topic.split('/').pop()});
    section = {list: document.createElement('dl'), rows: new Map()};
    section.nodes = [heading, section.list];
    panel.append(...section.nodes);
    sections.set(topic, section);
  }
  clearTimeout(section.timer);
  section.timer = setTimeout(() => {
    section.nodes.forEach(node => node.remove());
    sections.delete(topic);
  }, STALE_MS);
  for (const [key, value] of Object.entries(values)) {
    let row = section.rows.get(key);
    if (!row) {
      row = {term: Object.assign(document.createElement('dt'), {textContent: key}),
             value: document.createElement('dd'), canvas: document.createElement('canvas'), history: []};
      row.term.onclick = row.value.onclick = row.canvas.onclick = () => {
        if (!row.history.length) return;
        const name = `${topic} ${key}`;
        if (!graphs.delete(name)) graphs.add(name);
        try { localStorage.setItem('telemetry-graphs', JSON.stringify([...graphs])); } catch {}
        show(name, row);
      };
      section.list.append(row.term, row.value, row.canvas);
      section.rows.set(key, row);
    }
    row.value.textContent = typeof value === 'number' && !Number.isInteger(value) ? value.toFixed(3) : JSON.stringify(value);
    const number = Array.isArray(value) && value.length === 1 ? value[0] : value;  // DDPG's action
    if (typeof number === 'number') {
      row.history.push(number);
      if (row.history.length > HISTORY) row.history.shift();
    }
    show(`${topic} ${key}`, row);
  }
  // The panel grows to fit its longest value and keeps that width, instead of jittering.
  panel.style.minWidth = `${Math.max(parseFloat(panel.style.minWidth) || 0, panel.offsetWidth)}px`;
}
function show(name, row) {
  row.canvas.hidden = !(graphs.has(name) && row.history.length);
  row.term.classList.toggle('graphable', row.history.length > 0);
  if (!row.canvas.hidden) draw(row.canvas, row.history);
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
  panel.hidden = true;
  panel.replaceChildren();
  panel.style.minWidth = '';
  sections.forEach(section => clearTimeout(section.timer));
  sections.clear();
}
function disconnect() {
  generation++;
  clearTimeout(retry);
  subscription?.unsubscribe();
  subscription = undefined;
  manager?.destroy();
  manager = undefined;
  document.querySelector('#gz-scene').replaceChildren();
  hideCamera();
  hideTelemetry();
  enabled(false);
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
    enabled(ready);
    if (ready) {
      clearTimeout(retry);
      status.textContent = 'Connected';
      // Scene models are populated after the connection-ready notification.
      requestAnimationFrame(() => {
        if (current === generation) focusCar();
      });
      if (camera) showCamera();
      if (telemetry) showTelemetry();
    } else {
      status.textContent = 'Waiting for Gazebo…';
      clearTimeout(retry);
      retry = setTimeout(connect, 5000);
    }
  });
}
function focusCar() {
  const car = manager?.getModels().find(model => model.name === 'racecar');
  if (car) manager.thirdPersonFollow(car.gz3dName || car.name);
}
document.querySelector('#car').onclick = focusCar;
document.querySelector('#overview').onclick = () => {
  manager?.thirdPersonFollow(null);
  manager?.resetView();
};
document.querySelector('#camera').onclick = event => {
  camera = !camera;
  event.target.setAttribute('aria-pressed', camera);
  if (!manager || controls[0].disabled) return;
  // Unsubscribing stops the server's PNG encoding for this client.
  if (camera) showCamera();
  else { manager.unsubscribeFromTopic(CAMERA); hideCamera(); }
};
document.querySelector('#telemetry').onclick = event => {
  telemetry = !telemetry;
  event.target.setAttribute('aria-pressed', telemetry);
  if (!manager || controls[0].disabled) return;
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
