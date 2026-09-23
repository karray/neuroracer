import {SceneManager} from 'gzweb/src/SceneManager.ts';
const status = document.querySelector('#status');
const toggle = document.querySelector('#connect');
const controls = ['car', 'overview', 'play', 'pause'].map(id => document.getElementById(id));
let manager, subscription, retry, desired = true, generation = 0;
function enabled(value) { controls.forEach(button => button.disabled = !value); }
function disconnect() {
  generation++;
  clearTimeout(retry);
  subscription?.unsubscribe();
  subscription = undefined;
  manager?.destroy();
  manager = undefined;
  document.querySelector('#gz-scene').replaceChildren();
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
    } else {
      status.textContent = 'Waiting for Gazebo…';
      clearTimeout(retry);
      retry = setTimeout(connect, 5000);
    }
  });
}
function focusCar() {
  const car = manager?.getModels().find(model => model.name === 'racecar');
  if (car) manager.moveTo(car.gz3dName || car.name);
}
document.querySelector('#car').onclick = focusCar;
document.querySelector('#overview').onclick = () => manager?.resetView();
document.querySelector('#play').onclick = () => manager?.play();
document.querySelector('#pause').onclick = () => manager?.pause();
toggle.onclick = () => {
  desired = !desired;
  toggle.textContent = desired ? 'Disconnect' : 'Connect';
  if (desired) connect();
  else { disconnect(); status.textContent = 'Disconnected'; }
};
window.addEventListener('resize', () => manager?.resize());
document.addEventListener('visibilitychange', () => {
  if (document.hidden) { disconnect(); status.textContent = 'Preview suspended'; }
  else if (desired) connect();
});
window.addEventListener('pagehide', disconnect);
connect();
