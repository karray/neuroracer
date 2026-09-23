import {SceneManager} from 'gzweb/src/SceneManager.ts';
import {Topic} from 'gzweb/src/Topic.ts';
const CAMERA = '/camera/image_raw';
const status = document.querySelector('#status');
const toggle = document.querySelector('#connect');
const controls = ['car', 'overview', 'play', 'pause'].map(id => document.getElementById(id));
const pip = document.querySelector('#pip');
const frame = pip.querySelector('img');
const caption = pip.querySelector('figcaption');
let manager, subscription, retry, desired = true, camera = true, generation = 0;
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
function disconnect() {
  generation++;
  clearTimeout(retry);
  subscription?.unsubscribe();
  subscription = undefined;
  manager?.destroy();
  manager = undefined;
  document.querySelector('#gz-scene').replaceChildren();
  hideCamera();
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
