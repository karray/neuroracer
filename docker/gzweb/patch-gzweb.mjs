// Small, checked compatibility fixes for the pinned upstream GzWeb 3.0.2.
// Compile its source so Three.js is shared and these fixes remain auditable.
import {readFileSync, writeFileSync} from 'node:fs';

function patch(file, replacements) {
  let source = readFileSync(file, 'utf8');
  for (const [before, after] of replacements) {
    if (!source.includes(before)) throw new Error(`${file} changed; review patch: ${before}`);
    source = source.replaceAll(before, after);
  }
  writeFileSync(file, source);
}

patch('node_modules/gzweb/src/SceneManager.ts', [
  ['this.scene.scene.renderer', 'this.scene.renderer'],
  ['"ignition.msgs.WorldControl"', '"gz.msgs.WorldControl"'],
  ['"ignition.msgs.ServerControl"', '"gz.msgs.ServerControl"'],
  ['public disconnect(): void {', `public disconnect(): void {
    cancelAnimationFrame(this.cancelAnimation);
    this.previousRenderTimestampMs = 0;
    this.models = [];`],
  ['      if (this.scene.getParticleSystem()) {', `      // Limit preview rendering independently of physics / sensor rates.
      if (document.hidden || timestampMs - this.previousRenderTimestampMs < 1000 / 30) return;
      if (this.scene.getParticleSystem()) {`],
  ['      shaders: new Shaders(),', `      shaders: new Shaders(),
      defaultCameraPosition: new THREE.Vector3(2, 0.5, 2),
      defaultCameraLookAt: new THREE.Vector3(2, 3.7, 0.15),`],
  // Jetty 10.5's text framing drops empty payloads and truncates NUL bytes.
  // A nonempty header produces a valid default pause:false request without NUL.
  ['{ pause: false },', '{ pause: false, header: {data: [{key: "gzweb"}]} },'],
]);
patch('node_modules/gzweb/src/Scene.ts', [
  ['import * as JSZip from "jszip";', 'import JSZip from "jszip";'],
  ['Math.max(bboxSize.x, bboxSize.y, bboxSize.z)', 'Math.max(bboxSize.x, bboxSize.y, bboxSize.z, 1.5)'],
  ['    this.renderer.renderLists.dispose();', '    this.controls.dispose();\n    this.renderer.forceContextLoss();\n    this.renderer.renderLists.dispose();'],
  // Third-person follow defaults are sized for a full-size vehicle; frame the 0.5 m racecar.
  ['new THREE.Vector3(\n    -6,\n    -2,\n    1.5,\n  )', 'new THREE.Vector3(-1.2, 0, 0.6)'],
  ['new THREE.Vector3(12, -4, 0)', 'new THREE.Vector3(3, 0, 0)'],
  // Upstream conjugates the followed model's own quaternion after a drag, mirroring it.
  ['this.cameraTrackObject.quaternion.conjugate()', 'this.cameraTrackObject.quaternion.clone().conjugate()'],
]);
patch('node_modules/gzweb/src/Transport.ts', [
  // Jetty names camera messages gz.msgs.Image; only then are they streamed as PNG.
  ['"ignition.msgs.Image"', '"gz.msgs.Image"'],
]);
patch('node_modules/three-nebula/build/esm/utils/uid.js', [
  ["import uid from 'uuid/v1';", "import {v1 as uid} from 'uuid';"],
]);
