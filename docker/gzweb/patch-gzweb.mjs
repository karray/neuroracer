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
  ['      shaders: new Shaders(),', `      shaders: new Shaders(),
      defaultCameraPosition: new THREE.Vector3(2, 0.5, 2),
      defaultCameraLookAt: new THREE.Vector3(2, 3.7, 0.15),`],
]);
patch('node_modules/gzweb/src/Scene.ts', [
  ['import * as JSZip from "jszip";', 'import JSZip from "jszip";'],
  ['    this.renderer.renderLists.dispose();', '    this.controls.dispose();\n    this.renderer.forceContextLoss();\n    this.renderer.renderLists.dispose();'],
  // GzWeb's STLLoader.parse already returns a Mesh; see the STLLoader patch below.
  ['function (geometry: THREE.BufferGeometry) {\n        mesh = new THREE.Mesh(geometry);', 'function (loaded: THREE.Mesh) {\n        mesh = loaded;'],
]);
// Its parse expects a Uint8Array (as the WebSocket fallback delivers), so every HTTP
// STL load threw and silently re-fetched the mesh through Gazebo.
patch('node_modules/gzweb/include/STLLoader.js', [
  ['onLoad(scope.parse(text));', 'onLoad(scope.parse(new Uint8Array(text)));'],
]);
patch('node_modules/three-nebula/build/esm/utils/uid.js', [
  ["import uid from 'uuid/v1';", "import {v1 as uid} from 'uuid';"],
]);
