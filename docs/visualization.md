# Visualization

| Command | What runs | URL |
| --- | --- | --- |
| `./scripts/dev web` | Headless simulator, ROS bridge, GzWeb static server | http://localhost:8090/ |
| `./scripts/dev sim` | Headless simulator and ROS bridge only | — |
| `./scripts/dev vnc` | Gazebo desktop GUI attached to a running `web` or `sim` | http://localhost:8091/vnc.html?autoconnect=true&resize=scale |

Stop `web`/`sim` with Ctrl-C before switching modes. Both ports bind to host
loopback only.

GzWeb only shows the simulation and what the training processes publish
(**Telemetry**): the car's and the learner's progress, and the frame the agent
sees. The world stays paused except while `./scripts/dev train` or `drive`
steps it. A section disappears 5 s after its process stops publishing.

## Telemetry format

Each topic (`/telemetry/car`, `/telemetry/learner`) carries JSON messages as
`std_msgs/String`, which the ROS–Gazebo bridge forwards to GzWeb. A message
lists its fields in display order:

```json
{"version": 1, "fields": [
  {"name": "step", "type": "text", "value": 1234},
  {"name": "reward", "type": "graph", "value": 0.8},
  {"name": "camera", "type": "image", "value": "data:image/png;base64,iVBOR…"}
]}
```

| `type` | `value` | Shown as |
| --- | --- | --- |
| `text` | string, number, boolean or `null` | the value |
| `graph` | number | the value and a graph of its last 300 values |
| `image` | `data:image/png;base64,…` or `data:image/jpeg;base64,…` | a tile on the right |

A section shows exactly the fields of its latest message. A message with
another `version`, or a field with other keys, an unknown `type`, a value of the
wrong kind or a repeated `name`, is shown as an error. As JSON Schema:

```json
{
  "type": "object",
  "required": ["version", "fields"],
  "additionalProperties": false,
  "properties": {
    "version": {"const": 1},
    "fields": {"type": "array", "items": {
      "type": "object",
      "required": ["name", "type", "value"],
      "additionalProperties": false,
      "properties": {"name": {"type": "string"}, "type": {"enum": ["text", "graph", "image"]}, "value": true},
      "oneOf": [
        {"properties": {"type": {"const": "text"}, "value": {"type": ["string", "number", "boolean", "null"]}}},
        {"properties": {"type": {"const": "graph"}, "value": {"type": "number"}}},
        {"properties": {"type": {"const": "image"},
                        "value": {"type": "string", "pattern": "^data:image/(png|jpeg);base64,"}}}
      ]
    }}
  }
}
```

## Maintenance

`docker/gzweb/` holds the frontend. `scripts/dev web` rebuilds it with Docker
cache; during a running session use
`docker compose --profile web up -d --build --no-deps web` and reload the page.

GzWeb 3.0.2 is pinned and patched by `patch-gzweb.mjs`, which fails the build if
the upstream source changes.

The image builds Jetty 10.5.0's WebSocket plugin from pinned upstream source
with [a disconnect fix](../docker/websocket/disconnect.patch). Drop it once
[gz-sim#4016](https://github.com/gazebosim/gz-sim/pull/4016) is released.
