# Source and local fix

`src/WebsocketServer.cc` and `.hh` are from Gazebo Sim's Apache-2.0 licensed
`gz-sim10_10.5.0` release:
https://github.com/gazebosim/gz-sim/tree/gz-sim10_10.5.0/src/systems/websocket_server
Original copyright and license headers are retained.

The only source change is in `OnDisconnect`: subtract discarded queued messages
from `messageCount` while holding the connection and run mutexes. Upstream erased
the queue without adjusting the counter. The run-loop wait predicate then stayed
true indefinitely, consuming a CPU core after a browser disconnected mid-stream.

The library has a distinct filename and is loaded explicitly by our web launch.
The installed Gazebo plugin is not overwritten. Headless `sim` doesn't load
this plugin. Remove this package when the supported upstream release includes
the fix. Recheck ABI/source compatibility when upgrading Gazebo.
