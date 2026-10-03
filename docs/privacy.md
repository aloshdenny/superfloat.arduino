# Privacy and scope

The sentinel watches landscapes for smoke and heat. It does not watch people.
This is a design property, not only a policy.

- **The model has no person class.** SmokeNet outputs clear, smoke or flame for
  each 128 px tile of a downscaled frame. EmberNet outputs ambient or hotspot
  for a 32x24 thermal image, a resolution at which individuals are not
  identifiable. Nothing in the pipeline detects, counts, tracks or identifies
  people, vehicles or faces.
- **Frames are never stored or sent.** A frame lives in memory for one loop
  iteration: it is tiled, scored and dropped. An alert carries the node id, the
  tile index, a score, a timestamp and the node's configured coordinates.
- **The status page shows tile states, not video.** It is served on the
  board's local network only, from files on the board.
- **No cloud dependency.** Inference, alerting logic and local alarms all run
  on the board. The only outbound traffic is the alert webhook the operator
  configures.

## Siting guidance

Point cameras at ridgelines, slopes and horizon, not at homes, roads or trails.
Where a field of view unavoidably includes private property, mask those tiles in
the configuration. Tile masking is planned for stage 2 (docs/roadmap.md).
Deployments on land the operator does not own need the landowner's permission,
and should be signposted where people pass near the node.

## Contest scope

The challenge rules exclude surveillance or people tracking without consent and
personal or biometric data collection. This project collects neither. It is a
civilian hazard sensor, and its outputs describe the state of vegetation and
terrain, not the behaviour of anyone in it.
