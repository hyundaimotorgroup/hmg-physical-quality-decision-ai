# Video pipeline

`pqdai video --video clip.mp4 --out runs/clip [--domain weld] [--sample-fps 5] [--max-seconds 30]`

Every sampled frame is judged by the unchanged photo pipeline. Cross-frame identity then collapses repeated sightings of one physical object into one verdict:

* `camera_motion` - global ORB+RANSAC ego-motion plus per-track local correction (objects are static in the world).
* `video_tracker` - per-track Kalman filter, Hungarian assignment on IoU first; appearance (embedding) similarity only as a guarded fallback that rejects candidates plausible for several tracks (distractor-aware).
* `bead_clustering` - re-joins tracks split by identity hand-offs when they never co-existed.
* `track_aggregation` - safety-first: one unsafe observation makes the track (and the video) unsafe.
* Outputs: `results.json`, `report.html`, `overlay.mp4`, `dedup_audit.html` (frame-by-frame audit of duplicate counting), `match_log.json` (why two detections were judged to be the same object).

Video mode currently requires `['BAD', 'GOOD']` class names.
