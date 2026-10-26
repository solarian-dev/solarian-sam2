# solarian-sam2

Click on a camera image in the controller and get the clicked object's
bounding box as an observation field, tracked frame to frame by
[SAM 2](https://github.com/facebookresearch/sam2) (the real-time camera
predictor from `segment-anything-2-real-time`).

It is a Solarian *service*: a server that holds the model on a GPU host, and a
processor (`Sam2BboxProcessor`) that a `CalculatedField` declares, which calls
it. The processor runs wherever the field's `execution_place` says: on the
relay server, or on the robot in the camera's module process.

It is also a dataset op, `track`, that runs the same model over recorded
sessions to write training targets; see [Track recorded episodes](#track-recorded-episodes).

## Install

On every machine of the deployment (the processor is imported by whichever
host computes the field):

```bash
solarian install solarian-sam2
```

On the GPU host, also the server side and SAM 2 itself:

```bash
pip install -e 'plugins/solarian-sam2[server]'
pip install -e <path>/segment-anything-2-real-time
```

## Serve

```bash
solarian run-service sam2-tracker --model small --checkpoint-dir <dir>/checkpoints
```

The first start downloads the chosen `sam2.1_hiera_*.pt` into `--checkpoint-dir`
when that file is not already there.

Options: `--model tiny|small|base_plus|large` (default small), `--res N`
(internal resolution, a multiple of 16; default 512), `--compiled`
(torch.compile; the first call autotunes), `--port` (default 8765).

## Declare the field

```python
from solarian.core.data import BoundingBoxData
from solarian.core.observation import CalculatedField, ModuleObs, SensorField
from solarian_sam2 import Sam2BboxProcessor

class WristCameraObs(ModuleObs):
    image = SensorField(MyRobot.wrist_camera.camera.rgb_image)
    bbox = CalculatedField(
        inputs=[MyRobot.wrist_camera.camera.rgb_image],
        processor=Sam2BboxProcessor,
        output=BoundingBoxData(),
        conf=os.path.join(CONF_PATH, "bbox_tracker.yaml"),
    )
```

For the object's segmentation instead, declare `processor=Sam2MaskProcessor`
with `output=MaskData()`: a `(H, W)` uint8 0/1 mask at the camera's resolution,
None until tracking starts. A mask-target UMI policy reads it.

`conf/services/bbox_tracker.yaml` (seeded from `templates/bbox_tracker.yaml`) sets
`execution_place` (`server` or `robot`) and `server_url`, both from required
deployment keys (`bbox_tracker_execution_place`, `bbox_tracker_url`): a
deployment without them fails at startup rather than picking a host for you.
Moving the tracker from the relay to the robot is then a change of deployment
rather than of code. A skill that reads the box should take its own placement
from the same key.

## Start and stop tracking

Clicking an image in the controller GUI emits an `EventType.CLICK` event with
the point in image pixels. A controller turns it into a service start:

```python
@event_handler(EventType.CLICK)
def track_clicked_object(self, event):
    self.start_service(event.data["module"], "bbox", point=event.data["point"])
```

`stop_service("wrist_camera", "bbox")` releases the track; the field reads all
zeros until the next start.

The field lags the image by one tracker round trip: the processor never blocks
the observation stream, it publishes the box of the newest frame it finished.
The StatusBoard shows `SAM2 tracker call (ms)` and `SAM2 tracker age (ms)`.

## Track recorded episodes

The `track` op writes `episodes/<n>/targets/<camera>.npz` into a new version of
a session: a `bbox` per row (x0, y0, x1, y1 in video pixels, -1 where the
object is lost) and, unless `mask=false`, its bit-packed `mask`. That is what
`solarian-umi-skill` trains a target-conditioned policy on.

An episode is tracked when it has a prompt, a point on the object or a box
around it on one frame (`frame: -1` is the last). `prompt` applies to every
episode, `prompts` to single ones by index and wins over it. Both are
optional: an episode without one keeps the targets it had. Tracking runs
forward from the prompt frame and then backward from it, so the prompt can sit
wherever the object is clearest; for a pick, that is the last frame, with the
object in the gripper.

Track in a computed dataset, a tree of versions over the session, so the
session stays as recorded. Episode by episode, with the model loaded once:

```bash
solarian dataset new data/cups-pick data/cups-pick-curated
solarian dataset apply data/cups-pick-curated track --online --param camera=camera0.rgb
solarian dataset redo data/cups-pick-curated@01-track 0 --param 'prompt={frame: -1, point: [730, 711]}'
solarian dataset accept data/cups-pick-curated@01-track 0
# ... each episode in turn, then:
solarian dataset close data/cups-pick-curated@01-track
```

`solarian dataset session data/cups-pick-curated@01-track` serves the same
redo and accept requests as JSON lines, keeping SAM 2 warm between episodes.
Or track every episode in one batch from a shared prompt, review, and redo only
the episodes it got wrong (the others are not stored again):

```bash
solarian dataset apply data/cups-pick-curated@source track --review --param 'prompt={frame: -1, point: [820, 576]}'
solarian dataset redo data/cups-pick-curated@02-track 1 --param 'prompt={frame: 12, point: [810, 590]}'
```

`solarian dataset run track <session>` still tracks a standalone session
straight into a new version beside it.

Solarian Base runs the same op from a computed dataset's page. Episode by
episode, each episode opens on its last frame (or its first, going
forward); a click on the object, or a drag around it, tracks it right away.
The mask is drawn over each frame as it is tracked, and the player follows
it. If it follows the wrong object, or only part of it, a click on that frame
restarts tracking from there, and Esc stops it. When tracking ends, the
episode plays back at 4x with its mask. Enter accepts it and opens the next
one, R redoes it.

`--param model=tiny|small|base_plus|large` picks the size (default tiny) and
`res` the internal resolution (default 1024). Checkpoints live in
`$SOLARIAN_MODELS_DIR/sam2` (default `~/.cache/solarian/models/sam2`), or in
`SAM2_CHECKPOINT_DIR` when it is set, and are downloaded there when missing;
Solarian Base points `SOLARIAN_MODELS_DIR` at its own store. It runs on CUDA,
then MPS, then CPU, whichever is there.

To track on a GPU host instead, serve it there and point the op at it, with
`--param server=http://turing.local:8765` or `SAM2_TRACKER_URL` in the
environment (Solarian Base's interactive worker passes its own on):

```bash
# on turing.local
solarian run-service sam2-tracker --model tiny --res 1024
# here
SAM2_TRACKER_URL=http://turing.local:8765 solarian dataset session data/cups-pick-curated@01-track
```

The server must run the op's `model` and `res`, which the version records;
the op refuses one that does not. The op sends each episode's mp4 (a few MB)
and the server decodes it and streams the masks back, so it runs at the
model's speed even over Wi-Fi. Tiny at 1024, frames per second:

| GPU | uncompiled | `--compiled` |
| --- | --- | --- |
| T4 | 7 | |
| L4 | 23 | 29 |
| A10G | 25 | 31 |
| RTX 3090 | 32 | |
| A100 | 30 | |
| L40S | 38 | 72 |
| H100 | 48 | 92 |

`--compiled` spends 4 to 7 minutes compiling before the server reports
healthy. Served on Modal (`solarian modal-serve`, or Solarian Base), the
image build does that instead: it runs the server's `--warm-up` on the
endpoint's GPU type and keeps torch.compile's caches in the image, so a
container, cold start included, is ready in about 20 seconds. The image is
built again, compile included, whenever the shipped code, the server's args
or the GPU type change. The server tracks one episode at a time; a second
one, or a `/init` meanwhile, is turned away rather than queued.

SAM 2 squashes every frame to `res`×`res`, so the black around a fisheye's
circle costs resolution. `--param crop=auto` (the default) finds what is lit
in the session's first episode and gives SAM 2 only that; `'crop=[134, 0, 1310, 1080]'`
sets the box in pixels and `crop=none` turns it off. Prompts and targets stay
in frame pixels either way.

A prompt that misses the object tends to segment the table or the whole view,
which still counts as found on every frame; `box_area` in the episode's
`targets` coverage (the median share of the frame the box covers) gives it away.

## Wire format

`POST /init {image_jpeg, points, labels}` (or `bbox` instead of the points)
and `POST /track {image_jpeg, session?}` reply with `{"bbox": [x0, y0, x1, y1]
| null, "mask_png": str | null, "inference_ms"}`, and `/init` with the
`session` it started as well. A `/track` naming an older session is a 409.
`POST /reset {}` replies with `{"status"}`. Frames are JPEG bytes in the
`solarian.utils.wire` array encoding, decoded as RGB for SAM 2. `mask_png` is
the 0/1 mask at the size of the frame sent, base64 PNG.

`POST /track_episode?n=&prompt=&crop=` takes the episode's mp4 as the body,
with `prompt` the op's `Prompt` and `crop` an `[x0, y0, x1, y1]` box or `null`,
both JSON. It answers with JSON lines `{"frame", "mask_png"}`, in tracking
order: the prompt's frame, forward to the end, then backward to the first.
Masks are at the video's size; `mask_png` is null where the object is lost.
A failure mid-episode ends the stream with an `{"error"}` line.

`PUT /videos/<sha256>` keeps an mp4 on the server (on its local disk, the
least recently used dropped past 4 GB), and `/track_episode?...&video=<sha256>`
with no body tracks it; a video the server lost is a 404. One connection to
Modal carries ~0.6 MB/s, so sending an episode's ~4.5 MB with its track took
longer than an L40S tracks it. The op instead uploads the next 8 episodes
while one is tracked, 4 at a time (~2.2 MB/s together), and names them.
