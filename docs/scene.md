# The .fjs scene format

A scene is a small JSON file, `.fjs`, that points at `.fja` assets
(`docs/format.md`) and holds what belongs to the scene rather than to an
asset: where each object stands, the groups, the lights, the directional
lights, the sun, the sky and exposure, the time of day, the camera and
the render settings. `src/fjs.rs` reads and writes it. The editor writes
it on File > Save; `--save my.fjs` writes whatever is open.

Assets are heavy (a dense mesh with its levels of detail, texture tiles
and surface samples) and written once; a scene is a few kilobytes and is
what changes when things move. Several scenes can use the same asset.

## Example

    {
      "fjs": 1,
      "world": "built-in",
      "objects": [
        { "name": "dinosaurs", "pivot": [0.0, 0.0, 0.0] },
        { "name": "trex", "asset": "trex_highres.fja", "pivot": [0.0, 0.0, 0.0], "rotation": [0.0, 30.0, 0.0], "scale": 1.0, "parent": 0 },
        { "name": "car", "asset": "models/car.fja", "pivot": [7.5, 0.0, 5.0], "rotation": [0.0, -35.0, 0.0] }
      ],
      "lights": [
        { "position": [-14.0, 16.0, 18.0], "color": [1.148, 0.976, 0.804], "lumens": 1000000.0 }
      ],
      "directional": [
        { "on": true, "azimuth": 210.0, "elevation": 40.0, "color": [1.0, 1.0, 1.0], "lux": 2000.0 }
      ],
      "time": 18.2,
      "sun": { "direction": [-0.997, -0.055, -0.052], "color": [1.6, 0.88, 0.4], "brightness": 600.0 },
      "sky": { "on": true, "lux": 30.9, "tint": [0.93, 0.95, 1.08], "contrast": 0.5 },
      "backdrop": { "zenith": [0.5, 0.69, 1.33], "horizon": [1.32, 1.02, 0.78], "contrast": 0.5 },
      "exposure": 0.0128,
      "camera": { "target": [3.17, 2.68, 0.96], "distance": 20.4, "yaw": 0.0, "pitch": 4.0, "fov": 45.0 },
      "cameras": [ { "name": "camera1", "target": [0.0, 1.0, 0.0], "distance": 6.0, "yaw": 30.0, "pitch": 10.0, "fov": 35.0 } ],
      "view": "camera1",
      "render": { "shadows": true, "shadow_res": 512, "bounce": true, "backface_culling": true }
    }

The editor writes the same, indented, with every field filled in.

## Fields

Only `fjs` is required; everything else may be left out, and what is
left out keeps the program's own value.

| field | meaning |
|---|---|
| `fjs` | format version, 1 |
| `world` | the geometry outside every object: an `.fja` path, or left out for none; `"built-in"`, from files written before the ground and mountains were objects, gives both as objects |
| `objects` | objects and groups, in order; a group before what is in it |
| `lights` | point lights: `position` (metres, in the world), `color` (linear RGB), `lumens` (1 to 2,000,000), `size` (its radius, metres: how soft its shadows grow; 0.05 when left out), `falloff` (metres over which its shadows fade with their caster's distance; 0, they do not, when left out), `parent` (the object it is parented to, an index into `objects`; it keeps its place relative to it) |
| `directional` | up to four directional lights: `on`, `azimuth` and `elevation` (degrees; azimuth clockwise from -Z), `color`, `lux` (up to 150,000), `size` (degrees across; 1 when left out), `falloff` (as a light's) |
| `time` | time of day, hours; sets the sun, the sky and the exposure before the fields below, which override it |
| `sun` | `direction` (towards the sun), `color`, `brightness` (0 to 1000), `size` (degrees across; 0.6 when left out) |
| `sky` | `on`, `lux`, `tint`, `contrast` (lighting contrast, 0 to 1) |
| `backdrop` | the sky as seen: `zenith`, `horizon` colours, `contrast` (0 to 1) |
| `exposure` | display value per cd/m² before the tone curve |
| `camera` | the perspective camera, Maya's persp: `target`, `distance` (metres), `yaw`, `pitch`, `fov` (degrees) |
| `cameras` | the scene's cameras: each a `name` and the same fields as `camera` |
| `view` | the camera the view looks through, by name; persp when absent |
| `render` | `shadows`, `shadow_fields` (true: shadows through the objects' shadow fields; false: shadow maps), `shadow_res` (the maps'), `bounce`, `light_reach`, `occlusion_culling`, `adaptive`, `sky_occlusion`, `incremental_moves`, `backface_culling`, `wireframe`, `drag_preview`, `bake_over_frames`, `moving_rate`, `moving_detail` |

An object:

| field | meaning |
|---|---|
| `name` | shown in the outliner |
| `asset` | the `.fja` it is, or the program's own: `built-in:ground`, `built-in:mountains`, `built-in:sphere`, `built-in:cube`, `built-in:cylinder`, `built-in:plane`; left out for a group |
| `pivot` | where it stands: the point it turns and scales about, metres |
| `rotation` | degrees about X, Y and Z, applied in that order (Maya's xyz); 0 when left out |
| `scale` | along the object's own axes, before its rotation (Maya's order): one number when the same on every axis, else `[x, y, z]`; 1 when left out |
| `parent` | the group it is in, the index of an earlier entry in `objects` |
| `materials` | how its materials differ from its asset's in this scene: each an `index` (from 0, in the asset's order), its `name` (for reading), and whichever of `tint`, `roughness`, `metallic`, `specular` differ |
| `frozen` | what Freeze Transformations folded in: 12 numbers, the three axes then the translation of an affine map between the object's frame and its geometry; left out when nothing is frozen |
| `origin` | where its translate reads 0 (a frozen translate), metres; left out at the origin |
| `instance_of` | an instance of an earlier entry in `objects` (its index): that entry's geometry, materials and texture tiles, stored once; `asset` and `materials` are left out |

`pivot`, `rotation` and `scale` are absolute, from the rest the asset
stores (`docs/format.md`): an asset made by `fkj-import` rests at the
origin, so its `pivot` is where its origin lands.

## Where assets are found

An asset path is relative to the scene's folder, or absolute; `/`
between folders on every system. When it is not found there, its file
name is looked for in `$FKJENSEN_ASSETS`, `assets/models/` and
`assets/local/` (as for the built-in set), so a scene that refers to the
repository's models by an absolute path on one machine opens on another.

## What the machine decides

The micropolygon spacing and the frame size follow the machine's plan
and are not saved. Shadows, the shadow map size and bounce light are
saved, but the plan caps them: a scene saved with bounce light on opens
with it off on a machine whose plan has it off. Flags given on the
command line (`--time`, `--sun`, `--no-bounce` and the rest) override the
scene.

## Not done yet

- An asset used by several objects is read once per object, unless they
  are written as instances (`instance_of`).
- A scene saved with the high-resolution T-rex asks for it on every
  machine; the plan does not swap in the low-resolution one.
- Cameras are orbit cameras: no roll, no film back, no clipping planes
  to set.
- Textures and levels of detail are the asset's; a scene overrides only
  material values.
