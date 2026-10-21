# Assets

## models/trex.fkj

"Tyrannosaurus rex" by IagoMendez
(https://sketchfab.com/3d-models/tyrannosaurus-rex-497a7f22e77a4678b173fb511bea6064),
licensed under CC BY 4.0 (http://creativecommons.org/licenses/by/4.0/), as
stated in the glTF file's metadata. Changes made here: converted to `.fkj`
with `tools/fkj-import`, scaled to 12.3 m long, base colour baked to
per-triangle texture tiles, other maps dropped.

## models/trex_highres.fkj

The same Tyrannosaurus rex as `trex.fkj`, as a ZBrush 2022 sculpt exported
to OBJ (5.4 M quads, 10.8 M triangles), supplied by Simon Legrand, with the
glTF model's base colour texture. Converted with `tools/fkj-import`, scaled
to 12.3 m long.

## models/car.fkj

A Ford Explorer in "Jurassic Park" (1993) livery, from an FBX with its
textures. The download came with no author or licence information.
Changes made here: converted to `.fkj` with `tools/fkj-import`, scaled to
4.79 m long, base colour baked to per-triangle texture tiles, other maps
dropped.

## chess_set.fkj

No longer in the built-in set (21/10/2026); open it with
`fkjensen assets/chess_set.fkj`.

The Open Chess Set, converted to FkJensen's `.fkj` format (see
`docs/format.md` for what the conversion kept and dropped).

The asset was authored by Moeen Sayed and Mujtaba Sayed for the SideFX
Houdini tutorial "Karma | A Beautiful Game", contributed to the MaterialX
project by SideFX, converted to glTF and MaterialX by Chris Rydalch, Mark
Elendt and Jonathan Stone, made into USD by Pablo Delgado, and published by
the USD Working Group at https://github.com/usd-wg/assets
(`full_assets/OpenChessSet`).

Licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/) by
the Academy Software Foundation. Changes made here: converted from USD to
`.fkj`, textures reduced to per-vertex colour, materials dropped, lights
added.
