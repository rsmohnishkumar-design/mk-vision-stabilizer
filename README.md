# MK Vision Stabilizer module

One file, no dependencies: `mk-stabilizer.js`. Exposes `window.MKStabilizer`.
Motion analysis runs in a Web Worker (created from a blob URL), so the editor UI stays responsive.

## Prompt for Claude Code
> Merge `mk-stabilizer.js` into MK Vision as a "Stabilize" tool under the video/effects section.
> Load it as a script (or ES-module wrapper). On a clip: create `new MKStabilizer()`, `await load(clipFileOrVideoEl)`,
> `await analyze({onProgress, signal})` with a progress bar + cancel (AbortController),
> then controls: mode (lock / cinematic / natural), crop 5–30%, strength 10–150%, edge fill (blur / previous / black / zoom).
> Each control change calls `configure({...})` (instant, no re-analysis).
> In the preview/timeline render loop call `stab.drawFrame(ctx, video, mediaTime, w, h)` instead of drawing the raw frame.
> Save `stab.toJSON()` in the project and restore with `fromJSON()`. Use `export()` for a stand-alone render,
> or keep using drawFrame inside MK Vision's existing exporter. Don't change the stabilizer algorithm.

## API
```js
const stab = new MKStabilizer();
const info = await stab.load(fileOrUrlOrVideo);         // {width,height,duration,fps}
const ac = new AbortController();
await stab.analyze({ onProgress: p => {}, signal: ac.signal });
const stats = await stab.configure({
  mode: 'lock',        // 'lock' | 'cinematic' | 'natural'
  crop: 0.2,           // 0.05–0.30 per side
  edgeFill: 'blur',    // 'blur' | 'previous' | 'black' | 'zoom' (crop only, never shows borders)
  strength: 1,         // 0.1–1.5 (cinematic/natural)
});                    // stats.reduction = fraction of shake removed
stab.drawFrame(ctx, video, mediaTime, w, h);            // draw stabilized frame
stab.transformAt(t);                                    // raw correction if MK Vision does its own WebGL warp
stab.peakShake();                                       // {px, deg} telemetry
const blob = await stab.export({ onProgress });         // mp4/webm with audio, real-time render
const saved = stab.toJSON(); await stab.fromJSON(saved);
stab.dispose();
```

## Notes
- Analysis speed: ~20 ms/frame on normal footage, up to ~150 ms/frame on very blurry footage.
- Output zoom = 1 / (1 − 2·crop); 20% crop ≈ 1.67× zoom.
- Tested end-to-end in Chrome (load → analyze → modes → export).

## Test page
`index.html` is a standalone test page with the stabilizer built in. Open it in Chrome (or via GitHub Pages), pick a video, press Stabilize. It is only for testing and is not needed in MK Vision.
