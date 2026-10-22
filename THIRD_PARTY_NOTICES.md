# Third-party components

Reference Studio bundles **MediaPipe Tasks Vision 0.10.32** (Apache License 2.0).
Source: https://www.npmjs.com/package/@mediapipe/tasks-vision/v/0.10.32
License text: `vendor/mediapipe/LICENSE`.
Files retained from the upstream package:
- `vendor/mediapipe/vision_bundle.mjs`
- `vendor/mediapipe/wasm/vision_wasm_internal.js`
- `vendor/mediapipe/wasm/vision_wasm_internal.wasm`
- `vendor/mediapipe/wasm/vision_wasm_nosimd_internal.js`
- `vendor/mediapipe/wasm/vision_wasm_nosimd_internal.wasm`

Face detection model: **BlazeFace Short Range float16**, supplied by Google AI Edge / MediaPipe.
Official model source:
https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite

Only the face-location model is used: this feature does not identify who a person is.
The app requests camera access only when the user enables it. Detection and
photo/video generation run on the local device; this feature contains no frame upload endpoint.
The bundled resources are fetched from this site's origin after first load.
