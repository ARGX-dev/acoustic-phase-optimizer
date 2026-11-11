# Third-party code in web/vendor

Everything here is loaded only in the browser, and only when a file that needs it is imported. Files are
unmodified copies of the published packages unless noted. `sha256.txt` lists a checksum for each file.

| Path | Package and version | Licence |
|---|---|---|
| three/build, three/examples/jsm (except below) | three.js 0.170.0 | MIT (three/LICENSE) |
| three/examples/jsm/loaders/USDLoader.js, loaders/usd/* | three.js 0.186.0 (first release with a binary USD reader). **Modified:** USDCParser.js decodes inlined double and half vectors (marked "Local patch") | MIT |
| three/examples/jsm/libs/draco/gltf/* | Draco decoder as shipped with three.js 0.170.0 | Apache-2.0 (Google) |
| three/examples/jsm/libs/basis/* | Basis Universal transcoder as shipped with three.js 0.170.0 | Apache-2.0 (Binomial) |
| three/examples/jsm/libs/meshopt_decoder.module.js | meshoptimizer decoder as shipped with three.js 0.170.0 | MIT (Arseny Kapoulkine) |
| three/examples/jsm/libs/ktx-parse.module.js, zstddec.module.js | as shipped with three.js 0.170.0 | MIT (Don McCurdy) |
| three/examples/jsm/libs/fflate.module.js | as shipped with three.js 0.170.0 | MIT (Arjun Barrett) |
| web-ifc/web-ifc-api.js, web-ifc/web-ifc.wasm | web-ifc 0.0.78 (ThatOpen), unmodified | MPL-2.0 (web-ifc/LICENSE.md); source at https://github.com/ThatOpen/engine_web-ifc |
