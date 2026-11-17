// Met l'appli et le modèle en cache au premier chargement : ensuite tout fonctionne sans connexion.
const CACHE = "pneumo-radio-v2";
const FICHIERS = ["./", "index.html", "app.js", "config.json", "manifest.webmanifest", "icone.svg", "modele_fp32.onnx",
  "vendor/ort.wasm.min.js", "vendor/ort-wasm-simd-threaded.wasm", "vendor/ort-wasm-simd-threaded.mjs"];
self.addEventListener("install", e => e.waitUntil(caches.open(CACHE).then(c => c.addAll(FICHIERS)).then(() => self.skipWaiting())));
self.addEventListener("activate", e => e.waitUntil(
  caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim())));
self.addEventListener("fetch", e => e.respondWith(caches.match(e.request, { ignoreSearch: true }).then(r => r || fetch(e.request))));
