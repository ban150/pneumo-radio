// Même prétraitement que AI_Medical/data/pretraitement.py puis entrainement.py :
// niveaux de gris -> écrêtage percentiles 0,5/99,5 -> 0-255 -> 512x512 (INTER_AREA) -> arrondi -> 224x224 (INTER_AREA) -> normalisation ImageNet
"use strict";
let session = null, config = null;
const etat = document.getElementById("etat");

if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});

async function charger() {
  try {
    config = await (await fetch("config.json")).json();
    ort.env.wasm.wasmPaths = new URL("vendor/", location.href).href;
    ort.env.wasm.numThreads = 1;
    session = await ort.InferenceSession.create("modele_fp32.onnx", { executionProviders: ["wasm"] });
    etat.textContent = "Modèle prêt (MobileNetV2, " + config.version + "). Aucune image n'est envoyée sur internet.";
  } catch (e) {
    etat.textContent = "Erreur de chargement du modèle : " + e.message;
  }
}

// Redimensionnement par moyenne de surface (équivalent de cv2.INTER_AREA en réduction)
function redimAire(src, w, h, W, H) {
  const out = new Float32Array(W * H), sx = w / W, sy = h / H;
  for (let Y = 0; Y < H; Y++) {
    const y0 = Y * sy, y1 = y0 + sy;
    for (let X = 0; X < W; X++) {
      const x0 = X * sx, x1 = x0 + sx;
      let s = 0, a = 0;
      for (let y = Math.floor(y0); y < Math.min(h, Math.ceil(y1)); y++) {
        const wy = Math.min(y + 1, y1) - Math.max(y, y0);
        if (wy <= 0) continue;
        for (let x = Math.floor(x0); x < Math.min(w, Math.ceil(x1)); x++) {
          const wx = Math.min(x + 1, x1) - Math.max(x, x0);
          if (wx <= 0) continue;
          s += src[y * w + x] * wx * wy; a += wx * wy;
        }
      }
      out[Y * W + X] = a ? s / a : 0;
    }
  }
  return out;
}

function percentile(trie, p) {  // interpolation linéaire, comme numpy
  const k = (trie.length - 1) * p / 100, f = Math.floor(k), c = Math.min(f + 1, trie.length - 1);
  return trie[f] + (trie[c] - trie[f]) * (k - f);
}

function pretraiter(img) {
  const w = img.naturalWidth, h = img.naturalHeight;
  const cv = document.createElement("canvas"); cv.width = w; cv.height = h;
  const cx = cv.getContext("2d", { willReadFrequently: true });
  cx.drawImage(img, 0, 0);
  const px = cx.getImageData(0, 0, w, h).data;
  const gris = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) gris[i] = 0.299 * px[4 * i] + 0.587 * px[4 * i + 1] + 0.114 * px[4 * i + 2];
  const trie = Float32Array.from(gris).sort();
  const bas = percentile(trie, config.percentiles[0]), haut = percentile(trie, config.percentiles[1]);
  if (haut - bas < 1e-3) throw new Error("image uniforme");
  for (let i = 0; i < gris.length; i++) gris[i] = Math.min(1, Math.max(0, (gris[i] - bas) / (haut - bas))) * 255;
  const i512 = redimAire(gris, w, h, 512, 512).map(Math.round);
  const i224 = redimAire(i512, 512, 512, 224, 224);
  const n = 224 * 224, t = new Float32Array(3 * n);
  for (let c = 0; c < 3; c++)
    for (let i = 0; i < n; i++) t[c * n + i] = (i224[i] / 255 - config.moyenne[c]) / config.ecart_type[c];
  return { tenseur: t, apercu: i224 };
}

function dessiner(apercu) {
  const cv = document.getElementById("apercu"), cx = cv.getContext("2d"), im = cx.createImageData(224, 224);
  for (let i = 0; i < apercu.length; i++) { const v = apercu[i]; im.data.set([v, v, v, 255], 4 * i); }
  cx.putImageData(im, 0, 0);
}

async function analyser(fichier) {
  if (!session || !fichier) return;
  etat.textContent = "Analyse…";
  const img = new Image();
  img.src = URL.createObjectURL(fichier);
  await img.decode();
  try {
    const t0 = performance.now();
    const { tenseur, apercu } = pretraiter(img);
    const sortie = await session.run({ image: new ort.Tensor("float32", tenseur, [1, 3, 224, 224]) });
    const p = 1 / (1 + Math.exp(-sortie.logit.data[0]));
    const ms = Math.round(performance.now() - t0);
    dessiner(apercu);
    document.getElementById("resultat").hidden = false;
    document.getElementById("score").textContent = (p * 100).toFixed(0) + " %";
    const pos = p >= config.seuil_validation, v = document.getElementById("verdict");
    v.textContent = pos ? "Pneumonie suspectée : avis médical nécessaire" : "Pas de signe de pneumonie détecté par le modèle";
    v.className = "verdict " + (pos ? "pos" : "neg");
    document.getElementById("details").textContent =
      "Probabilité estimée par le modèle (seuil de décision " + (config.seuil_validation * 100).toFixed(0) +
      " %, choisi sur les données de validation). Analyse faite sur cet appareil en " + ms + " ms. " +
      "Un résultat négatif n'exclut pas une pneumonie, surtout chez un enfant malade.";
    etat.textContent = "";
  } catch (e) {
    etat.textContent = "Image non analysable : " + e.message;
  } finally {
    URL.revokeObjectURL(img.src);
  }
}

for (const id of ["fichier", "photo"])
  document.getElementById(id).addEventListener("change", e => analyser(e.target.files[0]));
charger();
