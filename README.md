# Blueberry Ripeness Estimation via Computer Vision

Prototype pipeline that estimates the ripeness of individual blueberries directly from field
images, without requiring a manually labeled ripeness dataset.

![Ripeness scoring example](docs/ripeness_grid_example.png)

## Problem

Manually grading blueberry ripeness at harvest is slow and inconsistent between graders. Building
a fully supervised ripeness classifier requires a labeled dataset (ripeness class per berry),
which is expensive to produce. This project explores a **label-free alternative**: can visual
similarity in a pre-trained CNN's feature space act as a usable ripeness signal on its own?

## Approach

1. **Detection → crop**: starting from bounding boxes for individual berries in a field photo
   (produced by any object detector, e.g. YOLO), each berry is cropped into its own sub-image.
2. **Feature extraction**: every crop is passed through **VGG16** (ImageNet weights, classification
   head removed) to get a high-dimensional visual embedding — no ripeness labels needed at this
   stage.
3. **Dimensionality reduction**: embeddings are projected to 3D with **PCA**, giving a compact
   color/texture "fingerprint" per berry.
4. **Ripeness scoring**: two reference points are anchored in that 3D space — one representative of
   clearly unripe berries, one of clearly ripe berries. Each berry is scored by its relative
   distance to both anchors:

   ```
   ripeness% = d_unripe / (d_unripe + d_ripe) × 100
   ```

   0% = sitting on the unripe anchor, 100% = sitting on the ripe anchor.

## Results

Applied to a field image with 332 detected berries, producing a ripeness percentage for every
individual berry (see grid image above). The first PCA component correlates visibly with the
blue/purple vs. green/white color progression of ripening.

## Stack

Python · OpenCV · TensorFlow/Keras (VGG16 transfer learning) · scikit-learn (PCA) · Plotly
(interactive 3D visualization) · Matplotlib

## Limitations

- **No ground-truth validation**: ripeness anchors were chosen by visual inspection, not validated
  against real maturity measurements (e.g. Brix/sugar content, harvest date). Treat the output as
  a **relative ranking signal**, not a calibrated ripeness percentage.
- **Single lighting/scene condition**: features come from one field image; robustness across
  lighting, camera angle, and cultivar is untested.
- **Detection is assumed given**: this notebook starts from existing bounding boxes; it does not
  cover the detector itself.

## Possible extensions

- Replace the 2-anchor heuristic with a small regression model trained on a handful of labeled
  ripeness scores.
- Test on multiple frames / harvest dates for robustness.
- Compare VGG16 embeddings against a lighter backbone (e.g. MobileNet) for on-device / field
  deployment.

## How to run

```bash
pip install -r requirements.txt
```

Open `Blueberry_Ripeness_Estimation.ipynb` in Jupyter or Google Colab. Update `IMG_PATH` and the
bounding-box source in the notebook to point to your own field image and detections — source
image/labels are not included in this repo.

## Repo structure

```
.
├── Blueberry_Ripeness_Estimation.ipynb
├── requirements.txt
├── README.md
└── docs/
    └── ripeness_grid_example.png
```
