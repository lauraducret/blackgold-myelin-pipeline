# blackgold-myelin-pipeline

**BlackGold Myelin Pipeline: Quantitative image analysis pipelines for BlackGold-stained myelin: cortical fiber orientation (tensor-based) and striatal pencil-fiber segmentation (Cellpose).**

This pipeline was developed to characterize myelin integrity in a novel rodent mouse model, as part of a Master's thesis at EPFL (Neuro-X), in collaboration with the Sulzer Lab, Columbia University. Myelin was stained using the reagent Black-Gold II (Schmued et al., 2008). 

## Overview of the pipeline
Two independent pipelines, both starting from BlackGold-stained brain coronal sections with manually drawn ROIs (see [Manual preprocessing](#manual-preprocessing-in-fiji) below):

1. **Cortical fiber orientation** [1_fibers_analysis.ipynb](notebooks/1_fibers_analysis.ipynb) — restricts each image to the cortex, enhances elongated fiber structures with directional filtering, and classifies fibers as radial or tangential using structure-tensor analysis, relative to the cortical surface. /!\ The Directional Filtering step must be run as a **standalone Fiji macro** (`fiji_macros/directional_filtering.ijm`) in Fiji (ImageJ). 


2. **Striatal pencil-fiber counting** [2_striatum_cellpose_fiber_counting_pipeline.ipynb](notebooks/2_striatum_cellpose_fiber_counting_pipeline.ipynb) — runs a custom-trained Cellpose-SAM model to detect and count striatal pencil fibers within the striatum and computes fiber density normalized to ROI area. The trained model used by the script is `cellpose_model/blackgold_striatal_pencilfiber_cellpose` — too large for git, see [Model weights](#model-weights) below to download it.


3. **Segmentation between ventral and dorsal striatum** [3_ventral_dorsal_striat.ipynb](notebooks/3_ventral_dorsal_striat.ipynb) — uses the output of [2_striatum_cellpose_fiber_counting_pipeline.ipynb](notebooks/2_striatum_cellpose_fiber_counting_pipeline.ipynb) and analyzes separately the dorsal and ventral striatum by separating them.

![cortical fiber orientation pipeline](/images/cortical_pipeline_overview.png)

*(A) Cortex restriction and radial/tangential classification relative to the cortical inner surface. (B) Output of the trained Cellpose-SAM model with segmented myelin pencil-fibers in the striatum. (C) Directional filtering sub-steps (i–v), run via the Fiji macro described below. (D) Separation between dorsal and ventral striatum.*

Both pipelines match raw images to their corresponding Fiji ROI sets automatically by filename (see `match_images_to_rois` in `utils.py`). No separate metadata table required.

## Repository structure

```
blackgold-myelin-pipeline/
├── notebooks/
│   └── 1_fibers_analysis.ipynb                     # cortical radial/tangential fiber pipeline
│   └── 2_striatum_cellpose_fiber_counting_pipeline.ipynb  # striatal Cellpose pencil-fiber pipeline
│   └── utils.py                                    # shared functions (ROI I/O, matching, overlays)
├── requirements.txt
├── LICENSE
├── fiji_macros/
│   └── directional_filtering.ijm               # required manual step for pipeline 1 (see below)
├── cellpose_model/                              # not tracked in git — see "Model weights" below
│   └── blackgold_striatal_pencilfiber_cellpose  # custom-trained Cellpose model, used by pipeline 2
├── data/
│   ├── tiff_files/                             # raw BlackGold images (.tif/.tiff)
│   └── roi_zip/                                # matching Fiji RoiSet.zip per image
└── outputs/
    ├── 1_cortical_fibers/
    │   ├── preprocessed/
    │   ├── tables/
    │   └── figures/
    └── 2_cellpose_striatum_counts/
        ├── masks/
        ├── overlays/
        ├── tables/
        └── figures/
```

## Requirements

- Python 3.13 (see `requirements.txt` for the full pinned environment)
- **A local Fiji/ImageJ installation** — required for two things: (1) the manual ROI-drawing step described below, and (2) pipeline 1's directional filtering step, which is run as a Fiji macro rather than through Python (see [why](#rationale-on-fiji-macro-for-directional-filtering)). Pipeline 1 also uses `pyimagej` to interface with a local ImageJ/Fiji installation from Python, which additionally requires a working Java installation — see [pyimagej's setup instructions](https://github.com/imagej/pyimagej) if you don't already have Fiji configured.
- A CUDA-capable GPU is recommended for pipeline 2 (Cellpose-SAM) but not required — it will fall back to CPU, just considerably slower.

### Installation

```bash
python3 -m venv venv
source venv/bin/activate          # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Model weights

The custom-trained Cellpose model (`cellpose_model/blackgold_striatal_pencilfiber_cellpose`, ~1.1GB) is too large for git and is **not** tracked in this repository. It's distributed as a [GitHub Release](https://github.com/lauraducret/blackgold-myelin-pipeline/releases) asset instead.

Before running `2_striatum_cellpose_fiber_counting_pipeline.ipynb`, download it from the [Releases page](https://github.com/lauraducret/blackgold-myelin-pipeline/releases) and place it at:

```
cellpose_model/blackgold_striatal_pencilfiber_cellpose
```

## Manual preprocessing (in Fiji)

Before running either notebook, each raw image needs a matching Fiji ROI set, saved as `<image_filename>_RoiSet.zip` in `data/roi_zip/` (see `match_images_to_rois` in `utils.py` for the exact matching rule — the ROI zip's base name just needs to be contained in the image filename).

For each image, draw and name the following ROIs in Fiji's ROI Manager, then save the full set:

| ROI name | Used by | What to draw |
|---|---|---|
| `Cortex` | Pipeline 1 | Outline of the cortical region of interest |
| `Inner` | Pipeline 1 | Inner surface of the cortical region of interest |
| `Striatum_R` | Pipeline 2 | Outline of the right striatum|
| `Striatum_L` | Pipeline 2 | Outline of the left striatum |
| `sep_R` | Pipeline 2 | Line separating dorsal from ventral striatum (right) |
| `sep_L` | Pipeline 2 | Line separating dorsal from ventral striatum (right)|


![Manual ROIs segmentation on coronal brain slice](/images/representation_manual_ROIs.png)

## Usage

1. Place raw `.tif`/`.tiff` images in `data/tiff_files/` and their matching ROI zips in `data/roi_zip/`.
2. Run `1_fibers_analysis.ipynb` for cortical fiber orientation analysis (remember the manual Fiji macro step partway through).
     After running the "restrict to cortex" cell in `1_fibers_analysis.ipynb`, open Fiji, run the macro [directional_filtering.ijm](fiji_macros/directional_filtering.ijm) on the `outputs/1_cortical_fibers/preprocessed/` folder, then continue with the next notebook cell.
3. Run `2_striatum_cellpose_fiber_counting_pipeline.ipynb` for striatal pencil-fiber counting.
4. Results (tables, overlays, figures) are written to `outputs/`.

### Rationale on FIJI Macro for directional filtering

The directional filtering step in pipeline 1 (MorphoLibJ, Max/Opening, line length 50px, 5 directions) is run as a **standalone Fiji macro** (`fiji_macros/directional_filtering.ijm`), not called directly from the notebook: running it through `pyimagej` inside the notebook causes the kernel to freeze. 

## References

Staining method:
> Schmued LC, Bowyer JF, Cozart M, Heard D, Binienda Z, Paule M. Introducing Black-Gold II, a highly soluble gold phosphate complex with several unique advantages for the histochemical localization of myelin. *Brain Res*. 2008;1229:210-217. doi:10.1016/j.brainres.2008.06.129

Cellpose-SAM:
> Pachitariu, M., Rariden, M., & Stringer, C. (2025). Cellpose-SAM: superhuman generalization for cellular segmentation. *bioRxiv*.

Cellpose 2.0:
> Pachitariu, M. & Stringer, C . Cellpose 2.0: How to Train Your Own Model”. In: Nature Methods 19.12 (Dec. 2022), pp. 1634–1641. issn: 1548-7105. doi: 10.1038/s41592-022-01663-4. PMID: 36344832.

## Acknowledgments

The Cellpose-SAM segmentation workflow in `2_striatum_cellpose_fiber_counting_pipeline.ipynb` is adapted from the official [Cellpose-SAM example notebook](https://github.com/MouseLand/cellpose/tree/main), from the [Cellpose](https://github.com/MouseLand/cellpose) repository (Pachitariu lab / MouseLand). The custom model, ROI-matching logic, striatum-specific counting, and density calculations are original to this pipeline.

## Citation

If you use this pipeline, please cite:

> Ducret, Laura, *Characterization of a new genetic rodent model of PARK9-associated MSA/PD*. Master's thesis, EPFL, 2026.

## License

MIT — see [LICENSE](LICENSE).
