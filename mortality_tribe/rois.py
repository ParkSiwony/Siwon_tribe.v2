"""Hypothesis-driven regions of interest on fsaverage5.

Labels are areas of the HCP multimodal parcellation (Glasser et al., 2016),
the same atlas used in the TRIBE v2 paper, resolved through
``tribev2.utils.get_hcp_roi_indices``. Each ROI is tied to a prediction:

* ``salience``  anterior insula + dorsal ACC: threat / interoceptive arousal
  evoked by the prime (MS and PAIN alike; MS > PAIN would be existential-specific).
* ``vmPFC``     ventromedial PFC / pregenual ACC: self-relevance and valuation;
  TMT predicts boosted valuation of worldview / self-esteem probes after MS,
  SST predicts boosted valuation of social / savoring probes under LT.
* ``PCC_prec``  posterior cingulate / precuneus: self-projection, default mode.
* ``TPJ``       temporo-parietal junction / angular gyrus: mentalizing, social meaning.
* ``ATL``       anterior temporal lobe: semantic and social-conceptual knowledge.
* ``MTL``       hippocampal / parahippocampal / retrosplenial: episodic simulation,
  prospection (imagining a limited or open future).
* ``dlPFC``     dorsolateral PFC: control / suppression (proximal defenses).
* ``OFC``       orbitofrontal cortex: reward value, savoring.
* ``language``  core left-lateralized language areas: control ROI that should
  respond to every sentence and show little prime-specific modulation.
"""

from __future__ import annotations

import numpy as np

HYPOTHESIS_ROIS: dict[str, list[str]] = {
    "salience": ["AVI", "AAIC", "MI", "FOP4", "a24pr", "p24pr", "a32pr", "33pr"],
    "vmPFC": ["10r", "10v", "p32", "s32", "a24", "pOFC"],
    "PCC_prec": ["31pd", "31pv", "v23ab", "d23ab", "7m", "PCV"],
    "TPJ": ["PGi", "PGs", "TPOJ1", "TPOJ2", "TPOJ3"],
    "ATL": ["TGd", "TGv", "TE1a"],
    "MTL": ["H", "EC", "PreS", "PHA1", "PHA2", "RSC", "POS1"],
    "dlPFC": ["46", "9-46d", "a9-46v", "p9-46v", "8C"],
    "OFC": ["OFC", "11l", "13l"],
    "language": ["44", "45", "55b", "STSdp", "STSda", "STSvp", "A5"],
}


def hcp_roi_indices(hemi: str = "both", mesh: str = "fsaverage5") -> dict[str, np.ndarray]:
    """Vertex indices of every hypothesis ROI (needs ``tribev2`` + ``mne``)."""
    from tribev2.utils import get_hcp_roi_indices

    out = {}
    for name, labels in HYPOTHESIS_ROIS.items():
        idx = []
        for label in labels:
            try:
                idx.append(get_hcp_roi_indices(label, hemi=hemi, mesh=mesh))
            except ValueError:  # label naming differs across atlas versions
                continue
        if not idx:
            raise ValueError(f"None of the labels of ROI {name!r} were found")
        out[name] = np.unique(np.concatenate(idx)).astype(int)
    return out


def fake_roi_indices(n_vertices: int) -> dict[str, np.ndarray]:
    """Disjoint contiguous blocks, for the fake predictor and tests."""
    names = list(HYPOTHESIS_ROIS)
    size = n_vertices // (len(names) + 1)  # leave some vertices outside any ROI
    return {n: np.arange(i * size, (i + 1) * size) for i, n in enumerate(names)}


def roi_means(maps: np.ndarray, rois: dict[str, np.ndarray]) -> np.ndarray:
    """``maps[..., n_vertices]`` -> ``[..., n_rois]`` (order of ``rois``)."""
    return np.stack([maps[..., idx].mean(-1) for idx in rois.values()], axis=-1)
