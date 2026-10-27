# SPDX-License-Identifier: Apache-2.0
"""Independent NumPy oracle for VKL temporal best-observation forward outputs."""
import numpy as np


def temporal_best_observation(pairwise_equivalence,target_quality,neighbor_quality,*,pairwise_quality=None,pairwise_confidence=None):
    equivalent=np.asarray(pairwise_equivalence,dtype=bool)
    target=np.asarray(target_quality,dtype=np.float32)
    quality=np.asarray(neighbor_quality if pairwise_quality is None else pairwise_quality,dtype=np.float32)
    confidence=equivalent.astype(np.float32) if pairwise_confidence is None else np.asarray(pairwise_confidence,dtype=np.float32)
    candidates=np.where(equivalent,quality,np.float32(-np.inf))
    # np.argmax, like Torch max, chooses the first NaN and otherwise first max.
    index=np.argmax(candidates,axis=1)
    best=candidates[np.arange(target.size),index]
    wins=~np.isfinite(best)|(target>=best)
    with np.errstate(invalid='ignore',over='ignore'):
        advantage=np.where(wins,np.float32(0),np.maximum(best-target,np.float32(0)))
        replacement=np.where(wins,np.float32(0),confidence[np.arange(target.size),index]*advantage)
    return {'best_index':np.where(wins,0,index+1).astype(np.int64),
            'best_quality':np.where(wins,target,best), 'replaceability':advantage,
            'replacement_confidence':replacement}
