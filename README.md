## Fork and Modifications

This repository is a fork of the official SepReformer implementation ([original repo](https://github.com/dmlguq456/SepReformer)).  In our fork, we have **only** added:

- **ECAPA-TDNN integration** for target-speaker conditioning:
  - After the SepReformer blind separation stage, we extract a fixed-length speaker embedding from a short enrollment sample using ECAPA-TDNN.
  - We compute cosine similarity between this embedding and each of SepReformer’s output streams.
  - The stream with the highest similarity score is selected and optionally refined, enabling targeted voice extraction in multi-speaker recordings.

All other code, configuration, and pretrained models remain unchanged from the original.  

For full details on SepReformer’s architecture and training, please refer to the [original SepReformer repo](https://github.com/dmlguq456/SepReformer).  
