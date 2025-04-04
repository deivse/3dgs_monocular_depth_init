# TODO
- [x] RANSAC (MSAC)
- [x] Adaptive sub-sampling (kinda, depending on depth)
- [x] Re-run experiments
- [x] Make it run on RCI cluster
- [x] Merging step: tried voxel downsampling
- [ ] Test on more data - tanksandtemples
- [ ] Capture my own scenes. What I want:
    - Scene with specular surfaces.
    - Scene where only a couple frames capture some secondary object.
- [ ] Test with outlier removal?
- [ ] Ablations?
- [ ] Write the thesis
- [ ] Profit

## What to test
- Try to add noise to depths (gaussian noise), see how much it impacts things
- Try running things with less downsamling -> Does the raw number of points help or number of points that are close to surfaces? 
- Try running things at higher resolution (are my images downsampled in current runs?) (Motivation similar to previous one)
