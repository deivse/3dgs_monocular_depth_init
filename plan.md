# TODO

- [x] Fix GPU memory not being freed when running via evaluator.py and after depth inference
- [x] Add my impl as nerfbaselines method and use nerfbaselines for evaluation.
- [x] Export init pts to PLY using Open3d
- [x] What makes it work so well on "room" and so bad on "kitchen" -> Depth alignment. Potential future improvement. potential culprit - sfm points projected into image bounds that are actually covered by some other points closer to camera. 
- [ ] Test!!
- [ ] Outlier removal from final point cloud [using open3d](https://www.open3d.org/docs/release/tutorial/geometry/pointcloud_outlier_removal.html)? 
- [ ] Analyze results, write report, make presentation, get zapocet.
- [ ] ? Merging step

## Overall direction
Focus on doing evaluations, focus on why some methods predict better than others.
Try to understand what's happening there. Would be sufficient for report.

## Questions:
- How much does depth predictor accuracy affect the results?
- How much does the initial number of splats (regardless of precision) affect the results?

## What to test
- Try to add noise to depths (gaussian noise), see how much it impacts things
- Try running things with less downsamling -> Does the raw number of points help or number of points that are close to surfaces? 
- Try running things at higher resolution (are my images downsampled in current runs?) (Motivation similar to previous one)

# Subjective quality of results on different 360v2 scenes
- Very good
- Pretty good (points correspond to object, low density of generated points in empty space)
    - Garden
- Acceptable (points aligned to the focal point of the scene, but high depth alignment variance in general)
    - Counter - The counter itself is fuzzy but ok, the further parts of the room are badly aligned, likely due to disproportionate parts of the images being covered by the counter itself.
    - Bicycle - Quite fuzzy, metric3d has sky problems. Depth alignment meh.
- Fail (Highly inconsistent )

# Future improvements
- Ways to improve point alignment to produce more precise depth.
