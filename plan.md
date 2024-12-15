# TODO

- [x] Fix GPU memory not being freed when running via evaluator.py and after depth inference
- [x] Add my impl as nerfbaselines method and use nerfbaselines for evaluation.
- [ ] Export init pts to PLY using Open3d
- [ ] Outlier removal from final point cloud [using open3d](https://www.open3d.org/docs/release/tutorial/geometry/pointcloud_outlier_removal.html)? 
- [ ] Test on other datasets -> Tanks and Temples pry dobry
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


# Questions for Monday

- Suggestions where to look for error with low number of sfm points reprojected into image bounds
- Show pointclouds - very bad alignment between predicted depths from different images in scenes other than garden
