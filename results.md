# Quantitative evaluation results

All results were measured using this implementation.
Please see the thesis (#TODO: URL) for a precise description of evaluation methodology, specification of preset name structure and for additional results. Best results indicated by "*", best preset in italic.

### Average results on MipNerf360 (9 scene version)

| Preset                          | PSNR         | SSIM        | LPIPS       | LPIPS(VGG)   |                                                                                                                                                        
|---------------------------------|--------------|-------------|-------------|--------------|                                                                                                                                                        
| Sfm                             | 27.447       | 0.815       | 0.179       | 0.256        |                                                                                                                                                        
| DA V2 Indoor [adaptive] Ransac  | 27.459       | 0.822       | 0.156       | 0.231        |
| DA V2 Outdoor [adaptive] Ransac | 27.045       | 0.819       | 0.161       | 0.238        |
| *Metric3Dv2 [adaptive] Ransac*    | * *27.653* | * *0.826* | * *0.149* | * *0.226*  |
| MoGe [adaptive] Ransac          | 27.413       | 0.825       | 0.150       | 0.227        |
| UniDepth [adaptive] Ransac      | 27.466       | 0.823       | 0.152       | 0.229        |

### Average results on Tanks&Temples

| Preset                          | PSNR         | SSIM        | LPIPS       |                                                                                                                                                                       
|---------------------------------|--------------|-------------|-------------|                                                                                                                                                                       
| Sfm                             | * *23.727* | 0.830       | 0.165       |                                                                                                                                                                       
| DA V2 Indoor [adaptive] Ransac  | 23.000       | 0.817       | 0.167       |
| DA V2 Outdoor [adaptive] Ransac | 23.415       | 0.827       | 0.161       |
| *Metric3Dv2 [adaptive] Ransac*    | 23.636       | * *0.833* | * *0.154* |
| MoGe [adaptive] Ransac          | 23.413       | 0.828       | 0.158       |
| UniDepth [adaptive] Ransac      | 23.529       | 0.830       | 0.157       |

# Qualitative Comparison

View synthesis results for select scenes from the MipNerf360 dataset are shown below.
See the thesis for more comparisons.

![Qualitative comparison of initialization methods](qualitative_results_full.png)

