# Trait_video
Analyse et traitement vidéo - M2 3IR- USPN
## Dehazing methods
We propose two methods for dehazing: Priors based method & Hybrid based method
### 1. Dark channel prior (Priors based method)
./code/dcp/dehaze.ipynb
+ Extract dark channel from hazy input image I.
+ Estimate Atmospheric light A.
+ Initial transmission map (omega=0.85), then refine by guided-filter (the size of filter r=60 and epsilon=1e-4).
+ Dehaze input from Atmospheric A and refined transmission map (lower bound tx=0.1 to restrict transmission).
### 2. Linear model based on Color Attenuation Prior (Hybrid based method)
./code/cap/dehaze.py
+ Build linear model to find linear model coefficients theta0 & 1 & 2 sigma is the loss of linear model.
+ Extract the depth map of hazy input from linear model coefficients, then refine it by guided filter.
+ Estimate Atmospheric light A.
+ Dehaze input from Atmospheric A and refined depth map (beta=1).
## Dehazing video
./code/main_dehaze_video.ipynb
+ Dehazing video by applying both above methods.
+ Consider two cases: With and without global atmospheric value.
## Evaluation dehazed video
./code/evaluation.ipynb
+ Compute quality metrics PSNR, SSIM, MSE of dehazed video.
+ Applied datasets: RESIDE, REVIDE and NYU.
