**Metrics**

The MSE and PSNR are commonly used metrics for evaluating image compression quality.

The **Mean Squared Error (MSE)** represents the average squared difference between the pixel values of the compressed and original images. A lower MSE indicates less overall error between the compressed and original images.

The **Peak signal-to-noise ratio (PSNR)** is a metric that measures the quality of a reconstructed image compared to the original, expressed in decibels. A higher PSNR value signifies better quality in the reconstructed image, indicating that it closely resembles the original. Conversely, a lower PSNR value indicates a higher level of distortion or noise in the processed image relative to the original image.

In summary, while MSE quantifies the overall squared error between the compressed and original images, PSNR provides a measure of the maximum error. Lower MSE and higher PSNR values are indicative of better image compression quality.

**Observations for Question 5a:**

*Using Gradient Descent function*

**A. A rectangular block of 30X30 is assumed missing from the image**

Metrics : RMSE=4.75 PSNR=2.85

a. High RMSE value indicates a relatively high level of reconstruction error.

b. Low PSNR value indicates noticeable distortion or noise in the reconstructed image compared to the original.

c. Overall, the reconstruction quality is poor, with visible artifacts and diminished fidelity to the original image 

**B. A random subset of 900 (30X30) pixels is missing from the image**

Metrics : RMSE=2.51 PSNR=5.39

a. RMSE of 2.51 and PSNR of 5.39 represent a lower reconstruction error and a higher signal-to-noise ratio compared to the rectangular block missing scenario.

b. The lower RMSE value suggests a better match between the original and reconstructed images, with fewer errors overall.

c. A higher PSNR value indicates improved reconstruction quality, with less visible distortion or noise in the reconstructed image compared to the original.

d. The reconstruction quality is significantly better than the rectangular block missing case, with smoother transitions and better preservation of image features despite the random subset of missing pixels.


==> For a rectangular block of 30X30 is assumed missing from the image :

*Using RFF + Linear Regression* 

RMSE = 0.3096   PSNR = 43.73

*Using Gradient Descent* 

RMSE = 4.85   PSNR = 2.79

Observations :

a. The reconstruction quality using RFF + Linear Regression is significantly better compared to GD. (from RMSE)

b. The reconstructed image using RFF + Linear Regression closely resembles the original image with minimal distortion, while the reconstruction using GD has a lower fidelity and is more distorted. (from PSNR)

c. In summary, RFF + Linear Regression outperforms GD in terms of reconstruction quality, providing a more faithful representation of the original image with lower RMSE and higher PSNR values.

==> For a random subset of 900(30X30) pixels is assumed missing from the image :

*Using RFF + Linear Regression* 

RMSE = 0.3095   PSNR = 43.74

*Using Gradient Descent* 

RMSE = 2.53   PSNR = 5.35

Same observations (as above)

**Observations for Question 5b:**

N = [20, 40, 60, 80]

**Using Gradient Descent function:**

*For rectangular block of NXN missing:*

a. As the size of the missing rectangular block increases (from 30x30 to 80x80), the RMSE values also increase significantly, ranging from approximately 3.25 to 11.16. This indicates a higher level of reconstruction error as the missing region becomes larger.

b. Correspondingly, the PSNR values decrease as the size of the missing rectangular block increases, ranging from approximately 4.16 to 1.21 dB. A lower PSNR value indicates poorer reconstruction quality and higher distortion in the reconstructed image.

c. Overall, the reconstruction quality for the rectangular block missing scenario deteriorates as the size of the missing region increases, leading to higher RMSE values and lower PSNR values.

*For random NXN pixels missing:*

a. The RMSE values remain relatively stable across different values of N (20, 40, 60, 80), ranging from approximately 2.53 to 2.69. This suggests that the reconstruction error does not vary significantly with changes in the image size.

b. Similarly, the PSNR values exhibit minor variations across different N values, ranging from approximately 5.04 to 5.36 dB. Despite these variations, the PSNR values consistently indicate moderate reconstruction quality and relatively low distortion in the reconstructed image.

c. The stability in RMSE and PSNR values across varying image sizes suggests that the performance of the reconstruction algorithm is robust and does not heavily depend on the specific dimensions of the image patch.

Despite minor variations, the random subset missing scenario consistently demonstrates moderate reconstruction quality and relatively low distortion in the reconstructed image. This suggests that the algorithm is effective in handling missing pixels randomly distributed throughout the image patch compared to the rectangular block missing scenario.

**Using Alternating Least Squares function**

*For rectangular block of NXN missing:*

a. the RMSE values also increase significantly, ranging from 10.34 to 100.81. This indicates a substantial increase in reconstruction error as more pixels are missing from the image.

b.  the PSNR values decrease notably with increasing N, ranging from 1.31 to 0.13 dB. The decrease in PSNR indicates a decline in reconstruction quality and an increase in image distortion as more pixels are missing.

*For random NXN pixels missing:*

a. The RMSE values range from 1.95 to 2.02, indicating consistent reconstruction error regardless of the number of missing pixels.

b. Similarly, the PSNR values range from 6.96 to 6.71 dB as N increases. This indicates a slight decrease in reconstruction quality and a slight increase in image distortion with larger N values, although the change is relatively small and the reconstruction quality remains relatively consistent across different N values.

The rectangular block missing scenario results in significantly higher RMSE and lower PSNR values compared to the random subset missing scenario for all N values. This indicates that the reconstruction quality is much poorer when a contiguous block of pixels is missing compared to when pixels are missing randomly from the image.

**Observations for Question 5c:**

(Using Alternating Least Squares function)

**A. A rectangular block of 30X30 is assumed missing from the image**

Metrics : RMSE=19.85 PSNR=0.68

a. Higher RMSE value tells us that there are large differences between the original and reconstructured images.

b. Low PSNR value means that there is a lot of noise in the reconstructed image, making it look rough.

c. Strange patterns in the missing area

**B. A random subset of 900 (30X30) pixels is missing from the image**

Metrics : RMSE=1.95 PSNR=6.94

a. The RMSE value (1.95) indicates smaller differnces between the original and reconstructed images compared to the rectangular block missing case.

b. The high PSNR value (6.94) means there's less noise in the reconstructed image, making it smoother and closer to the original.

c. No wierd patterns in the image

d. The reconstructed image with a random subset missing looks much better compared to the rectangular block case. The differences aren't as obvious.

e. It is closer to the original image.

==> For a rectangular block of 30X30 is assumed missing from the image :

*Using RFF + Linear Regression* 

RMSE = 0.3096   PSNR = 43.73

*Using ALS* 

RMSE = 19.85   PSNR = 0.68

Observations :

a. The reconstruction quality using RFF + Linear Regression is significantly better compared to ALS. (from RMSE)

b. The reconstructed image using RFF + Linear Regression closely resembles the original image with minimal distortion, while the reconstruction using ALS has a lower fidelity and is more distorted. (from PSNR)

c. In summary, RFF + Linear Regression outperforms ALS in terms of reconstruction quality, providing a more faithful representation of the original image with lower RMSE and higher PSNR values.

==> For a random subset of 900(30X30) pixels is assumed missing from the image :

*Using RFF + Linear Regression* 

RMSE = 0.3095   PSNR = 43.74

*Using ALS* 

RMSE = 1.95   PSNR = 6.94

Same observations (as above)

**Observations for Question 5d:**

**A. A Patch with mainly a single color**

r_values: [5, 10, 25, 50]

RMSE_values: [2.1341381072998047, 0.99114590883255, 0.29061999917030334, 0.15164442360401154]

PSNR_values: [6.343474601846786, 13.658787026055156, 46.58265404700004, 89.27364791099089]

a. As the value of r increases, both RMSE and PSNR values improve significantly.

b. For a single color patch, even with a low-rank approximation (r=5), the reconstruction quality is relatively decent, as indicated by the PSNR value of 6.34 dB.

c. However, as r increases, the reconstruction quality improves drastically, with RMSE decreasing from 2.13 to 0.15 and PSNR increasing from 6.34 to 89.27 dB for r=5 to r=50, respectively.

d. The low-rank matrix factorization is highly effective in capturing the structure of the single-color patch, resulting in highly accurate reconstruction with higher values of r.

**B. A Patch with 2-3 different colors**

r_values: [5, 10, 25, 50]

RMSE_values: [12.88453483581543, 4.584201335906982, 0.788094699382782, 0.187051460146904]

PSNR_values: [1.050705442842858, 2.9531536441148094, 17.177949415333234, 72.37500776448098]

a. Similar to the single-color patch, increasing the value of r leads to significant improvements in reconstruction quality.

b. However, even with a low-rank approximation, the reconstruction quality is comparatively poorer for patches with 2-3 different colors.

c. For r=50, the RMSE decreases to 0.187 and PSNR increases to 72.38 dB, indicating significantly better reconstruction compared to lower values of r.

**C. A Patch with atleast 5 different colors**

r_values: [5, 10, 25, 50]

RMSE_values: [26.10878562927246, 11.09401798248291, 1.4783204793930054, 0.22285181283950806]

PSNR_values: [0.5185170644364022, 1.220283841423865, 9.157588675256862, 60.74821967115557] 

a. For patches with at least 5 different colors, the reconstruction quality is notably poorer compared to patches with fewer colors.

b. Even with a high value of r (e.g., r=50), the RMSE (Root Mean Square Error) remains relatively high at 0.22, indicating significant reconstruction errors, while the PSNR (Peak Signal-to-Noise Ratio) value is modest at 60.75 dB.

c. The presence of multiple colors in the patch introduces complexity, making it challenging for low-rank matrix factorization to accurately capture all variations. This complexity contributes to the relatively high RMSE and modest PSNR values, as the algorithm struggles to faithfully reconstruct the patch.

d. Despite the challenges posed by multiple colors, increasing the value of r still leads to noticeable improvements in reconstruction quality, although the rate of improvement diminishes compared to patches with fewer colors.
