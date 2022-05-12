# powersmooth

Smoothing noisy time-series using ordinary "smooth.m" may cause artifacts, especially if one wants to estimate time-derivatives of the underlying noisy-free dynamics. The function "powersmooth.m" solves this problem, providing a smoothed time-series with faithful estimates of the first n time-derivatives of the noise-free dynamics. The function uses quadratic programming to simultaneously minimize (i) the residuals between the original, noisy time-series and the smoothed curve, and (ii) the (n+1)-th time-derivative of the smoothed curve. The user has to specify the noisy time-series (vec), the desired order n (order), and a regularization weight (weight).
Cite As
