
### Tensor-Based Interpolation of Multivariate Economic Time Series

### Overview
This repository demonstrates the interpolation of multivariate economic time series using the Functional Tucker Decomposition (FTD). The experiment uses monthly data from three OECD indicators — SHORT_TERM_INTEREST_RATE, HICP, and RETAIL_TRADE_VOLUME — across nine countries. The covered time frame spans from January 2003 to December 2024.[^1]

The goal is to interpolate full time series based on sparse training data: only every fourth timestamp is used during training. After fitting the FTD model, its interpolation performance is compared against a naïve baseline and a spline-based model.

The project was developed with MATLAB R2025b.

### Theory
The theoretical framework of the Functional Tucker Decomposition (FTD) is described in [todo].

### Usage
The script `prepare_data.m` loads the dataset `oecd.csv`, applies mean-normalization, and removes linear trends.  
Next, `build_models.m` determines the optimal Gaussian kernel parameter `d`, constructs the FTD model, and visualizes the resulting interpolations against the naïve and spline-based methods.

### Example
Below is an example of the interpolated time series of Belgium (BE) indicators for SHORT_TERM_INTEREST_RATE, HICP, and RETAIL_TRADE_VOLUME from beginning of 2020 to end of 2024 using:
- `d = 4`  
- `λ = 1`  
- `MAXITERS = 25`  
- `TAU = 1e-9`

#### SHORT_TERM_INTEREST_RATE of Belgium between Jan 2020 and Dec 2024
<img width="2467" height="1217" alt="image" src="https://github.com/user-attachments/assets/254dcd4f-1852-48c0-b630-091d4e5d488e" />

#### HICP of Belgium between Jan 2020 and Dec 2024
<img width="2443" height="1217" alt="image" src="https://github.com/user-attachments/assets/f6389a90-66d5-4ea4-afc0-276b5279c282" />

#### RETAIL_TRADE_VOLUME of Belgium between Jan 2020 and Dec 2024
<img width="2458" height="1217" alt="image" src="https://github.com/user-attachments/assets/70fbc3e0-9107-490b-a210-19f3d536a0fe" />

[^1]: The data was independently sourced from [data-explorer.oecd](https://data-explorer.oecd.org/) and combined by date.
