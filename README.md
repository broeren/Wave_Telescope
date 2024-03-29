# Wave_Telescope
Example code implementing the wave-telescope technique of wavevector identification. The model found in the [NEWTSS](https://github.com/broeren/NEWTSS) repo estiamtes the uncertainty in this estimation of wavevector depending on the geometry of the spacecraft configuration used.

## Application
This code uses synthetic measurements from a four-spacecraft configuration that is a perfect tetrahedron. The spacecraft measure plane-waves (of magnetic fields) that are of the form
```math
 B(\mathbf{r}, t) = e^{i(2 \pi \omega t + \mathbf{k} \cdot \mathbf{r} + \phi)} .
```
It then uses the [wave-telescope technique](https://doi.org/10.1029/2021JA030165) to estimate the wavevector and frequency of the waves measured using only data collected by the spacecrafts. This estimation is done by computing the spectral energy density $P(\mathbf{k},\omega)$ for a range of wavevectors $\mathbf{k}$ and frequencies $\omega$. The maximum of this function should correspond to the true wavevector and frequency of the measured wave, assuming that no spatial aliasing has occured due to a poorly shaped configuration of spacecraft.

## Results
For an example wave with the below frequency (`omega`) and wavevector (`k`), we show the accuracy of the method.
```
omega true: 5.49
omega found: 5.625 +/- 0.625
omega error: 2.49% 

k true: [k_r, k_theta, k_phi] = [1.000,2.827,1.257]
k found: [k_r, k_theta, k_phi] = [1.020,2.832,1.257] +/- [0.041,0.087,0.044]
k error: 0.64% 
```
We plot the spectral energy density over each of our four independent variables (frequency and 3 spherical components of wavevector).

<p align="center">
<img src="figures/WaveTelescope_Scan.png" alt="Example Scan" width="600"/>

We see that there is a large spike at the true value of wavevector and frequency (highlighted with vertical dashed lines). In each of these subplots, the spectral energy density is integrated over the variables not shown. We can also plot a spherical heatmap to visualize the directional resolution that the correctly identified wavevector has in 3D space.

<p align="center">
<img src="figures/directional_scan.png" alt="Directional Scan" width="600"/>
