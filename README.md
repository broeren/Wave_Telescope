# Wave_Telescope
Example code implementing the wave-telescope technique of wavevector identification. The model found in the [NEWTSS](https://github.com/broeren/NEWTSS) repo estiamtes the uncertainty in this estimation of wavevector depending on the geometry of the spacecraft configuration used.

## Application
This code uses synthetic measurements from a four-spacecraft configuration that is a perfect tetrahedron. The spacecraft measure plane-waves of the form
$ e^{i(2 \pi \omega t + \mathbf{k} \cdot \mathbf{r} + \phi)} $

```
omega true: 5.49
omega found: 5.625 +/- 0.625
omega error: 2.49% 

k true: [k_r, k_theta, k_phi] = [1.000,2.827,1.257]
k found: [k_r, k_theta, k_phi] = [1.020,2.832,1.257] +/- [0.041,0.087,0.044]
k error: 0.64% 
```
