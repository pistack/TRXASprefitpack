# Basic

## import module


```python
import TRXASprefitpack
```

## Get general infomation of module


```python
help(TRXASprefitpack)
```

    Help on package TRXASprefitpack:
    
    NAME
        TRXASprefitpack
    
    DESCRIPTION
        TRXASprefitpack: 
        package for TRXAS pre fitting process which aims for the first order dynamics
        TRXAS stands for Time Resolved X-ray Absorption Spectroscopy
        
        :copyright: 2021-2022 by pistack (Junho Lee)
        :license: LGPL3.
    
    PACKAGE CONTENTS
        mathfun (package)
        tools (package)
    
    VERSION
        0.5.0
    
    
    

## get version information


```python
TRXASprefitpack.__version__
```




    '0.5.0'



## get general information of subpackage
Since 0.5.0 version, docs subpackage is removed


```python
help(TRXASprefitpack.docs)
```


    ---------------------------------------------------------------------------

    AttributeError: module 'TRXASprefitpack' has no attribute 'docs'



```python
help(TRXASprefitpack.mathfun)
```

    Help on package TRXASprefitpack.mathfun in TRXASprefitpack:
    
    NAME
        TRXASprefitpack.mathfun
    
    DESCRIPTION
        mathfun:
        subpackage for the mathematical functions for TRXASprefitpack
        
        :copyright: 2021-2022 by pistack (Junho Lee).
        :license: LGPL3.
    
    PACKAGE CONTENTS
        A_matrix
        broad
        exp_conv_irf
        exp_decay_fit
        irf
        rate_eq
    
    FUNCTIONS
        cauchy_irf(t: Union[float, numpy.ndarray], fwhm: float) -> Union[float, numpy.ndarray]
        
        compute_model(t: numpy.ndarray, eigval: numpy.ndarray, V: numpy.ndarray, c: numpy.ndarray) -> numpy.ndarray
        
        compute_signal_cauchy(t: numpy.ndarray, fwhm: float, eigval: numpy.ndarray, V: numpy.ndarray, c: numpy.ndarray) -> numpy.ndarray
        
        compute_signal_gau(t: numpy.ndarray, fwhm: float, eigval: numpy.ndarray, V: numpy.ndarray, c: numpy.ndarray) -> numpy.ndarray
        
        compute_signal_pvoigt(t: numpy.ndarray, fwhm_G: float, fwhm_L: float, eta: float, eigval: numpy.ndarray, V: numpy.ndarray, c: numpy.ndarray) -> numpy.ndarray
            
        exp_conv_cauchy(t: Union[float, numpy.ndarray], fwhm: float, k: float) -> Union[float, numpy.ndarray]
        
        exp_conv_gau(t: Union[float, numpy.ndarray], fwhm: float, k: float) -> Union[float, numpy.ndarray]
        
        exp_conv_pvoigt(t: Union[float, numpy.ndarray], fwhm_G: float, fwhm_L: float, eta: float, k: float) -> Union[float, numpy.ndarray]
        
        fact_anal_exp_conv(t: numpy.ndarray, fwhm: Union[float, numpy.ndarray], tau: numpy.ndarray, base: Optional[bool] = True, irf: Optional[str] = 'g', eta: Optional[float] = None, data: Optional[numpy.ndarray] = None, eps: Optional[numpy.ndarray] = None) -> numpy.ndarray
        
        fact_anal_rate_eq_conv(t: numpy.ndarray, fwhm: Union[float, numpy.ndarray], eigval: numpy.ndarray, V: numpy.ndarray, c: numpy.ndarray, exclude: Optional[str] = None, irf: Optional[str] = 'g', eta: Optional[float] = None, data: Optional[numpy.ndarray] = None, eps: Optional[numpy.ndarray] = None) -> numpy.ndarray
        
        gau_irf(t: Union[float, numpy.ndarray], fwhm: float) -> Union[float, numpy.ndarray]
            Compute gaussian shape irf function
        
        gen_theory_data(e: numpy.ndarray, peaks: numpy.ndarray, A: float, fwhm_G: float, fwhm_L: float, peak_factor: float, policy: Optional[str] = 'shift') -> numpy.ndarray
        
        model_n_comp_conv(t: numpy.ndarray, fwhm: Union[float, numpy.ndarray], tau: numpy.ndarray, c: numpy.ndarray, base: Optional[bool] = True, irf: Optional[str] = 'g', eta: Optional[float] = None) -> numpy.ndarray

        pvoigt_irf(t: Union[float, numpy.ndarray], fwhm_G: float, fwhm_L: float, eta: float) -> Union[float, numpy.ndarray]
            Compute pseudo voight shape irf function
        
        rate_eq_conv(t: numpy.ndarray, fwhm: Union[float, numpy.ndarray], abs: numpy.ndarray, eigval: numpy.ndarray, V: numpy.ndarray, c: numpy.ndarray, irf: Optional[str] = 'g', eta: Optional[float] = None) -> numpy.ndarray

        solve_l_model(equation: numpy.ndarray, y0: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        
        solve_model(equation: numpy.ndarray, y0: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        
        solve_seq_model(tau)
    
    DATA
        __all__ = ['gen_theory_data', 'gau_irf', 'cauchy_irf', 'pvoigt_irf', '...
    
    
    

## Get general information of function defined in TRXASprefitpack


```python
help(TRXASprefitpack.exp_conv_gau)
```

    Help on function exp_conv_gau in module TRXASprefitpack.mathfun.exp_conv_irf:
    
    exp_conv_gau(t: Union[float, numpy.ndarray], fwhm: float, k: float) -> Union[float, numpy.ndarray]
        Compute exponential function convolved with normalized gaussian
        distribution
        
        Args:
          t: time
          fwhm: full width at half maximum of gaussian distribution
          k: rate constant (inverse of life time)
        
        Returns:
         Convolution of normalized gaussian distribution and exponential
         decay :math:`(\exp(-kt))`.
    
    
