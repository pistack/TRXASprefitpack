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
        
        :copyright: 2021 by pistack (Junho Lee)
        :license: LGPL3.
    
    PACKAGE CONTENTS
        data_process (package)
        mathfun (package)
        thy (package)
        tools (package)
    
    VERSION
        0.5.0
    
    FILE
        /home/lis1331/Documents/lecture/chem/TRXAS-pre-fit-pack/src/TRXASprefitpack/__init__.py
    
    


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

    AttributeError                            Traceback (most recent call last)

    /tmp/ipykernel_27429/1695297168.py in <module>
    ----> 1 help(TRXASprefitpack.docs)
    

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
        exp_conv_irf
        exp_decay_fit
        irf
        rate_eq
    
    FUNCTIONS
        cauchy_irf(t: Union[float, numpy.ndarray], fwhm: float) -> Union[float, numpy.ndarray]
            Compute lorenzian shape irf function
            
            Args:
              t: time
              fwhm: full width at half maximum of cauchy distribution
            
            Returns:
             normalized lorenzian function.
        
        compute_model(t: numpy.ndarray, eigval: numpy.ndarray, V: numpy.ndarray, c: numpy.ndarray) -> numpy.ndarray
            Compute solution of the system of rate equations solved by solve_model
            Note: eigval, V, c should be obtained from solve_model
            
            Args:
             t: time
             eigval: eigenvalue for equation
             V: eigenvectors for equation
             c: coefficient
            
            Returns:
              solution of rate equation
            
            Note:
              eigval, V, c should be obtained from solve_model.
        
        compute_signal_cauchy(t: numpy.ndarray, fwhm: float, eigval: numpy.ndarray, V: numpy.ndarray, c: numpy.ndarray) -> numpy.ndarray
            Compute solution of the system of rate equations solved by solve_model
            convolved with normalized cauchy distribution
            
            Args:
             t: time
             fwhm: full width at half maximum of normalized cauchy distribution
             eigval: eigenvalue for equation
             V: eigenvectors for equation
             c: coefficient
            
            Returns:
              Convolution of solution of rate equation and normalized cauchy
              distribution
            
            Note:
              eigval, V, c should be obtained from solve_model.
        
        compute_signal_gau(t: numpy.ndarray, fwhm: float, eigval: numpy.ndarray, V: numpy.ndarray, c: numpy.ndarray) -> numpy.ndarray
            Compute solution of the system of rate equations solved by solve_model
            convolved with normalized gaussian distribution
            
            Args:
             t: time
             fwhm: full width at half maximum of normalized gaussian distribution
             eigval: eigenvalue for equation
             V: eigenvectors for equation
             c: coefficient
            
            Returns:
              Convolution of solution of rate equation and normalized gaussian
              distribution
            
            Note:
              eigval, V, c should be obtained from solve_model.
        
        compute_signal_pvoigt(t: numpy.ndarray, fwhm_G: float, fwhm_L: float, eta: float, eigval: numpy.ndarray, V: numpy.ndarray, c: numpy.ndarray) -> numpy.ndarray
            Compute solution of the system of rate equations solved by solve_model
            convolved with normalized pseudo voigt profile
            
            .. math::
            
               \mathrm{pvoigt}(t) = (1-\eta) G(t) + \eta L(t),
            
            G(t) stands for normalized gaussian,
            L(t) stands for normalized cauchy(lorenzian) distribution
            
            Args:
             t: time
             fwhm_G: full width at half maximum of gaussian part
             fwhm_L: full width at half maximum of cauchy part
             eta: mixing parameter
             eigval: eigenvalue for equation
             V: eigenvectors for equation
             c: coefficient
            
            Returns:
              Convolution of solution of rate equation and normalized pseudo
              voigt profile.
            
            Note:
              eigval, V, c should be obtained from solve_model.
        
        exp_conv_cauchy(t: Union[float, numpy.ndarray], fwhm: float, k: float) -> Union[float, numpy.ndarray]
            Compute exponential function convolved with normalized cauchy
            distribution
            
            Args:
               t: time
               fwhm: full width at half maximum of cauchy distribution
               k: rate constant (inverse of life time)
            
            Returns:
              Convolution of normalized cauchy distribution and
              exponential decay :math:`(\exp(-kt))`.
        
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
        
        exp_conv_pvoigt(t: Union[float, numpy.ndarray], fwhm_G: float, fwhm_L: float, eta: float, k: float) -> Union[float, numpy.ndarray]
            Compute exponential function convolved with normalized pseudo
            voigt profile (i.e. linear combination of normalized gaussian and
            cauchy distribution)
            
            :math: `\eta C(\mathrm{fwhm}_L, t) + (1-\eta)G(\mathrm{fwhm}_G, t)`
            
            Args:
               t: time
               fwhm_G: full width at half maximum of gaussian part of
                       pseudo voigt profile
               fwhm_L: full width at half maximum of cauchy part of
                       pseudo voigt profile
               eta: mixing parameter
               k: rate constant (inverse of life time)
            
            Returns:
               Convolution of normalized pseudo voigt profile and
               exponential decay :math:`(\exp(-kt))`.
        
        fact_anal_exp_conv(t: numpy.ndarray, fwhm: Union[float, numpy.ndarray], tau: numpy.ndarray, base: Union[bool, NoneType] = True, irf: Union[str, NoneType] = 'g', eta: Union[float, NoneType] = None, data: Union[numpy.ndarray, NoneType] = None, eps: Union[numpy.ndarray, NoneType] = None) -> numpy.ndarray
            Estimate the best coefficiets when full width at half maximum fwhm
            and life constant tau are given
            
            When you fits your model to tscan data, you need to have
            good initial guess for not only life time of
            each component but also coefficients. To help this it solves
            linear least square problem to find best coefficients when fwhm and
            tau are given.
            
            Supported instrumental response functions are 
            
            irf
               1. 'g': gaussian distribution
               2. 'c': cauchy distribution
               3. 'pv': pseudo voigt profile
            
            Args:
               t: time
               fwhm: full width at half maximum of instrumental response function
               tau: life time for each component
               base: whether or not include baseline [default: True]
               irf: shape of instrumental
                    response function [default: g]
            
                      * 'g': normalized gaussian distribution,
                      * 'c': normalized cauchy distribution,
                      * 'pv': pseudo voigt profile :math:`(1-\eta)g + \eta c`
               eta: mixing parameter for pseudo voigt profile
                    (only needed for pseudo voigt profile,
                    default value is guessed according to
                    Journal of Applied Crystallography. 33 (6): 1311–1316.)
               data: time scan data to fit
               eps: standard error of data
            
            Returns:
             Best coefficient for given fwhm and tau, if base is set to `True` then
             size of coefficient is `num_comp + 1`, otherwise is `num_comp`.
            
            Note:
             data should not contain time range and
             the dimension of the data must be one.
        
        fact_anal_rate_eq_conv(t: numpy.ndarray, fwhm: Union[float, numpy.ndarray], eigval: numpy.ndarray, V: numpy.ndarray, c: numpy.ndarray, exclude: Union[str, NoneType] = None, irf: Union[str, NoneType] = 'g', eta: Union[float, NoneType] = None, data: Union[numpy.ndarray, NoneType] = None, eps: Union[numpy.ndarray, NoneType] = None) -> numpy.ndarray
            Estimate the best coefficiets when full width at half maximum fwhm
            and eigenvector and eigenvalue of rate equation matrix are given
            
            Supported instrumental response functions are 
            
            irf
               1. 'g': gaussian distribution
               2. 'c': cauchy distribution
               3. 'pv': pseudo voigt profile
            
            Args:
               t: time
               fwhm: full width at half maximum of instrumental response function
               eigval: eigenvalue of rate equation matrix 
               V: eigenvector of rate equation matrix 
               c: coefficient to match initial condition of rate equation
               exclude: exclude either 'first' or 'last' element or both 'first' and 'last' element.
            
               1. 'first' : exclude first element
               2. 'last' : exclude last element
               3. 'first_and_last' : exclude both first and last element  
               4. Do not specify: Do not exclude any element [default]
               irf: shape of instrumental
                    response function [default: g]
            
                      * 'g': normalized gaussian distribution,
                      * 'c': normalized cauchy distribution,
                      * 'pv': pseudo voigt profile :math:`(1-\eta)g + \eta c`
               eta: mixing parameter for pseudo voigt profile
                    (only needed for pseudo voigt profile,
                    default value is guessed according to
                    Journal of Applied Crystallography. 33 (6): 1311–1316.)
               data: time scan data to fit
               eps: standard error of data
            
            Returns:
             Best coefficient for each component.
            
            Note:
             1. eigval, V, c should be obtained from solve_model
             2. data should not contain time range and the dimension of the data must be one.
        
        gau_irf(t: Union[float, numpy.ndarray], fwhm: float) -> Union[float, numpy.ndarray]
            Compute gaussian shape irf function
            
            Args:
              t: time
              fwhm: full width at half maximum of gaussian distribution
            
            Returns:
             normalized gaussian function.
        
        model_n_comp_conv(t: numpy.ndarray, fwhm: Union[float, numpy.ndarray], tau: numpy.ndarray, c: numpy.ndarray, base: Union[bool, NoneType] = True, irf: Union[str, NoneType] = 'g', eta: Union[float, NoneType] = None) -> numpy.ndarray
            Constructs the model for the convolution of n exponential and
            instrumental response function
            Supported instrumental response function are
            
            irf
              * g: gaussian distribution
              * c: cauchy distribution
              * pv: pseudo voigt profile
            
            Args:
               t: time
               fwhm: full width at half maximum of instrumental response function
               tau: life time for each component
               c: coefficient for each component
               base: whether or not include baseline [default: True]
               irf: shape of instrumental
                    response function [default: g]
            
                      * 'g': normalized gaussian distribution,
                      * 'c': normalized cauchy distribution,
                      * 'pv': pseudo voigt profile :math:`(1-\eta)g + \eta c`
               eta: mixing parameter for pseudo voigt profile
                    (only needed for pseudo voigt profile,
                    default value is guessed according to
                    Journal of Applied Crystallography. 33 (6): 1311–1316.)
            
            Returns:
              Convolution of the sum of n exponential decays and instrumental
              response function.
            
            Note:
             1. *fwhm* For gaussian and cauchy distribution,
                only one value of fwhm is needed,
                so fwhm is assumed to be float
                However, for pseudo voigt profile,
                it needs two value of fwhm, one for gaussian part and
                the other for cauchy part.
                So, in this case,
                fwhm is assumed to be numpy.ndarray with size 2.
             2. *c* size of c is assumed to be
                num_comp+1 when base is set to true.
                Otherwise, it is assumed to be num_comp.
        
        pvoigt_irf(t: Union[float, numpy.ndarray], fwhm_G: float, fwhm_L: float, eta: float) -> Union[float, numpy.ndarray]
            Compute pseudo voight shape irf function
            (i.e. linear combination of gaussian and lorenzian function)
            
            Args:
              t: time
              fwhm_G: full width at half maximum of gaussian part
              fwhm_L: full width at half maximum of lorenzian part
              eta: mixing parameter
            Returns:
             linear combination of gaussian and lorenzian function with mixing parameter eta.
        
        rate_eq_conv(t: numpy.ndarray, fwhm: Union[float, numpy.ndarray], abs: numpy.ndarray, eigval: numpy.ndarray, V: numpy.ndarray, c: numpy.ndarray, irf: Union[str, NoneType] = 'g', eta: Union[float, NoneType] = None) -> numpy.ndarray
            Constructs signal model rate equation with
            instrumental response function
            Supported instrumental response function are
            
            irf
              * g: gaussian distribution
              * c: cauchy distribution
              * pv: pseudo voigt profile
            
            Args:
               t: time
               fwhm: full width at half maximum of instrumental response function
               abs: coefficient for each excited state
               eigval: eigenvalue of rate equation matrix 
               V: eigenvector of rate equation matrix 
               c: coefficient to match initial condition of rate equation
               irf: shape of instrumental
                    response function [default: g]
            
                      * 'g': normalized gaussian distribution,
                      * 'c': normalized cauchy distribution,
                      * 'pv': pseudo voigt profile :math:`(1-\eta)g + \eta c`
               eta: mixing parameter for pseudo voigt profile
                    (only needed for pseudo voigt profile,
                    default value is guessed according to
                    Journal of Applied Crystallography. 33 (6): 1311–1316.)
            
            Returns:
              Convolution of the solution of the rate equation and instrumental
              response function.
            
            Note:
                *fwhm* For gaussian and cauchy distribution,
                only one value of fwhm is needed,
                so fwhm is assumed to be float
                However, for pseudo voigt profile,
                it needs two value of fwhm, one for gaussian part and
                the other for cauchy part.
                So, in this case,
                fwhm is assumed to be numpy.ndarray with size 2.
        
        solve_model(equation: numpy.ndarray, y0: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Solve system of first order rate equation
            
            Args:
              equation: matrix corresponding to model
              y0: initial condition
            
            Returns:
               1. eigenvalues of equation
               2. eigenvectors for equation
               3. coefficient where y0 = Vc
    
    DATA
        __all__ = ['gau_irf', 'cauchy_irf', 'pvoigt_irf', 'exp_conv_gau', 'exp...
    
    FILE
        /home/lis1331/Documents/lecture/chem/TRXAS-pre-fit-pack/src/TRXASprefitpack/mathfun/__init__.py
    
    


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
    

