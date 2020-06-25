# CoTETE.jl

*Continuous-Time Event-based Transfer Entropy*

```@meta
CurrentModule = CoTETE
```
Transfer entropy is a measure of information flow between time series. It can be used to
infer "functional" networks of statistical associations. Under certain assumptions it
can also be used to estimate underlying [causal networks](https://doi.org/10.1063/1.5025050)
from observational data.

This package allows one to estimate the Transfer Entropy (TE) between event-based time series
(such as spike trains or social media post times) in continuous time (that is, without discretising
time into bins). Transfer entropy has already been widely applied recordings of the spiking activity of neurons.
Notable applications include:



It contains implementations of the estimator and local permutation scheme presented in
[Estimating Transfer Entropy in Continuous Time Between Neural Spike Trains or Other
Event-Based Data](https://doi.org/10.1101/2020.06.16.154377).


## Contents
```@contents
Pages = ["quickStart.md", "background", "public.md", "internals.md"]
Depth = 3
```

## Acknowledgements
The estimator implemented here was developed in collaboration with my PhD supervisor, Joe Lizier,
as well as Richard Spinney.


## Other Software
If you would like to apply TE to other data modalities, the [JIDT](https://github.com/jlizier/jidt) toolkit is highly
recommended.
