# Home

```@meta
CurrentModule = CoTETE
```

This package allows one to estimate the Transfer Entropy (TE) between event-based time series
(such as spike trains or social media post times) in continuous time (that is, without discretising
time into bins).

It contains implementations of the estimator and local permutation scheme presented in
[Estimating Transfer Entropy in Continuous Time Between Neural Spike Trains or Other
Event-Based Data](https://doi.org/10.1101/2020.06.16.154377).

Transfer entropy is a measure of information flow between a source and a target [^1][^2]. In the context of event-based data,
it measures how much the knowing the times of previous events in the source decreases our uncertainty about the occurrence
of events in the present of the target.
Getting a clearer picture of what TE is measuring is easiest done, initially at least, in discrete time.

![Discrete TE](intro_discrete.png)

The above diagram shows the raw membrane potentials of two neurons from which spikes are extracted.
Time is then discretised into bins of width ``\Delta t`` to give us two binary time series (labelled
``X`` and ``Y`` for the source and target, respectively). The binary values these processes take on
signify the presence of an event (spike) in the bin. For each such value ``x_t`` in the target
process, we can ask what the probability of that value is given the history of the target
process ``p(x_t \, | \, \mathbf{x}<t)``. In practice, such conditional probabilities can only be estimated for histories
of limited length. As such, we use a history embedding of ``m`` bins. These are usually chosen to be the ``m`` bins
preceding the ``t``-th bin under consideration. In the above diagram, ``m`` is chosen to be 5, and so we are estimating
``p(x_t \, | \, \mathbf{x}_{t-5:t-1})``

## Contents
```@contents
Pages = ["quickStart.md", "public.md", "internals.md"]
Depth = 3
```

## Other Software
If you would like to apply TE to other data modalities, the [JIDT](https://github.com/jlizier/jidt) toolkit is highly
recommended.



[^1] Schreiber, T. (2000). Measuring information transfer. Physical review letters, 85(2), 461.
[^2] Bossomaier, T., Barnett, L., Harré, M., & Lizier, J. T. (2016). An introduction to transfer entropy. Cham: Springer International Publishing, 65-95.
