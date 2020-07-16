# Quick Start (Python)

The first bit of these instructions is (mostly) a copy of the Julia instructions (for installation), so if
you have already followed them then skip down to the first use of pip. Note, however, the one change
of the addition of the installation of the Julia library `PyCall.jl`

[Install Julia](https://julialang.org/downloads/)

Clone this repo (make sure to include the --recurse-submodules flag so that the modified nearest neighbours
package gets included).

```console
david@home:~$ git clone --recurse-submodules https://github.com/dpshorten/CoTETE.jl.git
```

make sure that CoTETE.jl/src/ is on your JULIA\_LOAD\_PATH. eg:

```console
david@home:~$ export JULIA_LOAD_PATH=:/home/david/CoTETE.jl/src/
```

Fire up the Julia REPL

```console
david@home:~$ julia
```
You will need to add three prerequisite packages.

!!! tip "Tip for new Julia users"
    The Julia REPL has a nifty feature called *prompt pasting*, which means that it
    will automatically remove the `julia>` prompt when you paste. You can, therefore, just copy and paste the entire block
    below without worrying about these prompts.

```julia
julia> import Pkg
julia> Pkg.add("Distances")
julia> Pkg.add("StaticArrays")
julia> Pkg.add("SpecialFunctions")
julia> Pkg.add("Parameters")
julia> Pkg.add("PyCall")
```

Install PyJulia via pip

```console
david@home:~$ pip3 install julia
```

We can now estimate the TE between two uncoupled homogeneous Poisson processes (as covered in section II A of [1]). We the run the following code:
```python
from julia.api import Julia
jl = Julia(compiled_modules=False)
jl.eval("using CoTETE")
from julia import CoTETE
params = CoTETE.CoTETEParameters(l_x = 1, l_y = 1)
import numpy as np
target = 1e3*np.random.rand(1000); target = np.sort(target);
source = 1e3*np.random.rand(1000); source = np.sort(source);
CoTETE.estimate_TE_from_event_times(params, target, source)
```

The answer should be close to 0.

All the other CoTETE.jl functions documented elsewhere in these docs can be used in a similar manner. Just import the CoTETE package as above and call them as you would in Julia.
