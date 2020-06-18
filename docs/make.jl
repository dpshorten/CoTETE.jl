#include("/home/david/CoTETE.jl/src/CoTETE.jl")

using Documenter, CoTETE

makedocs(sitename="My Documentation")

deploydocs(
    repo = "github.com/dpshorten/CoTETE.jl.git",
    target = "build",
)
