#include("/home/david/CoTETE.jl/src/CoTETE.jl")

using Documenter, CoTETE, Test

doctest(CoTETE)

makedocs(
    sitename="CoTETE.jl",
    authors="David Shorten",
    pages = [
        "index.md",
        "public.md",
        "internals.md",
    ]
)
