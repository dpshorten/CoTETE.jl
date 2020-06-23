#include("/home/david/CoTETE.jl/src/CoTETE.jl")

using Documenter, CoTETE, Test

#doctest(CoTETE)

makedocs(
    sitename="CoTETE.jl",
    authors="David Shorten",
    pages = [
        "Home" => "index.md",
        "quickStart.md",
        "public.md",
        "internals.md",
    ]
)
