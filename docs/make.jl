#include("/home/david/CoTETE.jl/src/CoTETE.jl")

using Documenter, CoTETE, Test

DocMeta.setdocmeta!(CoTETE, :DocTestSetup, :(using CoTETE; using Random: randn); recursive=true)

doctest(CoTETE)

makedocs(
    sitename="CoTETE.jl",
    authors="David Shorten",
    pages = [
        "Home" => "index.md",
        "background.md",
        "quickStart.md",
        "quickStartPython.md",
        "public.md",
        "internals.md",
    ]
)
