using cuBLASLt
using Documenter

DocMeta.setdocmeta!(cuBLASLt, :DocTestSetup, :(using cuBLASLt); recursive=true)

makedocs(;
    modules=[cuBLASLt],
    authors="AntonOresten <antonoresten@proton.me> and contributors",
    sitename="cuBLASLt.jl",
    format=Documenter.HTML(;
        canonical="https://jool-space.github.io/cuBLASLt.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jool-space/cuBLASLt.jl",
    devbranch="main",
)
