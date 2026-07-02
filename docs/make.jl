using cuBLASLt
using Documenter

DocMeta.setdocmeta!(cuBLASLt, :DocTestSetup, :(using cuBLASLt); recursive=true)

makedocs(;
    modules=[cuBLASLt],
    authors="AntonOresten <antonoresten@proton.me> and contributors",
    sitename="cuBLASLt.jl",
    format=Documenter.HTML(;
        canonical="https://docs.jool.space/cuBLASLt.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jool-space/cuBLASLt.jl",
    deploy_repo="github.com/jool-space/docs",
    devbranch="main",
    dirname="cuBLASLt.jl"
)
