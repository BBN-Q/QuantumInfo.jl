using Documenter, QuantumInfo

makedocs(
	sitename = "QuantumInfo",
	modules  = [QuantumInfo],
	format   = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
	pages    = [
		"index.md"
	]
)

deploydocs(;
	repo        = "github.com/BBN-Q/QuantumInfo.jl.git",
	push_review = true
)
