using Documenter, QuantumInfo

makedocs(
	sitename = "QuantumInfo",
	modules  = [QuantumInfo],
	format   = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
	pages    = [
		"index.md"
	]
)

