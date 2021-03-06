defmodule Deepnet.Mixfile do
  use Mix.Project

  def project do
    [
      app: :deepnet,
      version: "0.1.0",
      elixir: "~> 1.5",
      start_permanent: Mix.env == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      applications: [
        :logger,
        :sfmt,
        :numerix,
        :matrix
      ],
      extra_applications: [:logger],
      mod: {Deepnet.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      # {:dep_from_hexpm, "~> 0.3.0"},
      # {:dep_from_git, git: "https://github.com/elixir-lang/my_dep.git", tag: "0.1.0"},
      {:matrix,   "~> 0.3.2"  },
      {:numerix,  "~> 0.4.2"  },
      {:sfmt,     "~> 0.13.0" }
    ]
  end
end
