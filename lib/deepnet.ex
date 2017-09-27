defmodule Deepnet do
  @moduledoc """
  Documentation for Deepnet.
  """
  
  # https://hexdocs.pm/elixir/Module.html#register_attribute/3
  Module.register_attribute __MODULE__,
    :custom_threshold_for_lib,
    accumulate: false, persist: true

  @tolerance 0.02


  defstruct [
    user_input: [[1, 0, 0],[1,0,1]],
    desired_target: [1, 1, 1]
  ]

  def start(_type, _args) do
    # List all child processes to be supervised
    children = [
      # Starts a worker by calling: Deepnet.Worker.start_link(arg)
      # {Deepnet.Worker, arg},
    ]

    # See https://hexdocs.pm/elixir/Supervisor.html
    # for other strategies and supported options
    opts = [strategy: :one_for_one, name: Deepnet.Supervisor]
    Supervisor.start_link(children, opts)
  end

  def learn() do
    Deepnet.Network.initialize_weights()
    data = %Deepnet{}
    Deepnet.Network.train(data.user_input, data.desired_target)
    learn(Map.fetch!(Deepnet.Network.get(), :error_rate), data,  0)
  end

  ###################
  #  IMPLEMENTATION #
  ###################

  defp learn(error_rate, data,  epoch) when error_rate >= @tolerance do
    Deepnet.Network.train(data.user_input, data.desired_target)
    error_rate = Map.fetch!(Deepnet.Network.get(), :error_rate)
    IO.puts("#{IO.ANSI.yellow}| EPOCH: #{epoch + 1} | ERROR RATE: #{error_rate}")
    learn(error_rate, data, epoch + 1)
  end

  defp learn(error_rate, data, epoch) when error_rate < @tolerance do
    IO.puts("""
     #{IO.ANSI.green}
     Learned to achieve target #{inspect(data.desired_target)} in #{epoch} epochs.
     Network operated with the user inputs #{inspect(data.user_input)}.
     The Final ERROR RATE for the network is #{error_rate}.
    """)
  end


end
