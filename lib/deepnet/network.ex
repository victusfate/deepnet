defmodule Deepnet.Network do
  
  defstruct [
    learning_rate: nil,
    input_nodes: nil,
    hidden_nodes: nil,
    output_nodes: nil,
    weights: nil,
    error_rate: nil,
    target: nil
  ]

  @doc """
  Creates the neural network. This function is called at startup by the project
  supervisor.

  # PARAMETERS
  - `input_nodes`: Integer.t()  number of inputs
  - `hidden_nodes`: Integer.t() number of neurons in the hidden layer
  - `output_nodes`: Integer.t() number of neurons in the output layer
  - `learning_rate`: Float.t()  rate at which the network learns. Default set to 1.0
  """
  def create(
    input_nodes,
    hidden_nodes,
    output_nodes,
    learning_rate \\ 1.0) when is_number(input_nodes)
                          and  is_number(hidden_nodes)
                          and  is_number(output_nodes)
                          and  is_float(learning_rate) do
    
    Agent.start_link(fn() -> 
      %Deepnet.Network{
        learning_rate:  learning_rate,
        input_nodes:    input_nodes,
        hidden_nodes:   hidden_nodes,
        output_nodes:   output_nodes,
      }
    end, [name: __MODULE__])                            
  end

  def get do
    Agent.get(__MODULE__, &(&1))
  end  

  def initialize_weights() do 
    :sfmt.seed(:os.timestamp)
    network = get()
    input_weights = Matrix.rand(network.input_nodes, network.hidden_nodes)

    weights = 
      Matrix.sub(input_weights,
        Matrix.new(network.input_nodes, network.hidden_nodes, 0.5))

    Agent.get_and_update(__MODULE__, fn(map) ->
      {map, Map.put(map, :weights, weights)}
    end)
  end

  defp calculate_errors(final_outputs, inputs) do 
    network_error =
      Map.fetch!(get(), :target)
      |> List.flatten
      |> Numerix.Distance.mse(List.flatten(final_outputs))

    Agent.update(__MODULE__,
      fn(map) -> Map.put(map, :error_rate, network_error) end)
    adjust_weights(final_outputs, inputs)
  end

  defp adjust_weights(output,input) do
    network = get()

    delta =
      Matrix.sub(output, network.target)
      |> List.flatten
      |> Numerix.LinearAlgebra.dot_product(List.flatten(input))

    gradient = pmap(List.flatten(output), fn(calc) -> [calc * delta * network.learning_rate] end)
    new_weights = Matrix.sub(network.weights, gradient)

    Agent.update(__MODULE__, fn(map) -> Map.put(map, :weights, new_weights) end)
  end

  defp calculate(inputs, weights) do
    pmap(Enum.zip(inputs, weights), fn(tuple) -> 
      {input, weight} = {elem(tuple,0), elem(tuple,1)}
      Numerix.LinearAlgebra.dot_product(input, weight)
      |> Numerix.Special.logistic
      |> List.wrap
    end)
  end

  defp feed_forward(input_list) do
    network = get()
    calculate(input_list, network.weights) |> feed_forward(network.weights, input_list)
  end

  defp feed_forward(outputs, old_weights, inputs) do
    final_outputs = calculate(outputs, old_weights)
    calculate_errors(final_outputs, inputs)
  end

  defp pmap(data, calc_function) do
    data
    |> Enum.map(&(Task.async(fn -> calc_function.(&1) end)))
    |> Enum.map(&Task.await/1)
  end

  def train(inputs, target) when is_list(inputs) and is_list(target) do
    {input_list, target_list} = {pmap(inputs, &List.wrap/1), pmap(target, &List.wrap/1)}
    Agent.get_and_update(__MODULE__, fn(map) -> 
      {map, Map.put(map, :target, target_list)}
    end)
    feed_forward(input_list)
  end

end












