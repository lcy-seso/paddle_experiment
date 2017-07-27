## 1. About `paddle.layer.memory` in PaddlePaddle
  - Every layer in PaddlePaddle has a unique name if the user does not name a layer explicitly, it will be named automatically.
  - Memory in PaddlePaddle is very like reference parameters C++. Itself is not a real layer, but points to a layer and retrieve whose output in the previous time step.
  - You have to explicitly give a name to the layer `paddle.layer.memory` points to, because `paddle.layer.memory` needs a layer's name to decide to retrieve which layer's output in previous time step.
  - In `paddle.layer.memory`, the name specified by the `name` parameter is not the name of the defnied `memory` layer, but the name of the real layer `memory` points to.


## 2. About `paddle.layer.dropout` and `paddle.attr.ExtraLayerAttribute(drop_rate=x)`
  - I think for most layers, a better way to use dropout is to set the droprate in `layer_attr` (every layer in `paddle.layer` has this attribute) by using `paddle.attr.ExtraLayerAttribute(drop_rate=0.5)` as below:
    ```python
    fc = paddle.layer.fc(
                input=input,
                bias_attr=paddle.attr.Param(initial_std=0.),
                param_attr=paddle.attr.Param(initial_std=5e-4),
                layer_attr=paddle.attr.ExtraLayerAttribute(drop_rate=0.5),
    ```
  - dropout in PaddlePaddle is actually implemented in activiation function, it is not a layer.
  - But `paddle.layer.lstmemory`, `paddle.layer.grumemory`, `paddle.layer.recurrent` are different, these layers does not activiate the output by calling the general activiation process, but implement the activiation process themselves. As a results, drop rate cannot be directly set in these layer. (I make a new PR to fix this. If it is set, PaddlePaddle will raise an error.)

  - `paddle.layer.dropout` actually defines a `paddle.layer.add_to` layer and set the droprate in this layer. This is a little waste of memory, becuase output value to drop is copied again and PaddlePaddle will not release the memory to improve the time efficiency. But if you want to drop a recurrent layer's output, you have to use `paddle.layer.dropout`.

## 3. About different recurrent layers in PaddlePaddle.
  - I found you both use `paddle.layer.recurrent_group` and `paddle.networks.simple_lstm` they are all recurrent units in PaddlePaddle. I write some notes about the differences about them.
  - If you do not need explicitly access to the intermedia values in a recurent unit (hidden states, input-to-hidden mapping, memory cells and so on), I recommend using `paddle.networks.simple_lstm` or `paddle.layer.lstmemory`.
  - `recurrent_group` is usefull in attention model, or NTM.

  - In PaddlePaddle we have (here I take LSTM for example, GRU is the same):

      1. `paddle.layer.lstmemory`
      2. `paddle.networks.simple_lstm`
      3. `paddle.networks.lstmemory_group`
      4. `paddle.networks.lstmemory_unit`
      5. `paddle.networks.bidirectional_lstm`


  * The above recurrent layers can be categorised into two type:
      1. recurrent layer implemented by recurrent_group:
         - you can access to any intermedia values (hidden states, input-to-hidden mapping, memory cells and so on) a recurent unit computes during one time step.
         - the above 3
      2. recurrent layer as a whole:
          - you can only access to its outputs.
          - the above 1 ~ 2, 5
      3. `paddle.networks.lstmemory_unit` is not a recurrent unit, it defines the computation a LSTM unit performed in one time step.
          - It only canbe used as the step function in `recurrent_group`.
          - the above 4

  * The second type (recurrent layer as a whole) is more computation efficient, because `recurrent_group` is made up of many basic layers (including add, element-wise multiplicationsm, matrix multiplication and so on), while recurrent layer as a whole is carefully optimized in both CPU and GPU.

  * But all recurrent layers(simple rnn, GRU, LSTM) in PaddlePaddle leave the input-to-hidden mapping outside the recurrent layer to make a larger matrix for LSTM and GRU to accerlate the computation speed.

    - This is the diffences between `paddle.layer.lstmmemory` and `paddle.network.simple_lstm`. Specifically:
      - `paddle.layer.lstmmemory` is not the LSTM in textbook, it is a LSTM unit without input-to-hidden projection.
      - `paddle.network.simple_lstm` is a wrapper which just adds the input-to-hidden projection into `paddle.layer.lstmmemory`. It is the LSTM in textbook.
    - `paddle.layer.lstmmemory` and `paddle.network.simple_lstm` in PaddlePaddle is **LSTM with peephole connection**. Be carefull to this, it may has more parameters than your previous model.

## 4. A bug of current PaddelPaddle: **optimizer must be defined before the network topology.**
  - the detail information can be found in this issue: https://github.com/PaddlePaddle/Paddle/issues/2621
  - very sorry, and **be careful to this**, we will fix this.
