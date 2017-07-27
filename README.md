1. I dont't have the PRE-TRAINED PARAMETER, so in `paddle_train.py` **pretrained embedding is not loaded. Static parameter must be intialized by pre-training**.
    - `load_pretrained_parameters` in [this line](https://github.com/lcy-seso/paddle_experiment/blob/master/paddle_train.py#L40) should be implemented first, otherwise the embedding should be learned from data.
    - please check this: [How to load pre-trained parameters that are trained by Paddle](https://github.com/lcy-seso/paddle_experiment/blob/master/notes/notes_about_PaddlePaddle.md#5-how-to-load-pre-trained-parameters-that-are-trained-by-paddle).

2. the lastes develop branch of Paddle is required (I fixed some bug.)
3. this fix is required to run `paddle_infer.py`: https://github.com/PaddlePaddle/Paddle/compare/develop...lcy-seso:fix_infer_dim （I will merge it soon after more test.）

---
* Below is example of inferring outputs:
  - one line is a word in document.
  - one line has four columns separated by TAB
    - 1. word
    - 2. sentence predictions: groudtruth [predicetd probability]
    - 3. start predictions: groudtruth [predicetd probability]
    - 4. end predictions: groudtruth [predicetd probability]

```text
Super   0[0.9902 0.0098]        0[0.9879 0.0121]        0[0.9866 0.0134]
Bowl    1[0.9961 0.0039]        0[0.9954 0.0046]        0[0.9951 0.0049]
50      0[0.9968 0.0032]        0[0.9962 0.0038]        0[0.9960 0.0040]
was     0[0.9969 0.0031]        0[0.9963 0.0037]        0[0.9961 0.0039]
an      0[0.9969 0.0031]        0[0.9963 0.0037]        0[0.9961 0.0039]
...
```
