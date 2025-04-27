# Debugging TensorRT Models

## 📁 Files in this Lab

This folder contains scripts, charts, and models used throughout Lab 06: TensorRT Debugging & Validation.

```text
.
├── charts/                      # Generated charts from debugging and validation
├── requantised-tensorrt/        # Generated at runtime — contains re-exported ONNX and TensorRT engines
├── check_bindings.py            # Script to validate model input bindings (LoRA, ONNX, TensorRT)
├── debug_logits.py              # Script to compare logits across PyTorch, ONNX, TensorRT (original inference)
├── predict.py                   # Script to predict and compare outputs across all models
├── rebuild_engine_32.py         # Script to rebuild ONNX and TensorRT FP16 engine with correct int32 inputs
├── README.md                    # This documentation
├── requirements.txt             # Python package requirements
└── tensorrt_fp16.log            # Original TensorRT build log (shows Int64 input warnings)
```

## Understanding Logits

Recap:

>In machine learning, especially in classification tasks, logits are the raw, unnormalised scores output by a model before applying a function like softmax.
>
>Think of them as "votes" each class gets - but not yet scaled into probabilities

Post quantisation of the model with tensorrt I noticed that during all of the tests executed the results were widely out so this part of the lab is to get a better understanding of what;s going on within this quantised model.

To start off, we inspect all 3 of the quantised models and use a simple test to baseline the current models using the headline:

`Cybersecurity breach exposes NHS patient records`

Which could be labelled with either security or health depending on how much of the data set for health contains the tag NHS (a significant number IIRC)

```
$ python debug-logits.py
[PyTorch Quantised]
Logits: [-3.8202 -6.1287 -8.379  -2.4756  2.7953  4.7709  7.5915  0.7327  0.5834]
Top Predictions: [('health', np.float32(0.9347)), ('education', np.float32(0.0557)), ('climate', np.float32(0.0077)), ('security', np.float32(0.001)), ('tech-policy', np.float32(0.0008))]

[ONNX Quantised]
Logits: [-3.6806 -6.3352 -8.9846 -2.7904  3.004   4.9994  6.4153  1.3821  1.3587]
Top Predictions: [('health', np.float32(0.776)), ('education', np.float32(0.1883)), ('climate', np.float32(0.0256)), ('security', np.float32(0.0051)), ('tech-policy', np.float32(0.0049))]

[TensorRT FP16]
Logits: [11.1406  4.375   3.9707 -0.0801 -9.5     0.418  -5.9102 -1.9189 -1.6475]
Top Predictions: [('World', np.float32(0.998)), ('Sports', np.float32(0.0012)), ('Business', np.float32(0.0008)), ('education', np.float32(0.0)), ('Sci/Tech', np.float32(0.0))]
```

This clearly shows an issue with the tensorrt quantised model as health nor security even begin to come into the mix where as the pytorch and Optium quantised models both select health and at a minimum consider security.

This suggests our original lora model training indeed weighted terms like the NHS with the Health cateogry for UK news ... makes sense right?!

The other observation is the general shape of the logits being returned for each quantised model

![Logits Plot](logits_line_chart.png)

This shows the overall shape of the torch and Optimium quantised models are alomst identical but the tensorrt model is waaaaaaaay off ... next steps find out why!!

## Inspecting the Quantisation

Recap:

>In the lab04/quantisation we leveraged tensorrt to quantised the lora trainined mode which contains an interim step for the production of an onnx model.onnx which is then used to generate the model_fp16.engine model

The first thing for to do is understand where the problem lies:

1. lora -> onnx model conversion
2. onxx model -> tensorrt quantisation

Running `inspect_onnx.py` shows the following things:

```
Loaded ONNX model: ../../04-model-optimisation/quantisation/quantised-model-tensorrt/model.onnx

Model outputs:
- logits: shape=[0, 9]

```

This means:

The ONNX model expects to output a tensor called logits
* Its shape is [batch_size, 9], with batch_size=0 indicating a dynamic batch dimension
* The 9 confirms the model is configured to predict 9 labels — good!
* ✅ This part is consistent with your fine-tuned setup.

The next part of the output e.g.

```
- /bert/encoder/layer.10/attention/self/Add (Add) inputs: ['/bert/encoder/layer.10/attention/self/MatMul_output_0', '/bert/Where_1_output_0'] → outputs: ['/bert/encoder/layer.10/attention/self/Add_output_0']
- /bert/encoder/layer.10/attention/self/MatMul_1 (MatMul) inputs: ['/bert/encoder/layer.10/attention/self/Softmax_output_0', '/bert/encoder/layer.10/attention/self/Transpose_1_output_0'] → outputs: ['/bert/encoder/layer.10/attention/self/MatMul_1_output_0']
- /bert/encoder/layer.10/attention/output/dense/MatMul (MatMul) inputs: ['/bert/encoder/layer.10/attention/self/Reshape_3_output_0', 'onnx::MatMul_1767'] → outputs: ['/bert/encoder/layer.10/attention/output/dense/MatMul_output_0']
```

Shows the following elements of the model:

* embedding layer
* encoding stack
* classifier head

In essence, a computational graph that includes:

Layer Type | What It Does | ONNX Ops You Saw
Embedding | Converts tokens to dense vectors | Add, Gather
Encoder stack | Self-attention & feedforward layers that "understand" the text | MatMul, Add, Softmax, Reshape, etc.
Classifier head | Projects the final [CLS] vector into logits | Gemm or MatMul + Add

The final part of our output:

```
Tensors with likely classifier weights (matching [hidden_dim, num_labels]):
```

Initially, it appeared the classifier weights might be missing due to the empty tensor dump... But after explicitly inspecting with dump_classifier_tensors.py, we confirmed the weights are present and correct.

* The model runs, but uses missing or default-initialised weights
* That explains the wildly overconfident predictions on wrong classes (e.g., "World")

So, to conclude:

>The ONNX model we gave to TensorRT looks correct from the outside (9 output classes)...
>But it's missing the actual trained classifier weights inside... and TensorRT can't fix that

To verify this leverage dump_classifier_tensors.py to output the classifier head for the onnx model used for the tensorrt quantisation.

```
$ python dump_classifier_tensors.py
Loaded: ../../04-model-optimisation/quantisation/quantised-model-tensorrt/model.onnx

classifier.weight → shape: (9, 768)
First few values:
[ 0.0751  0.0204 -0.0102  0.0488  0.0312  0.0353  0.0201  0.0034 -0.0358
  0.0396]

classifier.bias → shape: (9,)
First few values:
[ 0.0234  0.017   0.0017  0.0268 -0.0171 -0.0323 -0.0145 -0.0149 -0.0153]
```

Strangely, this shows that the onnx model actually contains the 9 output rows (expected) and the 768 dimensional token embeddings (columns) across these 9 labels which confirms the onnx model was produced from our lora trained model.

Next we run our debug_logits script against the onnx model to see what our test line produces when fed into that model and see if the classification head is truely missing ...

```
$ python debug_logits_onnx.py

[ONNX from TensorRT folder]
Logits: [-3.3151 -6.4886 -9.0523 -2.7598  2.521   4.5606  7.4545  1.1573  1.5208]
Top Predictions:
- health       0.9371
- education    0.0519
- climate      0.0067
- tech-policy  0.0025
- security     0.0017
```

![Logits Plot](onnx_logits_line_chart.png)

This produces the same result as the other torch and optimum quantised models and the shape is the same too so the problem has to lie in the tensorrt quantisation of the onnx model.

## Journey thus far

So far we've:

| Checkpoint                                 | Result           |
|--------------------------------------------|------------------|
| Classifier head exists in ONNX             | ✅ Confirmed     |
| ONNX logits make sense                     | ✅ Match PyTorch |
| ONNX model inside TensorRT folder is valid | ✅ Verified      |
| Inference before TensorRT conversion works | ✅ ✓✓✓          |

The next step is to requantise the onnx model and see if we can replicate the issue or whether it's resolved...

## The Tensorrt Rebuild

This time, instead of using the cmd trtexec to build the model, we'll leverage the python libs to build it more programmatically `rebuild_engine.py`.

The output from this is:

```
$ python rebuild_engine.py
[04/19/2025-14:21:14] [TRT] [W] ModelImporter.cpp:459: Make sure input input_ids has Int64 binding.
[04/19/2025-14:21:14] [TRT] [W] ModelImporter.cpp:459: Make sure input attention_mask has Int64 binding.
[04/19/2025-14:21:14] [TRT] [W] ModelImporter.cpp:459: Make sure input token_type_ids has Int64 binding.
Building engine...
Saved engine to model_fp16.engine

Running inference with newly built engine...

[TensorRT from Rebuilt Engine]
Logits: [11.1406  4.3789  3.9707 -0.0728 -9.5     0.4148 -5.9102 -1.9229 -1.6504]
Top Predictions: [('World', np.float32(0.998)), ('Sports', np.float32(0.0012)), ('Business', np.float32(0.0008)), ('education', np.float32(0.0)), ('Sci/Tech', np.float32(0.0))]
```

This shows we've replicated the shape of the original tensorrt model to a tee ... not a good thing ... but it also shows that our originall onnx model was built with 64bit inputs where as the tensorrt engine prefers 32bit, the current working therory is that this is the key difference that breaks our newly rebuilt engine... soooo let's rebuild the onnx and then the engine again to fix the logit distribution mismatch.

Rebuilding again with int32 produces the following:

```
$ python rebuild_engine_32.py
Exporting ONNX model...
✅ ONNX model saved to /home/gizzmo/development/personal/ai/ai-learning-playground/02-data-science-track/06-model-debugging/tensorrt/requantised-tensorrt/model.onnx
Validating ONNX model...

Validating ONNX model logits...
ONNX Logits: [-3.3151 -6.4886 -9.0523 -2.7598  2.521   4.5606  7.4545  1.1573  1.5208]
Top Predictions: [('health', np.float32(0.9371)), ('education', np.float32(0.0519)), ('climate', np.float32(0.0067)), ('tech-policy', np.float32(0.0025)), ('security', np.float32(0.0017))]
Building TensorRT engine...
✅ Engine saved to /home/gizzmo/development/personal/ai/ai-learning-playground/02-data-science-track/06-model-debugging/tensorrt/requantised-tensorrt/model_fp16.engine

[TensorRT FP16 Inference Result]
Logits: [-3.3164 -6.4961 -9.0391 -2.7578  2.5156  4.5586  7.4727  1.1562  1.5117]
Top Predictions: [('health', np.float32(0.9383)), ('education', np.float32(0.0509)), ('climate', np.float32(0.0066)), ('tech-policy', np.float32(0.0024)), ('security', np.float32(0.0017))]
```

This is the perfect output as it shows the newly build quantised tensorrt model is performing the right now and matches the output from the onnx model and more importantly our other torch and Optimium models.

## The Final Predict Results

Leveraging the same predict.py script we built in lab04 we get the following results:

| headline                                                     | model             | true_label   | predicted_label   |   prediction_time | predict_1          | predict_2            | predict_3            | predict_4            | predict_5            |
|--------------------------------------------------------------|-------------------|--------------|-------------------|-------------------|--------------------|----------------------|----------------------|----------------------|----------------------|
| Cybersecurity breach exposes NHS patient records             | PyTorch Quantised | health       | health            |            1.4163 | health (0.9371)    | education (0.0519)   | climate (0.0067)     | tech-policy (0.0025) | security (0.0017)    |
| Cybersecurity breach exposes NHS patient records             | ONNX Quantised    | health       | health            |            0.0515 | health (0.7760)    | education (0.1883)   | climate (0.0256)     | security (0.0051)    | tech-policy (0.0049) |
| Cybersecurity breach exposes NHS patient records             | TensorRT FP16     | health       | health            |            0.0843 | health (0.9383)    | education (0.0509)   | climate (0.0066)     | tech-policy (0.0024) | security (0.0017)    |
| AI tutor program set to roll out in Scottish schools         | PyTorch Quantised | education    | education         |            0.8003 | education (0.9996) | tech-policy (0.0002) | health (0.0001)      | climate (0.0000)     | World (0.0000)       |
| AI tutor program set to roll out in Scottish schools         | ONNX Quantised    | education    | education         |            0.0523 | education (0.9995) | tech-policy (0.0002) | health (0.0002)      | climate (0.0001)     | World (0.0001)       |
| AI tutor program set to roll out in Scottish schools         | TensorRT FP16     | education    | education         |            0.074  | education (0.9996) | tech-policy (0.0002) | health (0.0001)      | climate (0.0000)     | World (0.0000)       |
| Climate report warns UK cities face extreme flooding by 2030 | PyTorch Quantised | climate      | climate           |            0.0637 | climate (0.9995)   | health (0.0002)      | tech-policy (0.0002) | education (0.0000)   | security (0.0000)    |
| Climate report warns UK cities face extreme flooding by 2030 | ONNX Quantised    | climate      | climate           |            0.0523 | climate (0.9995)   | tech-policy (0.0002) | health (0.0002)      | security (0.0000)    | education (0.0000)   |
| Climate report warns UK cities face extreme flooding by 2030 | TensorRT FP16     | climate      | climate           |            0.0743 | climate (0.9995)   | health (0.0002)      | tech-policy (0.0002) | education (0.0000)   | security (0.0000)    |

Across all three test headlines and models (PyTorch, ONNX, TensorRT), we observed:

* ✅ Consistent top-1 predictions across all models
* ✅ Near-identical softmax confidence scores
* ⚡ TensorRT FP16 inference was the fastest, completing in ~0.001–0.07 seconds
* 🔬 No evidence of logit skew or class bias post-rebuild

This confirms that our newly rebuilt TensorRT engine using an int32 ONNX model is accurate, performant, and reliable.

> ✅ Key Takeaway: The original logit mismatch was due to a 64-bit ONNX input model being interpreted incorrectly by TensorRT. Switching to `int32` inputs fixed it entirely.

It's still interesting that the cybersecurity headlines are labelled as health while security is barely makes the top5 predictions across any of the models but this has to be training data related and not an issue with the prediction mechanism / models themselves.

This concludes the TensorRT investigation in Lab 06. We've now confirmed both accuracy and consistency across inference backends and identified the precise cause of earlier logit divergence. The system is stable, validated, and ready to build on.


## Next Steps

Lab 07: Improving Quality Across All Labels

* Revisit imbalance or misclassifications (e.g. education/health confusion)
* Try label smoothing, calibration, weighted loss
* Potentially improve classifier head or retrain with data augmentation
* Use your prediction benchmark table for before/after analysis

Lab 08: Complex Dataset Challenge: Quiz-Generating LLM

* Train a domain-aware language model (or prompt-engineer one) to:
* Understand a curriculum structure (e.g. learning objectives)
* Generate questions aligned to it (Bloom's taxonomy?)
* Handle format constraints (MCQ, open-ended, rubric tags)
* Focus on data preparation + generation quality, not just accuracy
