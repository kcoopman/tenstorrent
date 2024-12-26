This is a VIT attempt with Tentorrent Tools.

To run:

$ source ~/Ollama/venv_ollama/bin/activate


```
$ python3 vit.py train
This is the name of the program: vit.py
Number of elements including the name of the program: 2
Using device:  cuda (NVIDIA GeForce GTX TITAN X)
Using device:  cuda (NVIDIA GeForce GTX TITAN X)
Model's state_dict:
patch_embedding.linear_project.weight 	 torch.Size([9, 1, 16, 16])
patch_embedding.linear_project.bias 	 torch.Size([9])
positional_encoding.cls_token 	 torch.Size([1, 1, 9])
positional_encoding.pe 	 torch.Size([1, 5, 9])
transformer_encoder.0.ln1.weight 	 torch.Size([9])
transformer_encoder.0.ln1.bias 	 torch.Size([9])
transformer_encoder.0.mha.W_o.weight 	 torch.Size([9, 9])
transformer_encoder.0.mha.W_o.bias 	 torch.Size([9])
transformer_encoder.0.mha.heads.0.query.weight 	 torch.Size([3, 9])
transformer_encoder.0.mha.heads.0.query.bias 	 torch.Size([3])
transformer_encoder.0.mha.heads.0.key.weight 	 torch.Size([3, 9])
transformer_encoder.0.mha.heads.0.key.bias 	 torch.Size([3])
transformer_encoder.0.mha.heads.0.value.weight 	 torch.Size([3, 9])
transformer_encoder.0.mha.heads.0.value.bias 	 torch.Size([3])
transformer_encoder.0.mha.heads.1.query.weight 	 torch.Size([3, 9])
transformer_encoder.0.mha.heads.1.query.bias 	 torch.Size([3])
transformer_encoder.0.mha.heads.1.key.weight 	 torch.Size([3, 9])
transformer_encoder.0.mha.heads.1.key.bias 	 torch.Size([3])
transformer_encoder.0.mha.heads.1.value.weight 	 torch.Size([3, 9])
transformer_encoder.0.mha.heads.1.value.bias 	 torch.Size([3])
transformer_encoder.0.mha.heads.2.query.weight 	 torch.Size([3, 9])
transformer_encoder.0.mha.heads.2.query.bias 	 torch.Size([3])
transformer_encoder.0.mha.heads.2.key.weight 	 torch.Size([3, 9])
transformer_encoder.0.mha.heads.2.key.bias 	 torch.Size([3])
transformer_encoder.0.mha.heads.2.value.weight 	 torch.Size([3, 9])
transformer_encoder.0.mha.heads.2.value.bias 	 torch.Size([3])
transformer_encoder.0.ln2.weight 	 torch.Size([9])
transformer_encoder.0.ln2.bias 	 torch.Size([9])
transformer_encoder.0.mlp.0.weight 	 torch.Size([36, 9])
transformer_encoder.0.mlp.0.bias 	 torch.Size([36])
transformer_encoder.0.mlp.2.weight 	 torch.Size([9, 36])
transformer_encoder.0.mlp.2.bias 	 torch.Size([9])
transformer_encoder.1.ln1.weight 	 torch.Size([9])
transformer_encoder.1.ln1.bias 	 torch.Size([9])
transformer_encoder.1.mha.W_o.weight 	 torch.Size([9, 9])
transformer_encoder.1.mha.W_o.bias 	 torch.Size([9])
transformer_encoder.1.mha.heads.0.query.weight 	 torch.Size([3, 9])
transformer_encoder.1.mha.heads.0.query.bias 	 torch.Size([3])
transformer_encoder.1.mha.heads.0.key.weight 	 torch.Size([3, 9])
transformer_encoder.1.mha.heads.0.key.bias 	 torch.Size([3])
transformer_encoder.1.mha.heads.0.value.weight 	 torch.Size([3, 9])
transformer_encoder.1.mha.heads.0.value.bias 	 torch.Size([3])
transformer_encoder.1.mha.heads.1.query.weight 	 torch.Size([3, 9])
transformer_encoder.1.mha.heads.1.query.bias 	 torch.Size([3])
transformer_encoder.1.mha.heads.1.key.weight 	 torch.Size([3, 9])
transformer_encoder.1.mha.heads.1.key.bias 	 torch.Size([3])
transformer_encoder.1.mha.heads.1.value.weight 	 torch.Size([3, 9])
transformer_encoder.1.mha.heads.1.value.bias 	 torch.Size([3])
transformer_encoder.1.mha.heads.2.query.weight 	 torch.Size([3, 9])
transformer_encoder.1.mha.heads.2.query.bias 	 torch.Size([3])
transformer_encoder.1.mha.heads.2.key.weight 	 torch.Size([3, 9])
transformer_encoder.1.mha.heads.2.key.bias 	 torch.Size([3])
transformer_encoder.1.mha.heads.2.value.weight 	 torch.Size([3, 9])
transformer_encoder.1.mha.heads.2.value.bias 	 torch.Size([3])
transformer_encoder.1.ln2.weight 	 torch.Size([9])
transformer_encoder.1.ln2.bias 	 torch.Size([9])
transformer_encoder.1.mlp.0.weight 	 torch.Size([36, 9])
transformer_encoder.1.mlp.0.bias 	 torch.Size([36])
transformer_encoder.1.mlp.2.weight 	 torch.Size([9, 36])
transformer_encoder.1.mlp.2.bias 	 torch.Size([9])
transformer_encoder.2.ln1.weight 	 torch.Size([9])
transformer_encoder.2.ln1.bias 	 torch.Size([9])
transformer_encoder.2.mha.W_o.weight 	 torch.Size([9, 9])
transformer_encoder.2.mha.W_o.bias 	 torch.Size([9])
transformer_encoder.2.mha.heads.0.query.weight 	 torch.Size([3, 9])
transformer_encoder.2.mha.heads.0.query.bias 	 torch.Size([3])
transformer_encoder.2.mha.heads.0.key.weight 	 torch.Size([3, 9])
transformer_encoder.2.mha.heads.0.key.bias 	 torch.Size([3])
transformer_encoder.2.mha.heads.0.value.weight 	 torch.Size([3, 9])
transformer_encoder.2.mha.heads.0.value.bias 	 torch.Size([3])
transformer_encoder.2.mha.heads.1.query.weight 	 torch.Size([3, 9])
transformer_encoder.2.mha.heads.1.query.bias 	 torch.Size([3])
transformer_encoder.2.mha.heads.1.key.weight 	 torch.Size([3, 9])
transformer_encoder.2.mha.heads.1.key.bias 	 torch.Size([3])
transformer_encoder.2.mha.heads.1.value.weight 	 torch.Size([3, 9])
transformer_encoder.2.mha.heads.1.value.bias 	 torch.Size([3])
transformer_encoder.2.mha.heads.2.query.weight 	 torch.Size([3, 9])
transformer_encoder.2.mha.heads.2.query.bias 	 torch.Size([3])
transformer_encoder.2.mha.heads.2.key.weight 	 torch.Size([3, 9])
transformer_encoder.2.mha.heads.2.key.bias 	 torch.Size([3])
transformer_encoder.2.mha.heads.2.value.weight 	 torch.Size([3, 9])
transformer_encoder.2.mha.heads.2.value.bias 	 torch.Size([3])
transformer_encoder.2.ln2.weight 	 torch.Size([9])
transformer_encoder.2.ln2.bias 	 torch.Size([9])
transformer_encoder.2.mlp.0.weight 	 torch.Size([36, 9])
transformer_encoder.2.mlp.0.bias 	 torch.Size([36])
transformer_encoder.2.mlp.2.weight 	 torch.Size([9, 36])
transformer_encoder.2.mlp.2.bias 	 torch.Size([9])
classifier.0.weight 	 torch.Size([10, 9])
classifier.0.bias 	 torch.Size([10])
Optimizer's state_dict:
state 	 {}
param_groups 	 [{'lr': 0.005, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0, 'amsgrad': False, 'maximize': False, 'foreach': None, 'capturable': False, 'differentiable': False, 'fused': None, 'params': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88]}]
Epoch 1/3 loss: 1.668
Epoch 2/3 loss: 1.570
Epoch 3/3 loss: 1.554
Save to ./vit.pt
```


```
$ python3 vit.py test
This is the name of the program: vit.py
Number of elements including the name of the program: 2
Using device:  cuda (NVIDIA GeForce GTX TITAN X)
Testing ----------

Model Accuracy: 92 %
```


