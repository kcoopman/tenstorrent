This is a VIT attempt with Tentorrent Tools.

To run:

$ source ~/Ollama/venv_ollama/bin/activate
$ python3 vit.py  

```
This is the name of the program: vit.py
Number of elements including the name of the program: 1
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

Model Accuracy: 90 %
```
