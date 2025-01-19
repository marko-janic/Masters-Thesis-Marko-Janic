What I did:
- Implemented the ViT from torch library
- 
To discuss:
- Details about ViT
  - See https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py#L289
  - Patches are done automatically
  - Input image shape:  torch.Size([1, 3, 224, 224]), Default patch size:  16, Output shape after ViT:  torch.Size([1, 768])
  - Should I process the entire dataset once, save these latent representations and then use them during training or do it on the fly during training?
- Details about model
  - How should it look like, rn I just used some random linear classifier