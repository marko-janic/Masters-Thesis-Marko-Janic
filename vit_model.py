"""
ViT models and stuff: https://huggingface.co/docs/transformers/en/model_doc/vit
"""
import torch
import math

from transformers import ViTModel, ViTImageProcessor


def get_encoded_image(image: torch.Tensor, vit_model: ViTModel, vit_image_processor: ViTImageProcessor,
                      device, num_patch_embeddings: int):
    """
    :param image: torch tensor of shape batch x channels x height x width
    :param vit_model:
    :param vit_image_processor: See https://huggingface.co/docs/transformers/en/model_doc/vit
    :param device: The device to use
    :return:
    """

    inputs = vit_image_processor(images=image, return_tensors='pt', do_rescale=False).to(device)
    outputs = vit_model(pixel_values=inputs['pixel_values'], output_hidden_states=True)

    # 1: here because we don't need the class token
    latent_micrographs = outputs['last_hidden_state'].to(device)[:, 1:, :]
    # Right shape for model, we permute the hidden dimension to the second place
    latent_micrographs = latent_micrographs.permute(0, 2, 1)
    reshaping_value = int(math.sqrt(num_patch_embeddings))
    latent_micrographs = latent_micrographs.reshape(
        latent_micrographs.size(0), latent_micrographs.size(1), reshaping_value, reshaping_value)

    return latent_micrographs


def get_vit_model(model_name: str):
    """
    Gives you the vit model as well as a feature extractor for preprocessing
    Refer to https://colab.research.google.com/drive/12OmNW5dZsARio0Tzu11ParHxblOoez7u?usp=sharing#scrollTo=QLefByP4_CsW
    :param model_name: Which model to use
    :return:
    """
    print(f"Loading ViT model: {model_name}")
    vit_image_processor = ViTImageProcessor.from_pretrained(model_name)
    vit_model = ViTModel.from_pretrained(model_name)

    return vit_model, vit_image_processor
