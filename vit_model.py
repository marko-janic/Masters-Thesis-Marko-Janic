"""
ViT models and stuff: https://huggingface.co/docs/transformers/en/model_doc/vit
"""
import torch

from transformers import ViTModel, ViTImageProcessor


def get_encoded_image(image: torch.Tensor, vit_model: ViTModel, vit_image_processor: ViTImageProcessor,
                      device):
    """
    :param image: torch tensor of shape batch x channels x height x width
    :param vit_model:
    :param vit_image_processor: See https://huggingface.co/docs/transformers/en/model_doc/vit
    :param device: The device to use
    :return:
    """

    inputs = vit_image_processor(images=image, return_tensors='pt', do_rescale=False).to(device)
    outputs = vit_model(pixel_values=inputs['pixel_values'], output_hidden_states=True)

    return outputs


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
