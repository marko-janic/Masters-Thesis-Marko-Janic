"""
ViT models and stuff: https://huggingface.co/docs/transformers/en/model_doc/vit
"""
import torch

from transformers import ViTModel, ViTFeatureExtractor, ViTImageProcessor


def get_encoded_image(image: torch.Tensor, vit_model: ViTModel, vit_image_processor: ViTImageProcessor):
    """
    :param image: torch tensor of shape batch x channels x height x width
    :param vit_model:
    :param vit_image_processor: See https://huggingface.co/docs/transformers/en/model_doc/vit
    :return:
    """

    for tensor in image:  # TODO: Remove these unecessary cheks
        if torch.isnan(tensor).all():
            print(f"This one is fucked, min: {tensor.min()}, max: {tensor.max()}, dtype: {tensor.dtype}")

    with torch.no_grad():
        inputs = vit_image_processor(images=image, return_tensors='pt', do_rescale=False)
        outputs = vit_model(pixel_values=inputs['pixel_values'], output_hidden_states=True)

    return outputs


def get_vit_model():
    """
    Gives you the vit model as well as a feature extractor for preprocessing
    Refer to https://colab.research.google.com/drive/12OmNW5dZsARio0Tzu11ParHxblOoez7u?usp=sharing#scrollTo=QLefByP4_CsW
    :return:
    """
    vit_image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    return vit_model, vit_image_processor


def get_latent_representation(self, x: torch.Tensor):
    """
    We use this model to override the normal implementation since we don't want the classification head:
    https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py#L289
    """
    # Process input
    x = self._process_input(x)
    n = x.shape[0]

    # Expand the class token
    batch_class_token = self.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)

    # Pass through encoder
    x = self.encoder(x)

    # Return the class token representation
    return x[:, 0]
