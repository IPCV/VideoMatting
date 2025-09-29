import timm
import torch
import torch.nn as nn


class MobileViTEncoder(nn.Module):
    def __init__(self, variant="mobilevit_s", pretrained=False, weights_path=None):
        super().__init__()
        # If weights_path is provided, do not download
        self.backbone = timm.create_model(variant, features_only=True, pretrained=False)

        if weights_path:
            state_dict = torch.load(weights_path, map_location="cpu")
            self.backbone.load_state_dict(state_dict)
            print(f"Loaded backbone weights from {weights_path}")
        elif pretrained:
            # fallback to timm pretrained download
            self.backbone = timm.create_model(variant, features_only=True, pretrained=True)
            print("Loaded backbone weights from timm pretrained model")

        self.out_channels = self.backbone.feature_info.channels()

    def forward_single_frame(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward_time_series(self, x):
        B, T = x.shape[:2]
        features = self.forward_single_frame(x.flatten(0, 1))
        features = [f.unflatten(0, (B, T)) for f in features]
        return features

    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)


if __name__ == '__main__':

    # Initialize MobileViT encoder
    variant = "mobilevit_s"
    backbone = timm.create_model(variant, features_only=True, pretrained=True)

    # Save the backbone state dict to a file
    save_path = "mobilevit_s_backbone.pth"
    torch.save(backbone.state_dict(), save_path)

    print(f"Backbone weights saved to: {save_path}")
