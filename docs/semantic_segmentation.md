# Explaining Semantic Segmentation models

To explain predictions from a semantic segmentation model, the postprocessing and task functions should evaluate the similarity of the ground truth binary mask for a given target class against the model's prediction.

## Example
A Pascal VOC segmentation model is trained for 20 classes. If you wish to explain a prediction where the model has correctly segmented an aeroplane (corresponding to the 0-th channel) the postprocessing function could be

```python
lambda x: x[:, 0, :, :].unsqueeze(1)
```

The target should be of shape `(B, 1, H, W)` also, and should represent to the binary mask for the aeroplane class for the batch of images.

The resulting explanation will explain that particular class.
