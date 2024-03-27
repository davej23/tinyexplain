from tinygrad import Tensor


def cce_score(predictions: Tensor, targets: Tensor) -> Tensor:
    return predictions.sparse_categorical_crossentropy(targets)


def mse_score(predictions: Tensor, targets: Tensor) -> Tensor:
    return (predictions - targets).square().mean()


def mae_score(predictions: Tensor, targets: Tensor) -> Tensor:
    return (predictions - targets).abs().mean()


def drise_score(
    predictions: Tensor, targets: Tensor, height: int, width: int
) -> Tensor:
    """Get DRISE score for predicted bounding boxes versus ground truth bounding boxes
    Expects:
        predictions : (B, 5+NC, N)
        targets     : (B, 4+NC, N)
    Returns mean DRISE score (B, 1)

    """

    predictions = relative_to_absolute(predictions, height, width)
    targets = relative_to_absolute(targets, height, width)
    predictions = xcycwh_to_xyxy(predictions)
    targets = xcycwh_to_xyxy(targets)

    for b in range(predictions.shape[0]):  # for each in batch
        ious = iou_score(predictions[b], targets[b])
        objectness = predictions[b, 4, :].unsqueeze(0)
        classification_similarity = (
            predictions[b, 5:, :] * targets[b, 4:, :]
        ).sum() / (
            (predictions[b, 5:, :].dot(predictions[b, 5:, :].transpose())).sum()
            * (targets[b, 4:, :].dot(targets[b, 4:, :].transpose())).sum()
        )

    score = ious * objectness * classification_similarity  # score per predicted bbox
    return score.mean(1, keepdim=True)


def xcycwh_to_xyxy(xcycwh: Tensor) -> Tensor:
    assert len(xcycwh.shape) == 3

    xyxy = xcycwh[:, 0, :] - 0.5 * xcycwh[:, 2, :]
    xyxy = xyxy.cat(xcycwh[:, 1, :] - 0.5 * xcycwh[:, 3, :], dim=1)
    xyxy = xyxy.cat(xcycwh[:, 0, :] + 0.5 * xcycwh[:, 2, :], dim=1)
    xyxy = xyxy.cat(xcycwh[:, 1, :] + 0.5 * xcycwh[:, 3, :], dim=1)
    xyxy = xyxy.unsqueeze(-1)
    xyxy = xyxy.cat(xcycwh[:, 4:, :], dim=1)

    return xcycwh


def relative_to_absolute(coords: Tensor, height: int, width: int) -> Tensor:
    assert len(coords.shape) == 3
    new_coords = coords[:, 0, :] * width
    new_coords = new_coords.cat(coords[:, 1, :] * height, dim=1)
    new_coords = new_coords.cat(coords[:, 2, :] * width, dim=1)
    new_coords = new_coords.cat(coords[:, 3, :] * height, dim=1)
    new_coords = new_coords.unsqueeze(-1)
    new_coords = new_coords.cat(coords[:, 4:, :], dim=1)
    return new_coords


def iou_score(predictions: Tensor, targets: Tensor) -> Tensor:  # pylint: disable=too-many-locals
    """Get max IoU between a set of bounding box predictions and ground truth bounding boxes

    TODO: Find best way to account for different nb predicted and target bboxes
    Currently returns single IoU value for each predicted box

    """

    assert len(predictions.shape) == 2
    assert len(targets.shape) == 2
    assert predictions.shape[0] >= 4
    assert targets.shape[0] >= 4

    predictions_c = predictions.permute((1, 0))
    targets_c = targets.permute((1, 0))

    ious = []
    for i in range(predictions_c.shape[0]):  # for each predicted box
        best_iou = Tensor(0.0)
        pred = predictions_c[i]
        for j in range(targets_c.shape[0]):  # for each target box
            targ = targets_c[j]
            left = pred[0].maximum(targ[0])
            bottom = pred[1].maximum(targ[1])
            right = pred[2].maximum(targ[2])
            top = pred[3].maximum(targ[3])

            inter = (right - left).maximum(0) * (top - bottom).maximum(0)
            pred_area = (pred[2] - pred[0]) * (pred[3] - pred[1])
            targ_area = (targ[2] - targ[0]) * (targ[3] - targ[1])
            union = pred_area + targ_area - inter

            iou = inter / (union + 1e6)
            if (iou > best_iou).numpy():
                best_iou = iou

        ious.append(best_iou)

    ious_out = Tensor.stack(ious).unsqueeze(0)
    return ious_out
