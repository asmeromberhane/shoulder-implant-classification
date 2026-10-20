# Shoulder Implant Manufacturer Classification: Project Report

**Author:** Asmerom Berhane

## 1. Objective

Classify the manufacturer of a shoulder implant (Cofield, Depuy, Tornier or Zimmer) from an X-ray image. Knowing the implant model before revision surgery helps surgeons plan the procedure and choose the right tools, but records are often unavailable. The project covers exploratory analysis, pre-processing, transfer learning, class-imbalance handling and evaluation.

## 2. Dataset

The UCI *Shoulder Implant X-Ray Manufacturer Classification* dataset contains 597 de-identified X-ray images:

| Class | Images | Share |
|---|---:|---:|
| Depuy | 294 | 49% |
| Zimmer | 149 | 25% |
| Cofield | 83 | 14% |
| Tornier | 71 | 12% |

The classes are imbalanced, so accuracy alone would be misleading. **Macro F1** (the unweighted mean of per-class F1 scores) is the main metric.

## 3. Pre-processing

- Images are converted to grayscale, resized to 224×224 and scaled to [0, 1].
- The single channel is repeated to three channels, and ImageNet normalisation is applied, to match the pretrained models.
- Two stratified splits (random state 42) produce 417 training, 90 validation and 90 test images with the same class proportions.
- Class weights are computed from the training labels and used in the cross-entropy loss.

## 4. Method

The project runs several **independent experiments**. Each CNN training experiment starts from fresh ImageNet weights and none resumes a checkpoint from an earlier experiment. The HOG baseline does not use ImageNet weights, and the TTA and ensemble step loads checkpoints saved by earlier steps.

**Head-training baselines (steps 3–4).** MobileNetV2, ResNet50 and EfficientNet-B0 backbones are frozen and only a new classification head is trained. This tests how well general-purpose image features transfer to X-rays.

**Standard fine-tuning (steps 5–6).** All three backbones are trained with their final feature blocks unfrozen, for up to 60 epochs with early stopping (patience 8). Test results for this experiment are not included in this report.

**Extended ResNet50 fine-tuning (step 5b).** A fresh ImageNet ResNet50 is trained with:

- `layer4` unfrozen, with a learning rate ten times lower than the head (5e-5 vs 5e-4);
- a two-layer head with dropout;
- data augmentation: horizontal flips, ±20° rotation, translation and scaling, brightness and contrast changes;
- ReduceLROnPlateau scheduling and early stopping on validation macro F1 (patience 15, up to 80 epochs).

The checkpoint with the best validation macro F1 (epoch 32 of 47 run) was then evaluated on the test set.

**Further experiments (implemented, results not included).** HOG features with SVM and Random Forest as a classical baseline, 5-fold stratified cross-validation of ResNet50, random oversampling of minority classes, and test-time augmentation with soft-voting ensembles. Two of these need care before their results are compared with the CNNs:

- The HOG classifiers are fitted on training **plus validation** images, while the CNNs train on the training subset only.
- The TTA step uses the extended ResNet50 checkpoint when it exists, but compares against standard fine-tuned ResNet50 results.

## 5. Results (test set, 90 images)

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Majority class (always "Depuy") | 0.489 | 0.164 | n/a |
| ResNet50, head training | 0.422 | 0.413 | 0.436 |
| EfficientNet-B0, head training | 0.467 | 0.423 | 0.464 |
| MobileNetV2, head training | 0.478 | 0.441 | 0.488 |
| **ResNet50, extended fine-tuning** | **0.611** | **0.541** | **0.617** |

Per-class results for the extended ResNet50 fine-tuning model:

| Class | Precision | Recall | F1 | Test images |
|---|---:|---:|---:|---:|
| Cofield | 0.53 | 0.69 | 0.60 | 13 |
| Depuy | 0.83 | 0.75 | 0.79 | 44 |
| Tornier | 0.29 | 0.45 | 0.36 | 11 |
| Zimmer | 0.50 | 0.36 | 0.42 | 22 |

Sources: `results/baseline_comparison.csv`, `models/resnet50_extended_results.json` and the classification reports in `results/`. The majority-class row is calculated from the test-set class counts.

## 6. Discussion

- **The extended fine-tuning configuration achieved the highest macro F1.** It reached 0.541, against 0.413 for the head-trained ResNet50. Unfreezing, head design, augmentation, learning rates and training length all differ between the two experiments, so the improvement reflects the combined configuration rather than any single change.
- **The head-training baselines learned real class information.** Their accuracy is close to always guessing Depuy, but their macro F1 (0.41–0.44) is far above that strategy's 0.16, so they do recognise the minority classes.
- **The smallest class scored lowest.** Tornier has the fewest images (71 in total, 11 in the test set) and the lowest F1 (0.36). Its recall (0.45) is higher than its precision (0.29). Class imbalance and the class-weighted loss are plausible explanations for both patterns, but the included experiments do not test them: that would need, for example, a comparison with and without class weights, or the oversampling experiment's results. With only 11 Tornier test images, these per-class scores are also very uncertain.
- **The training curves show a generalisation gap.** Training accuracy rises to about 75% while validation accuracy levels off at 50–55%. Early stopping keeps the checkpoint with the best validation macro F1 rather than the last epoch.

## 7. Limitations

- **Small test set.** Only 90 images (11 Tornier). Confidence intervals for the test metrics have not yet been calculated, so differences between models should be read with caution.
- **Single split.** Cross-validation is implemented (step 8), but its results are not included, so the reported scores come from one train/validation/test split.
- **No patient grouping.** Patient-level information is not available, so images of the same patient could appear in more than one split.
- **Reproducibility.** PyTorch, NumPy and Python random seeds are not set explicitly, so retraining may give different scores.
- **BatchNorm in frozen layers.** Head training calls `model.train()`, so the running statistics of frozen BatchNorm layers still update during training. This has not been controlled for.
- This is a research and learning project, not a clinically validated tool.

## 8. Future work

- Report the standard fine-tuning, HOG, cross-validation, oversampling and ensemble experiments, after aligning the HOG training data and the TTA checkpoint comparison.
- Compare training with and without class weights to test how imbalance affects the minority classes.
- Add bootstrap confidence intervals for test metrics.
- Set all random seeds, save split file lists, and make every script import-safe.
- Decide and document a BatchNorm policy for frozen layers.
- Try larger modern backbones (ConvNeXt, Vision Transformers) and stronger augmentation for minority classes.
- Add Grad-CAM visualisations to check that the model focuses on the implant rather than image artefacts.

## 9. Conclusion

Transfer learning can identify shoulder implant manufacturers from X-rays well above a majority-class baseline. The extended ResNet50 fine-tuning model reached 61.1% accuracy and a macro F1 of 0.541 on a small, imbalanced dataset. Likely next steps are more data for the minority classes, experiments that test the effect of class weighting, and repeated evaluation through cross-validation.

## References

- UCI Machine Learning Repository (2020). *Shoulder Implant X-Ray Manufacturer Classification*. https://doi.org/10.24432/C5F893 (CC BY 4.0)
- He, K. et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
- Sandler, M. et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. *CVPR*.
- Tan, M. and Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML*.
- PyTorch documentation: [BatchNorm2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html).
