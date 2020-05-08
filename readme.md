# Declutterization
## Nathan Vance

There were several questions that this project attempts to answer, copied below for reference:

 1. What is the minimum amount of information necessary to robustly identify everyday objects within a tote?
 2. How much occlusion can be present on a single object yet still result in a positive high confidence match?
 3. What is the minimum amount of information necessary in the matching database and how small can the footprint of the database be?
 4. What is the robustness of the proposed methods to varying illumination, scale, rotation and selected properties of sensors?
 5. Is training necessary? If so, what is the least amount of training required to robustly match occluded objects?
 6. Is the approach scalable to large databases? If so, how can we test this?
 7. Could the approach extend to classes of objects or even novel objects?

This project focuses on questions 1 and 5, creating a system that uses a small set of images of individual objects to generate synthetic data, then perform classification using an SVM, thus achieving low training (as compared to neural networks). This work also attempts to answer question 4 via the experiment design.

The solution is in three parts: synthetic image generation (i.e., **clutterization**), **training**, and **inference**.

## Clutterization

Synthetic data is generated using the *clutterizer*. This system reads in images of individual objects with transparent backgrounds. Then seven objects are randomly selected, resized, rotated, and layered onto a randomly colored scene. The system uses masks of the alpha channel to keep a record of how objects occlude each other. Finally, the resulting image is saved along with a labelme formatted json file containing object masks and labels. While this process cannot change the illumination (all images of individual objects were taken with the same illumination), it does insert random scales and rotations of objects into the data. Furthermore, it is trivial to take a subset of images that were acquired using different sensors, e.g., only those taken with a mobile phone.

![Clutter Example](presentation/generated71.jpg)

Shown here is an example of generated clutter as is produced by the Clutterizer. With the labelme ground truth included, the scene is as follows:

![Clutter with Labelme](presentation/clutter.jpg)

Finally, COCO bounding boxes are used for the evaluation. These are displayed as follows:

![Clutter Ground Truth](presentation/genTruth.jpg)

This technique was developed so that the system would need only a small number of images of the individual objects for the training process.

## Training

Both training and inference involve a sliding window sized 100x100 px with a stride of 50 px. At each location a feature vector is generated containing the following information:

 * Uniform LBP histogram with number of points and radius as: (24, 8), (16, 4), (12, 2), and (8, 1).
 * Color histogram calculated over the Hue channel of the HSV colorspace and quantized into 32 bins.
 * Variance normalized to [0, 1].

These features are used to train an SVM using the RBF kernel for categorization, and `N` KNN-based Local Outlier Factor models to determine outliers, where `N` is the number of objects (in our case, 10). These models were chosen so that the system would not require extensive training.

70 synthetic images are used to train the model, and 30 are used to validate the trained model.

## Inference

Once the model has been trained, the same sliding window technique that was used in training is used in inference. For each window, the SVM is used to perform a classification, and then the confidence of the classification is as follows:

```python
confidence = max(compareHistogramData(histogram1, histogram2) for histogram2 in histograms[object]) * LocalOutlierFactors[object].confidence(histogram1)
```

Where:

 * `compareHistogramData` calculates IoU for two histograms
 * `histogram1` is the histogram calculated for the current sliding window
 * `histograms` are the histograms seen in training organized by object
 * `object` is the object that was inferred by the SVM
 * `LocalOutlierFactor[object].confidence()` calculates the percent confidence that a histogram is an inlier

Finally, the resulting confidence is compared to a threshold value to determine if it is a positive match or should be considered image background.

Once the individual sliding window positions have been categorized, adjacent regions belonging to the same object are merged and the results are written to a COCO formatted file. The inferred results for a synthetic example are shown below.

![Clutter Results](presentation/results.png)

## Experiments

To evaluate the system, the COCO results were compared with the ground truth using the official `pycocotools` python implementation. Furthermore, IoU was calculated for each individual object as well as the cumulative IoU, which was calculated by the sum of the individual intersections over the sum of the individual unions.

The first round of experiments investigated the best kernel for the SVM. The results are shown below. It was found that the RBF kernel outperforms the other evaluated kernels on the validation set, so it was used for further tests.

[SVM kernel](solution_final/plots/kernelIoU.png)

The second round of experiments was to determine the optimal LBP setting. In the plot below, the LBP setting corresponds to the LBP (number of points, radius) pair omitted, where (full) contains all LBP pairs. It was found that removing the setting with 24 points at a radius of 8 pixels produced the best results. This may be because the features at a larger radius were less meaningful due to occlusions.

[LBP setting](solution_final/plots/lbpIoU.png)

The effect of the image source was also investigated. In this experiment, we train on object images taken with one camera, either the c615 or a mobile phone, and the inference is performed on images taken with the other camera. We find that training on the mobile camera tends to result in a better IoU. Interestingly, whereas the worst validation IoU was with both training and inference with the c615, the worst test IoU was with the mobile camera for training and the c615 for evaluation. This may be because the c615 camera produces less meaningful LBP data due to more blurriness.

[Image source](solution_final/plots/sourceIoU.png)

There was further round of tests on completely unknown data, which did not contain the objects that the system was trained on. An example result is shown below.

[Test Result](presentation/unknown1.png)

## Accuracy

The accuracy metric used was IoU. The system achieved an IoU of 11.58%. The breakdown by object is as follows:

```
IOU for object brush: 0.0571806678096284
IOU for object cardboard: 0.12600002687734424
IOU for object catan: 0.26467856354656216
IOU for object router: 0.1503588970962888
IOU for object stand: 0.0
IOU for object nerf: 0.39942951524867465
IOU for object fuse: 0.004658772462787922
IOU for object pig: 0.00023087233407994738
IOU for object vader: 0.00464094600706666
IOU for object glasses: 0.0
Total IOU: 0.11583656097745643
```

In the object layouts evaluated, the stand and glasses were never observed.

In addition to IoU, the Precision and Recall reported by the COCO software was recorded. The precision for IoU=0.50 was 0.008, and recall was 0.029.

Finally, for the unknown test, the IoU was 0 because there was no valid intersection yet there was a positive union.

## Discussion

The solution, of course, isn't perfect. There were many false positives observed, and even the instances were objects were correctly identified, there was often such a low IoU that it wasn't counted by the COCO evaluation. The low IoU is likely because the system is not good at detecting objects at their edges, so it might only report a small region near the center of the objects.

Another source of error was the naive way in which detected regions were merged. It was possible, given how merging regions worked, to have significant overlap between the resulting bounding boxes for different objects. This source of error is a result of choosing to use COCO bounding boxes for reporting results. It is possible that creating some sort of hull around identified regions would have resulted in much improved IoU, even if it would have rendered evaluation using the COCO competition software invalid.

Finally, evaluation on the unknown dataset revealed many false positives. This reflects a shortcoming of the confidence reporting, which was done using a Local Outlier Factor model. Perhaps a more sophisticated model would have produced better results, at the cost of more in-depth training.

## Running the Code

The solution was implemented in Python 3.6. I have found that Python 3.7 handles parallelism in such a way that opencv chokes, but the code in demo.py is written such that it won't use parallelism in those contexts.

The project depends on opencv, sklearn, skimage, scipy, and imutils.

To run, simply navigate to the project's directory and execute `./demo.py`. The script will generate the synthetic training data, train the SVMs, infer results on a validation and a test case, and display those results.

## Consent

I have no intention of working for Amazon, and I don't think that it would help me as I work toward my PhD. However, reflective of the MIT license added to this repository, I do give my consent for you to distribute my report, source code, and any comments you would like to add to whomever you choose.
