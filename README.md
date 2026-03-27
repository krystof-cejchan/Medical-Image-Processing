# Medical-Image-Processing

# Segmentation using traditional methods

**Author: Kryštof Čejchan**
## Introduction
In this document, I describe what the first task was about, how I solved this task, and what the results are.

## Project description

The first project deals with the use of monadic and morphological operations on bitmap images, their binarization, and interpretation.

<img alt="Tv21.tiff" height="510" src="Segmentation%20using%20traditional%20methods/pics/popisprojektu.png" width="685"/>

The input is a grey-scale bitmap image.

1. The image undergoes pre-processing, which removes unwanted objects from the image and highlights the desired objects in the image.
2. The image is binarized, with the desired objects in white, the rest in black
3. The parts that are white in the binary image are cut out from the original gray-scale image
4. These parts go through another wave of operations, with the end result being a binary image again; but this time it is a given "cell"

In the coming chapters I will describe the exact operations that were tested or used.
## Program description
This chapter deals with how to run the given Python script and how to use the program.

**Warning: the program has a graphical user interface, so you need to have the tkinter library installed (https://docs.python.org/3/library/tkinter.html)**
### Running the program
The program is started using the `main.py` file, which opens the GUI.

![img.png](Segmentation%20using%20traditional%20methods/pics/run_script.png)

Now the program is "empty" and its functionality cannot be used.
### Using the program
First we need to select the input image using `File>Open...`.
Then we can use the operations in the top menu.

- **File**: used to open and save images, and exit the program.
- **Adjust**: contains primarily operations from the first lecture; i.e. monadic operations
- **Process**: contains primarily operations from the second lecture; i.e. morphological operations
- **Filters&Blurs**: contains primarily operations from the third lecture; i.e. filters and image blurring
- **Histograms**: used to display histograms and CDF (cumulative distribution function), and also offers operations related to histogram (equalization and CLAHE)
- **View**: used to delete performed operations on the image
- **Pipelines**: contains pre-created sequences of operations to meet the task objective

![img.png](Segmentation%20using%20traditional%20methods/pics/ukazka_programu.png)

### Orientation in the source code
This subsection deals with the project structure from the point of view of the source code and other necessary directories.

- `./main.py` is a startup script, it calls the `ImageEditorApp` class, which starts the GUI
- `./image_editor_ui.py` contains a class with GUI components (buttons, text fields, etc.), this class also stores the input and edited image. Unfortunately, the file is very large because the GUI contains many components.
- `./image_operations/*.py` contains implemented operations that have been tested or used
- `monadic_operations.py` contains monadic operations, e.g. negation, quantization, or brightness, gamma, contrast adjustment
- `advanced_operations.py` contains "more advanced" operations such as CLAHE, erosion, dilatation, opening, closing, contour finding, or even splitting connected "cells"
- `filters.py` contains filtering and blurring operations, e.g.: gaussian blur, difference of gaussians, laplacian of gaussian...
- `pipelines.py` contains pre-created sequences of operations that will be run in order to fulfill the task's goal
- `./utils/*.py` contains helper methods, e.g. for converting between Float ⟨0.1⟩ -> uint8 ⟨0.255⟩, and contains functions for calculating histogram and CDF

## Solution to the task
Thanks to the GUI implementation, I had the opportunity to play with different operations, test their parameters, and see the result immediately.
When solving the problem, I implemented several pipelines, but in the end I kept only one that had the best results.
Each pipeline contains two parts:
1) operations on the input image
2) operations on the ROI (rectangle of interest)
### Pipeline 1
This pipeline ultimately came out as the best.
```python
def pipeline1(img: np.ndarray) -> np.ndarray:
im = img.copy()
im = adjust_brightness(im, brightness=0.4) #pre-processing
im = adjust_gamma(im, gamma=0.1)
im = clahe(im, clip_limit=3, tile_grid=16)
for _ in range(30):
im = median_filter(im, ksize=3)
im = negate(im) # binarization
im = otsu(im)
im = opening(im, 10) # post-processing
return im
```
The function first applies two basic operations: it increases the brightness and reduces the gamma. These two operations, although simple, effectively
ensure that only the most prominent objects remain in the image, i.e., it essentially removes unwanted background noise; see. image below

![img.png](Segmentation%20using%20traditional%20methods/pics/brightgamma1.png)

After using CLAHE and median filters, another small noise is smoothed. (Median filter is in a loop because OpenCV does not allow setting kernel size larger than 4)
Next is negation and OTSU, which provide binarization.

![img.png](Segmentation%20using%20traditional%20methods/pics/clahe_otsu1.png)

5 000 / 5 000
To remove other unwanted objects, opening is used, which "cuts" these objects so much that they are not recorded when searching for contours.

![img.png](Segmentation%20using%20traditional%20methods/pics/opening1.png)

The result of this is a set of rectangular gray-scale images of cells. These images go through the next phase: binarization and searching for nuclei in the cell.
```python
def roi_binarization1(img: np.ndarray) -> np.ndarray:
im = img.copy()
im = erosion(im, erosion_size=1, interactions_no=3)
im = gaussian_blur(im, ksize=5, sigma=0)
im = negate(im)
im = adjust_gamma(im, gamma=0.16)
im = otsu(im)
im = opening(im, size=4, shape=cv.MORPH_RECT, iterations_no=1)
return im
```
First, erosion is run, which highlights and enlarges the black parts of the image; in our case, these are the cell nuclei.

![img.png](Segmentation%20using%20traditional%20methods/pics/cellerso1.png)

Next, a gaussian filter is applied, which subtly blurs the black dots that appeared after erosion.
After negating the image, the gamma is reduced (if the image were not negated, the gamma could be increased for a similar effect). The result of the gamma adjustment is an image with white cells highlighted.

![img.png](Segmentation%20using%20traditional%20methods/pics/cellgamma1.png)

After using OTSU, the image is complete (at least in this case); in general, this may not be the case, so we will use opening. Opening is used here with the shape `cv.MORPH_RECT`, because it has been found that a rectangular shape is better at separating nuclei that are close together than an ellipse.

![img.png](Segmentation%20using%20traditional%20methods/pics/cellotsu.png)
![img_1.png](Segmentation%20using%20traditional%20methods/pics/cell_open1.png)

Since in the vast majority of images the middle two kernels merge into one, a method was chosen that inserts a black line into the middle kernel. See the code below.
```python
def split_middle_cell(img, line_thickness=2):
_, contours = find_contours(img, 30) #finds kernels
h, w = img.shape[:2]
cx, cy = w / 2.0, h / 2.0

best = None
best_abs_d = float("inf")

for c in contours: # using pointPolygonTest, finds the contour that is closest to the center of the image
d = cv.pointPolygonTest(c, (cx, cy), True)
if abs(d) < best_abs_d:
best_abs_d = abs(d)
best = c

middle_contour = best

x, y, bw, bh = cv.boundingRect(middle_contour)
x0 = max(0, x)
y0 = max(0, y)
x1 = min(w, x + bw)
y1 = min(h, y + bh)

y_line = int((y0 + y1) / 2)
cv.line(img, (x0, y_line), (x1, y_line), (0, 0, 0), thickness=line_thickness) #"paints" a black line in the middle

return img
```

The cell contours must still be numbered throughout the pipeline. The entire pipeline and its result then look like this:
```python
def apply_pipeline(self, pipeline_number: int, contour_rect_size:int, binarization_method:int, contour_bin_cell_size:int=150):
fun = getattr(pipelines, f'pipeline{pipeline_number}')# look up the pipeline by function name
self.edited = fun(self.original) # run the pipeline on the input image; save the result to the edited image
self.extract_rects_current(contour_rect_size,10) #find "cells" using rectangular contours
self.last_rois_binary.clear()
for i in range(len(self.last_rois) - 1, -1, -1):#for each cell found...
roi = self.last_rois[i]
fun_bin = getattr(pipelines, f'roi_binarization{binarization_method}')
binarized_roi = fun_bin(roi) #...run cell binarization
binarized_roi = split_middle_cell(binarized_roi) #...split the middle cell
contours = self.find_contours(binarized_roi, contour_bin_cell_size) #..find "cell nuclei" using contours
#remove everything that is not in the contours
mask = np.zeros(binarized_roi.shape[:2], dtype=np.uint8)
cv.drawContours(mask, contours, -1, 1, thickness=cv.FILLED)
binarized_roi[mask == 0] = 0
if not (len(contours)==11): #if the cell is defective, i.e. does not contain 11 nuclei, delete it
self.last_rois.pop(i)
continue
counting = 0
for c in contours:
# number the nuclei and write the total number of nuclei in the upper right corner
x, y, bw, bh = cv.boundingRect(c)
h, w = binarized_roi.shape[:2]

x0 = max(0, x)
y0 = max(0, y)
x1 = min(w - 1, x + bw)
y1 = min(h - 1, y + bh)

cx = int(x0 + (x1 - x0) / 2)
cy = int(y0 + (y1 - y0) / 2)

counting +=1
cv.putText(binarized_roi, str(counting), (cx, cy), cv.FONT_HERSHEY_PLAIN,0.5, 0,1)
cv.putText(binarized_roi, str(counting), (5, 5), cv.FONT_HERSHEY_PLAIN,0.5, 1,1)
self.last_rois_binary.append(binarized_roi)
self.save_rects_current() #opens a dialog for selecting a target directory and saves the output there (binarized and grey-scale images = dataset)
```

![binary_roi_006.png](Segmentation%20using%20traditional%20methods/pics/pipeline_1/img1/binary_roi_006.png)

### Overview of implemented operations
Basic **single-image (monadic)** operations – change the brightness, contrast and intensity properties of pixels.

| Method | Description | Usage |
|--------|--------|------|
| **`negate(img)`** | Inverts the image (swaps light and dark areas). | Edge enhancement |
| **`adjust_gamma(img, gamma)`** | Performs gamma correction — adjusts the brightness of the image nonlinearly. | Compensates for differences in lighting, brightening or darkening images. |
| **`adjust_brightness(img, brightness)`** | Linearly adds or subtracts brightness. | Corrects underexposed or overexposed images. |
| **`adjust_contrast(img, contrast)`** | Changes contrast by multiplying pixel values. | Enhances details, improves sharpness. |
| **`non_linear_contrast(img, alpha)`** | Nonlinear contrast adjustment using a transition between highlights and shadows. | More natural contrast improvement than the linear method. |
| **`logarithmic_scale(img, s)`** | Logarithmic brightness transformation. | Enhance details in dark areas. |
| **`quantization(img, q)`** | Reduce the number of brightness levels (quantization). | Image size reduction, stylization, preprocessing for classification. |

---
Collection of **filtering and edge methods** – used to remove noise, blur or highlight structures.

| Method | Description | Usage |
|--------|--------|----------|
| **`mean_blur(img, ksize)`** | Averaging filter. | Noise reduction, image smoothing. |
| **`gaussian_blur(img, ksize, sigma)`** | Gaussian blur. | Noise suppression, detail removal before segmentation. |
| **`median_filter(img, ksize)`** | Median filter. | Remove impulse noise ("salt & pepper") without blurring edges. |
| **`bilateral_filter(img, d, sigma_color, sigma_space)`** | Bilateral filter - preserves edges when smoothing. | Reduce noise while preserving cell or object contours. |
| **`canny(img, threshold1, _, threshold2)`** | Canny edge detector. | Detect edges and shapes of cells, objects or structures. |
| **`diff_of_gauss(img, sigma, k)`** | Difference of two Gaussians (DoG). | Edge enhancement similar to Laplace, often used in biological processing. |
| **`laplacian_of_gauss(img, sigma, ksize_lap)`** | Laplace filter after Gaussian blur (LoG). | Detect edges and intensity transitions with less noise. |
---
Advanced **morphological and segmentation** methods – useful for analyzing cells, objects and shapes.

| Method | Description | Usage |
|-------------------------------------------|----------------------------------------------------------|--------------------------------------------------------------|
| **`hist_equalization(img)`** | Histogram normalization – equalizes brightness and contrast. | Improves visibility of details. |
| **`clahe(img, clip_limit, tile_grid)`** | Adaptive histogram equalization. | Contrast equalization for inhomogeneously illuminated images. |
| **`erosion(img, size)`** | Erosion – reduces bright areas. | Removes small noise or separates nearby objects. |
| **`dilatation(img, size)`** | Dilation – expands bright areas. | Fills gaps, highlights structures. |
| **`opening(img, size)`** | Opening (erosion + dilation). | Remove small noise points while preserving shapes. |
| **`closing(img, size)`** | Closing (dilation + erosion). | Smooth edges, remove small dark holes. |
| **`otsu(img)`** | Otsu's algorithm for automatic threshold segmentation. | Automatic image division into background and objects. |
| **`find_contours(img, min_area)`** | Find object outlines. | Extract cell or particle shapes. |
| **`save_rect_images(rect_imgs, out_dir)`** | Save cropped areas as separate images. | Export individual objects for dataset. |
| **`split_middle_cell(img)`** | Find and split the central cell with a line. | ​​Help with separating connected cells. |
| **`image_reconstruct(marker, img)`** | Morphological reconstruction. | Recovering objects from damaged or partially deleted data. |

---
## Results
The following table describes how my project turned out.

| Image name | Number of "cells" | Number of defective cells | Percentage of defective | Final number of cells |
|-------------------|--------------|-------------------|-------------------|-------------------|
| PCD1.tiff | 69 | 3 | 4.34% | 66 |
| PCD2.tiff | 18 | 7 | 38.88% | 11 |
| PCD3.tiff | 50 | 2 | 4.00% | 48 |
| Tv8.tiff | 18 | 0 | 0.00% | 18 |
| Tv11.tiff | 66 | 5 | 8.12% | 61 |
| Tv17.tiff | 24 | 3 | 12.50% | 21 |
| Tv21.tiff | 29 | 1 | 3.44% | 28 |
| Tv31.tiff | 30 | 5 | 16.66% | 25 |
| Tv33.tiff | 37 | 0 | 0.00% | 37 |
| **SUM/AVERAGE** | **341** | **26** | **7.63%** | **315** |

# CNN implementation

The goal of this task was to use the dataset from the first task and train a U-Net on it.

![test_00006.png](CNN%20implementation/out/test_predictions_25-11-04_14-14-44/test_preds/test_00006.png)

## Part 1: Dataset Preparation

The dataset consisted of about 300 images from the first task,
with the change that the black line dividing the middle ciliary tobule was removed.

The dataset was divided into the original cilia images and their black-and-white masks.

The original bitmap images were loaded as grey-scale and enlarged to a resolution of 256x256 pixels using bilinear interpolation.

The masks were loaded in the same way, but enlarged using nearest neighbor interpolation, which in this case was relatively easy to implement, since these are black-and-white images.

To increase the size of the input dataset, augmentation was used in the form of rotating the images by 90°, 180° and 270°. Thanks to the rotations chosen in this way, we did not have to deal with what would happen to the empty space if we rotated the image by, for example, 45°. It was also found that increasing the number of rotations (e.g., if we rotated the image by 15°), training the network would take too much time (assuming we did not set the number of epochs to some low number).
## Part 2: U-Net Architecture

U-Net is implemented in a five-layer architecture (5 downsample blocks followed by a bottleneck and 5 upsample blocks). The number of convolutional filters on the input layer is 16; in each subsequent encoder stage, the number of filters doubles (i.e. 16, 32, 64, 128, 256). Given that the image is 256x256 at the input, after pooling in the encoder it gets to 16x16.

The decoder is symmetric to the encoder and in each level it gradually reduces the number of channels back to 16, using skip connections connecting the corresponding layers.

When implementing the resulting solution, some modifications were tested, but they did not work effectively enough and therefore were not used. Among these modifications were: 1） reducing the number of layers to 2, which caused the network to mark some parts as false positives.

![test_00005fp.png](CNN%20implementation/out/test_predictions_25-11-04_07-12-49/test_preds/test_00005.png)

2） The number of convolutional filters on the input layer was set to 32. This led to an increase in the computational complexity during training without any changes in the efficiency of the resulting network. Therefore, the initial 16 was chosen.

## Part 3: Model Training

The model training used the created U-Net, dividing the dataset into three parts, a combination of two loss functions and early-stopping.

### Dataset division

The dataset was divided into three parts: training, validation and testing. The ratio of these groups was 70/15/15. This number was chosen due to the size of the dataset, its augmentation and also because it was mentioned in the lecture.

```python
train_size = int(0.7 * len(ds)) # 70%
val_size = int(0.15 * len(ds)) # 15%
test_size = len(ds) - train_size - val_size # 100-70-15 = 15%
```

### Batch size

During training, a batch size of 8 was also chosen primarily because of my memory size, however, according to my informal research, increasing the batch size to 16 or 32 would not have a significant effect.

### Loss function

During training the model, two loss functions were tested, namely dice loss, binary cross-entropy loss and subsequently their combination. This combination used both functions at the same time, with their ratio being determined by the alpha value, which determined the "weight" of BCE. For example, if alpha=0.8, BCE had a weight of 80% and dice only 20%; if alpha=0.5, the features would have the same weight.

In the end, it worked best to use either BCE or dice; their combination was successful only if alpha was very low.

### Early stop and number of epochs

The number of epochs and early-stop are closely related, so I will describe them in one subsection.

The number of epochs was set to 120, but due to the early-stop algorithm, the training algorithm reached a maximum of 30.

The early-stop algorithm checks the validation loss after each epoch; if it has "improved", the algorithm continues, if it has not improved 5 times in a row, the training ends and the best validation loss is restored.
The following transcription is considered an improvement: `val_loss < best_val_loss - 1e-5`

### Visualizations of the process

In this subsection, you can find graphs of various metrics that were measured during training the network with different parameters. Among these parameters we usually find the loss function, since the early-stop and the number of network slots were set statically and its changes did not occur often.

When choosing the BCE+Dice loss function with a BCE weight of 0.9, training ended relatively quickly (after 12 epochs), but the loss values ​​were not good, see the figure below.

![bcedice09.png](CNN%20implementation/out/test_predictions_25-11-06_08-50-55_bce09/loss_curves.png)
![bcedice09.png](CNN%20implementation/out/test_predictions_25-11-06_08-50-55_bce09/dice_curve.png)

The opposite was choosing BCE with 0.1, which caused the network to train for 59 epochs before hitting an early-stop, but its results were more favorable compared to the previous attempt.

![bcedice01.png](CNN%20implementation/out/test_predictions_25-11-06_09-05-23/loss_curves.png)
![bcedice01.png](CNN%20implementation/out/test_predictions_25-11-06_09-05-23/dice_curve.png)

When choosing BCE 0.5 (i.e., both loss functions had the same weight), the result was again better. It was achieved that the network was trained for +-30 epochs with a loss < 0.1

When choosing only the BCE loss function, the best results were achieved compared to the other attempts. The number of epochs was only 32, with losses and dice showing better values ​​than previous attempts.

![bcedice05.png](CNN%20implementation/out/test_predictions_25-11-06_10-23-06/loss_curves.png)
![bcedice05.png](CNN%20implementation/out/test_predictions_25-11-06_10-23-06/dice_curve.png)

## Part 4: Model Evaluation

### Qualitative

Qualitative evaluation of the model is relatively difficult, because the results turn out to be better than the ground truth. This is because in the first task the morphological operation opening with a rectangular kernel was used on the tobula; therefore the network results are not so square and look "better".

The chosen network did not ultimately suffer from significant quantitative fluctuations. These fluctuations (e.g. false positives) were described in previous chapters, and were caused by a "shallow" network or a bad combination of loss functions.

### Quantitative

In this chapter, quantitative evaluations of the best network are shown.

![bcedice05.png](CNN%20implementation/out/test_predictions_25-11-06_10-23-06/prf1_curves.png)

![bcedice05.png](CNN%20implementation/out/test_predictions_25-11-06_10-23-06/iou_curve.png)

![bcedice05.png](CNN%20implementation/out/test_predictions_25-11-06_10-23-06/accuracy_curve.png)

![bcedice05.png](CNN%20implementation/out/test_predictions_25-11-06_10-23-06/conf_matrix_epoch_37.png)

![bcedice05.png](<./out/test_predictions_25-11-06_10-23-06/progress_contours1(Copy).png>)
![bcedice05.png](<./out/test_predictions_25-11-06_10-23-06/progress_contours(Copy).png>)

## Part 5: Hyperparameter Tuning

In the project I used U-Net, which is a convolutional encoder–decoder architecture for image segmentation. The network consists of two parts:

- encoder (downsampling): gradually reduces the resolution and increases the number of filters, thus extracting more abstract features from the image,

- decoder (upsampling): restores spatial resolution, using skip-connections from the encoder, so we combine low-level details with high-level semantics.

Each level contains 2× 3×3 convolutions and ReLU activations. Between levels, MaxPooling (2×2) is used to reduce the resolution and transposed convolution (2×2) is used to increase the resolution in the decoder. The output layer is a 1×1 convolution that produces a logit map (1 channel), suitable for binary segmentation using BCEWithLogitsLoss.

When using a U-Net with 2 layers, the number of pixels as false positives increased; when using 10 layers, the training was unnecessarily long.

The five-level variant came out as the best:

- provides sufficient representation even for fine details,

- does not have such a large number of parameters that it overtrains,

- has the best performance / training time / memory ratio,

- stable training and the highest validation Dice and IoU.

Average pooling had a problem with preserving edges and textures; max pooling came out as a better option in this regard.

## Sample results

![vysledek](./out/test_predictions_25-11-06_10-23-06/test_preds/test_00076.png)

![vysledek](./out/test_predictions_25-11-06_10-23-06/test_preds/test_00040.png)

![vysledek](./out/test_predictions_25-11-06_10-23-06/test_preds/test_00077.png)

![vysledek](./out/test_predictions_25-11-06_10-23-06/test_preds/test_00028.png)





# Assignment 02

The goal of this assignment was to use the dataset from the first assignment and train the U-Net on it.

![test_00006.png](CNN%20implementation/out/test_predictions_25-11-04_14-14-44/test_preds/test_00006.png)

## Part 1: Dataset Preparation

The dataset consisted of about 300 images from the first assignment,
with the change that the black line dividing the middle ciliary tobule was removed.

The dataset was divided into the original cilia images and their black and white masks.

The original bitmap images were loaded as grey-scale and enlarged to a resolution of 256x256 pixels using bilinear interpolation.
The masks were also loaded in the same way, but they were enlarged using nearest neighbor interpolation, which in this case was relatively easy to do, since these are black and white images.

To increase the size of the input dataset, augmentation was used in the form of rotating the images by 90°, 180° and 270°. Thanks to the rotations chosen in this way, we did not have to consider what would happen to the empty space if we rotated the image by, for example, 45°. It was also found that if we increased the number of rotations (e.g. if we rotated the image by 15°), training the network would take too much time (assuming we did not set the number of epochs to a low number).

## Part 2: U-Net Architecture

U-Net is implemented in a five-layer architecture (5 downsample blocks followed by a bottleneck and 5 upsample blocks). The number of convolutional filters on the input layer is 16; at each subsequent stage of the encoder, the number of filters doubles (i.e. 16, 32, 64, 128, 256). Given that the image is 256x256 at the input, after pooling in the encoder it reaches 16x16.
The decoder is symmetrical to the encoder and in each level it gradually reduces the number of channels back to 16, using skip connections connecting the corresponding layers.

When implementing the resulting solution, some modifications were tested, but they did not work effectively enough and therefore were not used. Among these modifications were: 1） reducing the number of layers to 2, this caused the network to mark some parts as false positives.

![test_00005fp.png](CNN%20implementation/out/test_predictions_25-11-04_07-12-49/test_preds/test_00005.png)

2） The number of convolutional filters on the input layer was set to 32. This led to an increase in the computational complexity during training without any changes in the efficiency of the resulting network. Therefore, the initial 16 was chosen.

## Part 3: Model Training

The model training used the created U-Net, dividing the dataset into three parts, a combination of two loss functions and early-stopping.

### Dataset division

The dataset was divided into three parts: training, validation and testing. The ratio of these groups was 70/15/15. This number was chosen due to the size of the dataset, its augmentation and also because it was mentioned in the lecture.

```python
train_size = int(0.7 * len(ds)) # 70%
val_size = int(0.15 * len(ds)) # 15%
test_size = len(ds) - train_size - val_size # 100-70-15 = 15%
```

### Batch size

A batch size of 8 was also chosen for training primarily due to my memory size, however, according to my informal research, increasing the batch size to 16 or 32 would not have a significant effect.
### Loss function

Two loss functions were tested when training the model, namely dice loss, binary cross-entropy loss and subsequently their combination. This combination used both functions at the same time, with their ratio being determined by the alpha value, which determined the "weight" of BCE. For example, if alpha=0.8, BCE had a weight of 80% and dice only 20%; if alpha=0.5, the functions would have the same weight.

In the end, it worked best to use either BCE or dice; their combination was successful only if alpha was very low.

### Early stop and number of epochs

The number of epochs and early-stop are closely related, so I will describe them in one subsection.

The number of epochs was set to 120, but due to the early-stop algorithm, the training algorithm reached a maximum of 30.

The early-stop algorithm checks the validation loss after each epoch; if it "improved", the algorithm continues, if it did not improve 5 times in a row, the training ends and the best validation loss is restored.
The following transcription is considered an improvement: `val_loss < best_val_loss - 1e-5`

### Visualizations of the process

In this subsection, you can find graphs of various metrics that were measured during network training with different parameters. Among these parameters, we usually find the loss function, since the early-stop and the number of network slots were set statically and its changes did not occur often.

When choosing the BCE+Dice loss function with a weight of BCE 0.9, the training ended relatively quickly (after 12 epochs), but the loss values ​​were not good, see. the image below.

![bcedice09.png](CNN%20implementation/out/test_predictions_25-11-06_08-50-55_bce09/loss_curves.png)
![bcedice09.png](CNN%20implementation/out/test_predictions_25-11-06_08-50-55_bce09/dice_curve.png)

The opposite was choosing BCE with 0.1, which caused the network to train for 59 epochs before hitting an early-stop, but its results were more favorable compared to the previous attempt.

![bcedice01.png](CNN%20implementation/out/test_predictions_25-11-06_09-05-23/loss_curves.png)
![bcedice01.png](CNN%20implementation/out/test_predictions_25-11-06_09-05-23/dice_curve.png)

When choosing BCE 0.5 (i.e., both loss functions had the same weight), the result was again better. It was achieved that the network was trained for +-30 epochs with a loss < 0.1

When choosing only the BCE loss function, the best results were achieved compared to the other attempts. The number of epochs was only 32, with losses and dice showing better values ​​than previous attempts.

![bcedice05.png](CNN%20implementation/out/test_predictions_25-11-06_10-23-06/loss_curves.png)
![bcedice05.png](CNN%20implementation/out/test_predictions_25-11-06_10-23-06/dice_curve.png)

## Part 4: Model Evaluation

### Qualitative

Qualitative evaluation of the model is relatively difficult to perform, because the results turn out to be better than the ground truth. This is because in the first task the morphological operation opening with a rectangular kernel was used on the tobula; therefore the network results are not so square and look "better".

The chosen network did not ultimately suffer from significant quantitative fluctuations. These fluctuations (e.g. false positives) were described in previous chapters, and were caused by a "shallow" network or a bad combination of loss functions.

### Quantitative

In this chapter, quantitative evaluations of the best network are shown.

![bcedice05.png](CNN%20implementation/out/test_predictions_25-11-06_10-23-06/prf1_curves.png)

![bcedice05.png](CNN%20implementation/out/test_predictions_25-11-06_10-23-06/iou_curve.png)

![bcedice05.png](CNN%20implementation/out/test_predictions_25-11-06_10-23-06/accuracy_curve.png)

![bcedice05.png](CN N%20implementation/out/test_predictions_25-11-06_10-23-06/conf_matrix_epoch_37.png)

![bcedice05.png](./CNN%20implementation/out/test_predictions_25-11-06_10-23-06/progress_contours1(Copy).png)
![bcedice05.png](./CNN%20implementation/out/test_predictions_25-11-06_10-23-06/progress_contours(Copy).png)

## Part 5: Hyperparameter Tuning

In the project I used U-Net, which is a convolutional encoder–decoder architecture for image segmentation. The network consists of two parts:

- encoder (downsampling): gradually reduces the resolution and increases the number of filters, thus extracting more abstract features from the image,

- decoder (upsampling): restores spatial resolution, using skip-connections from the encoder, so we combine low-level details with high-level semantics.

Each level contains 2× 3×3 convolutions and ReLU activations. Between levels, MaxPooling (2×2) is used to reduce the resolution and transposed convolution (2×2) is used to increase the resolution in the decoder. The output layer is a 1×1 convolution that produces a logit map (1 channel), suitable for binary segmentation using BCEWithLogitsLoss.

When using a U-Net with 2 layers, the number of pixels as false positives increased; when using 10 layers, the training was unnecessarily long.

The five-level variant came out as the best:

- provides sufficient representation even for fine details,

- does not have such a large number of parameters that it overtrains,

- has the best performance / training time / memory ratio,

- stable training and the highest validation Dice and IoU.

Average pooling had a problem with preserving edges and textures; max pooling came out as a better option in this regard.

## Sample results

![vysledek](./CNN%20implementation/out/test_predictions_25-11-06_10-23-06/test_preds/test_00076.png)

![vysledek](./CNN%20implementation/out/test_predictions_25-11-06_10-23-06/test_preds/test_00040.png)

![vysledek](./CNN%20implementation/out/test_predictions_25-11-06_10-23-06/test_preds/test_00077.png)

![vysledek](./CNN%20implementation/out/test_predictions_25-11-06_10-23-06/test_preds/test_00028.png)



# Advanced CNN, transfer learning, and XAI

> Author: Kryštof Čejchan
## Objective
Implementation and evaluation of classification models (CNN), analysis of their decision (XAI) and Siamese networks.
## Part 1: Classification (Original vs. Inpainted)

### Dataset preparation

The dataset was prepared from bitmap images from previous tasks. In total, the dataset had ± 300 images, of which 50% were inpainted and the rest were left unchanged (50%).

For inpainted, cilia masks were used, which were dilated and then at least five random cells were "repainted" using the function `cv.inpaint`.

<img alt="img_008.png" height="256" src="Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/img_008.png" width="256"/>
<img alt="7.png" height="256" src="Advanced%20CNN,%20transfer%20learning,%20and%20XAI/data/orig_inpainted/inpainted/7.png" width="256"/>

### CNN Models

#### Custom Network

Feature Extractor: A series of five convolutional blocks that gradually reduce the spatial dimension of the image and increase the depth of the features (number of channels).

Classifier: Fully Connected layers that convert the extracted features into a final prediction.

The input is a greyscale image (i.e. one channel) with a resolution of 256x256. The other layers are as follows:

| Network Part | Layer | Operation Type | Configuration | Output Tensor |
|:-------------|:------------|:----------------|:----------------------------|:----------------|
| Input | - | - | - | (1, 256, 256) |
| Block 1 | conv1 | Convolution + ReLU | k=5, s=1, p=2 | (16, 256, 256) |
| - | pool1 | Max Pooling | k=2, s=2 | (16, 128, 128) |
| Block 2 | conv2 | Convolution + ReLU | k=5, s=1, p=2 | (32, 128, 128) |
| - | pool2 | Max Pooling | k=2, s=2 | (32, 64, 64) |
| Block 3 | conv3 | Convolution + ReLU | k=3, s=1, p=1 | (64, 64, 64) |
| - | pool3 | Max Pooling | k=2, s=2 | (64, 32, 32) |
| Block 4 | conv4 | Convolution + ReLU | k=3, s=1, p=1 | (128, 32, 32) |
| - | pool4 | Max Pooling | k=2, s=2 | (128, 16, 16) |
| Block 5 | conv5 | Convolution + ReLU | k=3, s=1, p=1 | (256, 16, 16) |
| - | pool5 | Max Pooling | k=2, s=2 | (256, 8, 8) |
| Flatten | - | Flatten | - | (256 * 64) |
| Classifier | fc1 | Linear + ReLU | Input: 256 * 64, Output: 512 | (512) |
| - | fc2 | Linear + ReLU | Input: 512, Output: 128 | (128) |
| - | fc3 | Linear (Logits) | Input: 128, Output: 2 | (2) |

`k = kernel_size; s = stride; p = padding`

#### Transfer learning

The standard ResNet18 is designed for color RGB images (3 channels) and classification up to 1000 classes. For the needs of this task (greyscale images [1 channel], classification up to 2 classes) the following modifications were made:

1. Adaptation of the input layer (Grayscale)

The original input convolutional layer (conv1) expects 3 input channels (RGB). Since we are working with grayscale images (1 channel), this layer was replaced by a new convolution:

Original: in_channels=3
New: in_channels=1 (other parameters such as kernel size, stride and padding were preserved).

In order not to lose the information learned from the RGB version, the weights of the new layer were not initialized randomly. Instead, the average of the weights over the original 3 channels was calculated.

```py
self.model.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)
```

This technique allows the network to respond to structural features in a grayscale image in a similar way as it would respond to the luminance component of a color image.

2. Head Modification

The original fully connected layer (fc), which mapped features to 1000 ImageNet classes, has been removed and replaced with a new linear layer corresponding to our specification:

Input: 512 features (output from the last ResNet block).

Output: num_classes (in our case 2: Original vs. Inpainted).

3. Training Strategy (Freezing)

The class supports the freeze_base parameter, which allows freezing the weights of the feature extractor:

If freeze_base=True: Gradients are calculated only for the new classification head (fc). This is suitable for fast fine-tuning, when we assume that the learned features from ImageNet are general enough.

If freeze_base=False (default in the code): The entire network is trained. The weights from ImageNet serve as a very good starting point for initialization, but during training they are finely adjusted to the specifics of the target dataset.

### Training

1. Data division and preparation

Data is loaded from a directory structure, where classes (original, inpainted) are separated into subdirectories. Before the actual training, the following processing takes place:

Dataset division: All available images are randomly shuffled and divided into three disjoint sets based on defined ratios (70/15/15):

Training set: Used to optimize the model weights.

Validation set: Used to continuously evaluate the model and decide on early stopping.

Test set: Used exclusively after training for the final measurement of model performance.

Data Augmentation: To increase the robustness of the model and prevent overfitting, data augmentation is applied to the training set in the form of rotations of 90°, 180° and 270°. The validation and test sets remain without rotations (angle 0°).

2. Training Configuration

The following parameters and components were selected for model optimization:

Loss function: CrossEntropyLoss was used

Optimizer: The Adam (Adaptive Moment Estimation) algorithm was selected, which effectively adapts the learning rate for individual network parameters.

Model: The script uses the ResNetClassifier or Net class (depending on whether it is transfer learning or not), which is initialized and moved to the CPU computing device.

3. Training Loop and Early Stopping

Training takes place in cycles (epochs). Each epoch consists of two phases:

Training phase (model.train()):

The model processes data in batches.

For each batch, the error (loss) is calculated, backpropagation is performed, and the weights are updated using the optimizer.

Validation phase (model.eval()):
The model is switched to evaluation mode (dropout is disabled, batch norm is fixed).

Without calculating gradients (torch.no_grad()), prediction is performed on the validation set.

Metrics calculation: The network outputs (logits for 2 classes) are transformed into a binary prediction by the difference of scores (class_1 - class_0), which allows the calculation of Accuracy, Precision, Recall and F1-Score using the Metrics class.

Early Stopping Strategy: To avoid overfitting and wasting computational time, the Early Stopping mechanism is implemented.

The validation loss value is monitored.

If the current validation loss is lower than the best recorded so far, the model (its weights) is saved as the best candidate.

If the loss does not improve after a specified number of epochs (PATIENCE parameter), training is automatically terminated.

4. Visualization and final testing

After training, the script generates progress graphs:

Loss Graph: Comparison of the evolution of training and validation errors over time.

Accuracy Graph: Evolution of the model accuracy on the validation set.

In the last phase, the weights of the model with the lowest validation error (not the weights from the last epoch) are loaded and inference is performed on the test set.

Based on the prediction, the images are physically sorted into predicted_original and predicted_inpainted folders for visual inspection.

Final performance metrics are calculated on data that the model has never seen during training.

### Outputs
My network:

![accuracy_net.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/accuracy_net.png)

Transfer learning:

![accuracy_net.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/resnet/accuracy_net.png)

My network:

![graph_loss_net.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/graph_loss_net.png)

Transfer learning:

![graph_loss_net.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/resnet/graph_loss_net.png)


My network:

![confusion_matrix.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/confusion_matrix.png)

| Metric   | Score  |
|-----------|--------|
| Precision | 0.7391 |
| Recall    | 1.0000 |
| F1-Score  | 0.8500 |
| IoU       | 0.7391 |


Transfer learning:

![confusion_matrix.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/resnet/confusion_matrix.png)


| Metric   | Score  |
|-----------|--------|
| Precision | 1.0000 |
| Recall    | 1.0000 |
| F1-Score  | 1.0000 |
| IoU       | 1.0000 |

## Part 2: Model Interpretability

To validate the decision-making process of the neural network and verify that the model focuses on relevant visual features, a set of Explainable AI (XAI) methods was implemented. For this purpose, the Captum library was used, which allows analyzing the contributions of individual pixels to the final prediction of the model.

The analysis was performed on the trained model (Net or ResNetClassifier) ​​using three different gradient methods. Each of them provides a different perspective on what the model considers important.

Visualization methods used

1. Saliency (Gradient-based):

A basic method that calculates the gradient of the output relative to the input image. The resulting map indicates which pixels would have the most impact on the resulting class score if changed slightly.

Visualization: Uses the absolute value of the gradients (sign="absolute_value") and the inferno color map to highlight the areas with the highest sensitivity regardless of the direction of the influence.

2. Integrated Gradients (IG):

This method solves the problem of gradient saturation by integrating gradients along the path from the reference "null" input (black image) to the current input. It provides more stable and less noisy results than a simple Saliency map.

Visualization: Shows only positive contributions (sign="positive", Reds map), i.e. those areas that directly increase the probability of the predicted class.

3. Guided Grad-CAM:

Combines the localization capability of the Grad-CAM method with the detailed resolution of Guided Backpropagation. This method tracks activations in the last convolutional layer of the network, which contains the highest level of semantic information.

Configuration: The conv5 layer was chosen as the target layer (target_layer) for the Net network itself (in ResNet it would be the last block layer4). This layer serves as the source for calculating the importance weights of individual feature maps.

Visualization: The output is a detailed map (viridis map) that highlights key structures (e.g. edges or cilia textures) leading to a decision.

### My network
![compare_18.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/compare_18.png)

![compare_98.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/compare_98.png)

![compare_99.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/compare_99.png)

![compare_102.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/compare_102.png)

### Resnet

![compare_0.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/resnet/compare_0.png)

![compare_1.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/resnet/compare_1.png)

![compare_10.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/resnet/compare_10.png)

![compare_55.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/resnet/compare_55.png)

![compare_62.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/resnet/compare_62.png)


### Comparison (my network vs. resnet)
![compare_0.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/compare_0.png)

![compare_0.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/resnet/compare_0.png)

![compare_99.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/compare_99.png)

![compare_99.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/resnet/compare_99.png)

## Part 3: Siamese Networks
A Siamese Neural Network was implemented for the inpainting detection task. Unlike classical classification networks that learn to assign a specific class to the input, a Siamese network learns a similarity metric. The goal is to transform the input images into a vector space (embedding space) so that the vectors of images of the same class are close to each other and the vectors of different classes are far from each other.

1. Model architecture

ResNet18, pre-trained on ImageNet, was chosen as the backbone of the model, which ensures robust feature extraction.

Input adaptation: Since the input data is black and white (1 channel), the first convolutional layer of ResNet was modified. The original weights for the 3 RGB channels were averaged into one channel, which allows the use of pre-trained information for grayscale inputs as well.

Shared weights: The network consists of two identical branches that share the same weights. Both images in the pair undergo the same transformation.

Embedding layer: The original ResNet classification head has been replaced by a Linear -> ReLU -> Linear sequence that maps the extracted features into a 128-dimensional output vector.

2. Data preparation and matching

Pairing is key to training a Siamese network. The SiameseDataset class generates training samples dynamically:

Positive pair (Label 0): Two different images of the same class (e.g. Original–Original or Inpainted–Inpainted).

Negative pair (Label 1): Two images of different classes (Original–Inpainted).

Balancing: The dataset is constructed so that the probability of selecting a positive and negative pair is 50:50, which prevents the network from biasing towards one of the variants.

3. Loss function (Contrastive Loss)

The Contrastive Loss function was used to optimize the weights. This function works with the Euclidean distance Dw
between the output vectors of the network.

4. Training process

Training is performed using the Adam optimizer with a learning rate of 0.0005.

Accuracy evaluation: The accuracy of the model is not measured classically, but based on distance thresholding. If the distance between vectors is less than threshold=margin/2, the pair is classified as "identical".

Early Stopping: To prevent overfitting, the validation loss (Loss) is monitored. If it does not improve after a specified number of epochs (PATIENCE), training is terminated early and the model with the lowest validation error is saved.

### Metrics
#### EVALUATION RESULTS (Threshold=0.5)
| Metrics | Score |
|-----|------|
| Accuracy | 0.6 |
| Recall | 0.72 |
| F1-Score | 0.51 |
| IoU | 0.6 |

| Metrics | Score |
|---------|-------|
| Accuracy | 0.6 |
| Recall | 0.72 |
| F1-Score | 0.51 |
| IoU | 0.6 |

| Class | Precision | Recall | F1-Score | Support |
|:-----------------|:--------:|:-------:|:-------:|:-------:|
| **Same (0)** | 0.51 | 0.72 | 0.60 | 25 |
| **Different (1)** | 0.72 | 0.51 | 0.60 | 35 |
| | | | | |
| **Accuracy** | | | | **0.60** | **60** |
| **Macro Avg** | 0.62 | 0.62 | 0.60 | 60 |
| **Weighted Avg** | 0.63 | 0.60 | 0.60 | 60 |


![training_metrics_siam.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/siam/training_metrics_siam.png)
![siamese_embeddings_vis.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/siam/siamese_embeddings_vis.png)
![siamese_confusion_matrix.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/siam/siamese_confusion_matrix.png)


# Movement detection & optical flow

> Author: Kryštof Čejchan


_Note.: readme contains gif files that are large and hence may taky a while to load._

In this task, I implemented algorithms and methods for superpixels and motion detection.

# Part 1: Comparison of superpixel methods and parameter choices (number of superpixels, threshold values).

This part of the task focused on image segmentation using superpixel methods. The goal was to replace working with
individual pixels with a more efficient representation using superpixels and classify them based on color similarity with a
reference sample.

The first step was to define the target color that we want to detect in the image.
For this purpose, the input image is first displayed to the user, who selects an ROI with the cursor. The colors of the pixels in this ROI are
averaged, which is the resulting target color.

```python
roi_rect = cv.selectROI("select rectangle sample", img, showCrosshair=True, fromCenter=False)
```

During testing, the following color was chosen as the target: `(L*a*b*): [ 60.25 146.97 122.47]`

Superpixel extraction was implemented using three methods: SLICO, LSC, SEEDS. The threshold for segmentation was set to 85.

## SLICO

Two algorithm parameter settings were tested, their effect did not differ significantly. With `region_size=20` it can be seen that
the regions are smaller (logically), but otherwise the segmentation is almost identical; the heatmap seems to be more accurate without
large regions, but it was at the cost of computational complexity.

The `ruler` parameter did not have much effect on the resulting image, except for a few new segments.

### parameters: region_size=30, ruler=15.0

![SLICO - Segmentation_screenshot_16.12.2025.png](Movement%20detection%20&%20optical%20flow/output/superpixels/3_new-run/SLICO%20-%20Segmentation_screenshot_16.12.2025.png)
![SLICO - Distance Heatmap_screenshot_16.12.2025.png](Movement%20detection%20&%20optical%20flow/output/superpixels/3_new-run/SLICO%20-%20Distance%20Heatmap_screenshot_16.12.2025.png)

### parameters: region_size=30, ruler=100.0

![SLICO - Segmentation_screenshot_16.12.2025.png](Movement%20detection%20&%20optical%20flow/output/superpixels/3_new_run_3/SLICO%20-%20Segmentation_screenshot_16.12.2025.png)
![SLICO - Distance Heatmap_screenshot_16.12.2025.png](Movement%20detection%20&%20optical%20flow/output/superpixels/3_new_run_3/SLICO%20-%20Distance%20Heatmap_screenshot_16.12.2025.png)

### parameters: region_size=20, ruler=8.0

![SLICO - Segmentation_screenshot_16.12.2025.png](Movement%20detection%20&%20optical%20flow/output/superpixels/3_new_run_2/SLICO%20-%20Segmentation_screenshot_16.12.2025.png)
![SLICO - Distance Heatmap_screenshot_16.12.2025.png](Movement%20detection%20&%20optical%20flow/output/superpixels/3_new_run_2/SLICO%20-%20Distance%20Heatmap_screenshot_16.12.2025.png)

## LSC

`region_size` again changed only the size of the registered segmentations. `ratio` changed the "roundness" of the segment, the larger
the value, the more "square" the segment looked.

### parameters: region_size=30, ratio=0.075

![LSC - Segmentation_screenshot_16.12.2025.png](Movement%20detection%20&%20optical%20flow/output/superpixels/3_new-run/LSC%20-%20Segmentation_screenshot_16.12.2025.png)
![LSC - Distance Heatmap_screenshot_16.12.2025.png](Movement%20detection%20&%20optical%20flow/output/superpixels/3_new-run/LSC%20-%20Distance%20Heatmap_screenshot_16.12.2025.png)

### parameters: region_size=20, ratio=0.5

![LSC - Segmentation_screenshot_16.12.2025.png](Movement%20detection%20&%20optical%20flow/output/superpixels/3_new_run_2/LSC%20-%20Segmentation_screenshot_16.12.2025.png)
![LSC - Distance Heatmap_screenshot_16.12.2025.png](Movement%20detection%20&%20optical%20flow/output/superpixels/3_new_run_2/LSC%20-%20Distance%20Heatmap_screenshot_16.12.2025.png)

## SEEDS

It is interesting to note the heatmap for the SEEDS method, where it can be seen that superpixels were selected based on `num_superpixels`,
which often leads to "empty" superpixels.

### parameters: num_superpixels=8000, num_levels=20

![SEEDS - Segmentation_screenshot_16.12.2025.png](Movement%20detection%20&%20optical%20flow/output/superpixels/3_new-run/SEEDS%20-%20Segmentation_screenshot_16.12.2025.png)
![SEEDS - Distance Heatmap_screenshot_16.12.2025.png](Movement%20detection%20&%20optical%20flow/output/superpixels/3_new-run/SEEDS%20-%20Distance%20Heatmap_screenshot_16.12.2025.png)

### parameters: num_superpixels=6000, num_levels=10

![SEEDS - Segmentation_screenshot_16.12.2025.png](Movement%20detection%20&%20optical%20flow/output/superpixels/3_new_run_2/SEEDS%20-%20Segmentation_screenshot_16.12.2025.png)
![SEEDS - Distance Heatmap_screenshot_16.12.2025.png](Movement%20detection%20&%20optical%20flow/output/superpixels/3_new_run_2/SEEDS%20-%20Distance%20Heatmap_screenshot_16.12.2025.png)

# Part 2: Visual comparison of background subtraction methods.

This part is dedicated to the analysis of motion in video sequences using background subtraction and optical flow methods.

## Background Subtraction

The output is a clean mask of moving objects, which serves as input for further analysis or object counting.

![Screencast from 2025-12-13 13-31-04.gif](https://krystofcejchan.cz/projects/medical_img_processing/Screencast%20from%202025-12-13%2013-31-04.gif)

# Sparse Optical Flow

First, "salient points" are detected in the image using the `goodFeaturesToTrack` detector. These points
typically correspond to corners or prominent textures on vehicles.

The method then calculates the displacement of these points between successive frames.

The result is a visualization of motion vectors that show the path of movement of individual objects over time. Unlike
`dense optical flow`, this method was computationally inexpensive and was able to process the video in real time.

![Screencast from 2025-12-13 13-40-29.gif](https://krystofcejchan.cz/projects/medical_img_processing/Screencast%20from%202025-12-13%2013-40-29.gif)

![Screencastfrom2025-12-1315-33-59-ezgif.com-video-to-gif-converter.gif](https://krystofcejchan.cz/projects/medical_img_processing/Screencastfrom2025-12-1315-33-59-ezgif.com-video-to-gif-converter.gif)

# Dense Optical Flow

Another implemented method was Dense Optical Flow, which calculates the motion vector
for each pixel in the image - the calculation was demanding, so a low-resolution video was chosen.

The output is a dense vector field that describes the movement of the entire scene.

A color code (HSV space) was used for visualization, where the color determines the direction of movement (e.g. red = right, blue =
left) and saturation/brightness determines the speed of movement (brighter = faster movement).

## Dense optiocal flow without morphological operations

![Screencastfrom2025-12-1313-50-54-ezgif.com-video-to-gif-converter.gif](https://krystofcejchan.cz/projects/medical_img_processing/Screencastfrom2025-12-1313-50-54-ezgif.com-video-to-gif-converter.gif)

## Dense optiocal flow with morphological operations

When using the morphological operation `close`, the mask was filled better, which caused the bounding rectangle not to shrink (
see previous image)

![Screencastfrom2025-12-1314-54-53-ezgif.com-video-to-gif-converter.gif](Movement%20detection%20&%20optical%20flow/output/dense/Screencastfrom2025-12-1314-54-53-ezgif.com-video-to-gif-converter.gif)

# Part 2 (Discussion): Answers to the implementation challenges in Optical Flow (handling new objects, duplicates, and collisions).

## New object detection

### Sparse

Every third frame (`frame_idx % detect_interval == 0`) the `cv.goodFeaturesToTrack` function is called. This will find new prominent points in the image
and add them to the tracks list.

### Dense

The `cv.calcOpticalFlowFarneback` function calculates the motion vectors for the entire image in each frame. Then, a threshold (`cv.threshold`) is applied to the size of the vectors.

## Duplicate resolution

### Sparse

Before calling the new point detector, a mask is created where the positions of the currently tracked points are blackened (
`value 0`). The `cv.goodFeaturesToTrack` function searches for points only in the white areas of the mask. This mathematically guarantees that
you will not start tracking a point that the system already registers.

```jupyter
mask = np.zeros_like(frame_gray)
mask[:] = 255
for tr in tracks:
x, y = tr[-1]
cv.circle(mask, (int(x), int(y)), 5, 0, -1)
```

### Dense

Dense optical flow can be noisy and one object can look like many small spots, i.e. one object would be
registered multiple times. The morphological operation close will merge these nearby points into one solid blob, thus preventing
multiple detection of one car.

```jupyter
processed_mask = cv.morphologyEx(motion_mask, cv.MORPH_CLOSE, k, iterations=9)
```

## Collision Resolution

### Sparse

The algorithm calculates the motion of a point from time t to t+1 and then the backward motion from t+1 to t.
If the tracking is correct, the point should return exactly to its original location. If there is a collision or occlusion, the point
is "lost" or gets stuck on another object, and the backward motion ends somewhere else. If the distance d is greater than 1 pixel, the point
is discarded as unreliable.

```jupyter
p1, _, _ = cv.calcOpticalFlowPyrLK(img0, img1, p0, ...)
p0r, _, _ = cv.calcOpticalFlowPyrLK(img1, img0, p1, ...)
d = abs(p0 - p0r).reshape(-1, 2).max(-1)
good = d < 1
```

### Dense

The identity of the objects is not addressed here, only the detection of motion. In case of collision, their motion masks are merged into one large contour.
Unlike the sparse method, there is no mechanism that would recognize that a specific point has been "lost". The object is simply perceived as one large moving cluster after the collision.
