Download Link: https://assignmentchef.com/product/solved-csci5561-homework-1-histogram-of-oriented-gradients
<br>
Histogram of oriented gradients. HOG feature is extracted and visualized for (a) the entire image and (b) zoom-in image. The orientation and magnitude of the red lines represents the gradient components in a local cell.

In this assignment, you will implement a variant of HOG (Histogram of Oriented Gradients) in MATLAB proposed by Dalal and Trigg [1] (2015 Longuet-Higgins Prize Winner). It had been long standing top representation (until deep learning) for the object detection task with a deformable part model by combining with a SVM classifier [2]. Given an input image, your algorithm will compute the HOG feature and visualize as shown in Figure 1 (the line directions are perpendicular to the gradient to show edge alignment). The orientation and magnitude of the red lines represents the gradient components in a local cell.

function [hog] = HOG(im)

<strong>Input: </strong>input gray-scale image with uint8 format.

<strong>Output: </strong>HOG descriptor.

<strong>Description: </strong>You will compute the HOG descriptor of input image im. The pseudocode can be found below:

<strong>Algorithm 1 </strong>HOG

1: Convert the gray-scale image to double format.

2: Get differential images using GetDifferentialFilter and FilterImage

3: Compute the gradients using GetGradient

4: Build the histogram of oriented gradients for all cells using BuildHistogram 5: Build the descriptor of all blocks with normalization using GetBlockDescriptor 6: Return a long vector (hog) by concatenating all block descriptors.

<h2>1.1            Image filtering</h2>

(a) Input image                           (b) Differential along x                  (c) Differential along y

direction                                           direction

Figure 2: (a) Input image dimension. (b-c) Differential image along <em>x </em>and <em>y </em>directions.

function [filter_x, filter_y] = GetDifferentialFilter() <strong>Input: </strong>none.

<strong>Output: </strong>filter_x and filter_y are 3×3 filters that differentiate along <em>x </em>and <em>y </em>directions, respectively.

<strong>Description: </strong>You will compute the gradient by differentiating the image along <em>x </em>and <em>y </em>directions. This code will output the differential filters.

function [im_filtered] = FilterImage(im, filter)

<strong>Input: </strong>im is the gray scale <em>m </em>× <em>n </em>image (Figure 2(a)) converted to double (refer to im2double built-in function); filter is a filter (<em>k </em>× <em>k </em>matrix)

<strong>Output: </strong>im_filtered is <em>m </em>× <em>n </em>filtered image. You may need to pad zeros on the boundary on the input image to get the same size filtered image.

<strong>Description: </strong>Given an image and filter, you will compute the filtered image. Given the two functions above, you can generate differential images by visualizing the magnitude of the filter response as shown in Figure 2(b) and 2(c).

<h2>1.2            Gradient Computation</h2>

Figure 3: Visualization of (a) magnitude and (b) orientation of image gradients. (c-e) Visualization of gradients at every 3rd pixel (the magnitudes are re-scaled for illustrative purpose.).

function [grad_mag, grad_angle] = GetGradient(im_dx, im_dy)

<strong>Input: </strong>im_dx and im_dy are the <em>x </em>and <em>y </em>differential images (size: <em>m </em>× <em>n</em>).

<strong>Output: </strong>grad_mag and grad_angle are the magnitude and orientation of the gradient images (size: <em>m </em>× <em>n</em>). Note that the range of the angle should be [0<em>,π</em>), i.e., unsigned angle (<em>θ </em>== <em>θ </em>+ <em>π</em>).

<strong>Description: </strong>Given the differential images, you will compute the magnitude and angle of the gradient. Using the gradients, you can visualize and have some sense with the image, i.e., the magnitude of the gradient is proportional to the contrast (edge) of the local patch and the orientation is perpendicular to the edge direction as shown in Figure 3.

<h2>1.3            Orientation Binning</h2>

<em>M</em>

Figure 4: (a) Histogram of oriented gradients can be built by (b) binning the gradients to corresponding bin.

function ori_histo = BuildHistogram(grad_mag, grad_angle, cell_size) <strong>Input: </strong>grad_mag and grad_angle are the magnitude and orientation of the gradient images (size: <em>m </em>× <em>n</em>); cell_size is the size of each cell, which is a positive integer. <strong>Output: </strong>ori_histo is a 3D tensor with size <em>M </em>× <em>N </em>× 6 where <em>M </em>and <em>N </em>are the number of cells along <em>y </em>and <em>x </em>axes, respectively, i.e., <em>M </em>= b<em>m/</em>cell_sizec and <em>N </em>= b<em>n/</em>cell_sizec where b·c is the round-off operation as shown in Figure 4(a). <strong>Description: </strong>Given the magnitude and orientation of the gradients per pixel, you can build the histogram of oriented gradients for each cell.

ori_histo(<em>i,j,k</em>) = <sup>X </sup>grad_mag(<em>u,v</em>)                if grad_angle(<em>u,v</em>) ∈ <em>θ<sub>k                      </sub></em>(1)

(<em>u,v</em>)∈C<em><sub>i,j</sub></em>

where C<em><sub>i,j </sub></em>is a set of <em>x </em>and <em>y </em>coordinates within the (<em>i,j</em>) cell, and <em>θ<sub>k </sub></em>is the angle range of each bin, e.g., ),

), and          ). Therefore, ori_histo(i,j,:)

returns the histogram of the oriented gradients at (<em>i,j</em>) cell as shown in Figure 4(b). Using the ori_histo, you can visualize HOG per cell where the magnitude of the line proportional to the histogram as shown in Figure 1. Typical cell_size is 8.

<h2>1.4            Block Normalization</h2>

Figure 5: HOG is normalized to account illumination and contrast to form a descriptor for a block. (a) HOG within (1,1) block is concatenated and normalized to form a long vector of size 24. (b) This applies to the rest block with overlap and stride 1 to form the normalized HOG.

function ori_histo_normalized = GetBlockDescriptor(ori_histo, block_size) <strong>Input: </strong>ori_histo is the histogram of oriented gradients without normalization. block_size is the size of each block (e.g., the number of cells in each row/column), which is a positive integer.

<strong>Output: </strong>ori_histo_normalized is the normalized histogram (size: (<em>m </em>− 1) × (<em>n </em>− 1) × (6 × block_size<sup>2</sup>).

<strong>Description: </strong>To account for changes in illumination and contrast, the gradient strengths must be locally normalized, which requires grouping the cells together into larger, spatially connected blocks (adjacent cells). Given the histogram of oriented gradients, you apply <em>L</em><sub>2 </sub>normalization as follow:

<ol>

 <li>Build a descriptor of the first block by concatenating the HOG within the block. You can use block_size=2, i.e., 2 × 2 block will contain 2 × 2 × 6 entries that will be concatenated to form one long vector as shown in Figure 5(a).</li>

 <li>Normalize the descriptor as follow:</li>

</ol>

(2)

where <em>h<sub>i </sub></em>is the <em>i</em><sup>th </sup>element of the histogram and <em>h</em><sup>ˆ</sup><em><sub>i </sub></em>is the normalized histogram. <em>e </em>is the normalization constant to prevent division by zero (e.g., <em>e </em>= 0<em>.</em>001).

<ol start="3">

 <li>Assign the normalized histogram to ori_histo_normalized(1,1) (white dot location in Figure 5(a)).</li>

 <li>Move to the next block ori_histo_normalized(1,2) with the stride 1 and iterate 1-3 steps above.</li>

</ol>

The resulting ori_histo_normalized will have the size of (<em>m </em>− 1) × (<em>n </em>− 1) × 24.