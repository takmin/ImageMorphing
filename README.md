ImageMorphing
=============

A function for image morphing using OpenCV
It requires OpenCV 2.3 or later.

Just copy the whole functions in Morph.cpp to your code or write header file for the function ImageMorphing().

In order to use ImageMorphing(), you need to prepare two input images and corresponding points on the both images.


//! Image Morphing
/*!
\param[in] src_img1 Input image 1
\param[in] src_points1 Points on the image 1
\param[in] src_img2 Input image 2
\param[in] src_points2 Points on the image 2, which must be corresponded to src_point1
\param[out] dst_img Morphed output image
\param[out] dst_points Morphed points on the output image
\param[in] shape_ratio blending ratio (0.0 - 1.0) of shape between image 1 and 2.  If it is 0.0, output shape is same as src_img1.
\param[in] color_ratio blending ratio (0.0 - 1.0) of color between image 1 and 2.  If it is 0.0, output color is same as src_img1. If it is negative, it is set to shape_ratio.
*/
void ImageMorphing(const cv::Mat& src_img1, const std::vector<cv::Point2f>& src_points1,
	const cv::Mat& src_img2, const std::vector<cv::Point2f>& src_points2,
	cv::Mat& dst_img, std::vector<cv::Point2f>& dst_points,
	float shape_ratio = 0.5, float color_ratio = -1)

