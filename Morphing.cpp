#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>


cv::Mat PointVec2HomogeneousMat(const std::vector<cv::Point2f>& pts)
{
	int num_pts = pts.size();
	cv::Mat homMat(3, num_pts, CV_32FC1);
	for(int i=0; i<num_pts; i++){
		homMat.at<float>(0,i) = pts[i].x;
		homMat.at<float>(1,i) = pts[i].y;
		homMat.at<float>(2,i) = 1.0;
	}
	return homMat;
}


// Morph points
void MorphPoints(const std::vector<cv::Point2f>& srcPts1, const std::vector<cv::Point2f>& srcPts2, std::vector<cv::Point2f>& dstPts, float s = 0.5)
{
	assert(srcPts1.size() == srcPts2.size());
	
	int num_pts = srcPts1.size();

	dstPts.resize(num_pts);
	for(int i=0; i<num_pts; i++){
		dstPts[i].x = (1.0 - s) * srcPts1[i].x + s * srcPts2[i].x;
		dstPts[i].y = (1.0 - s) * srcPts1[i].y + s * srcPts2[i].y;
	}
}


void GetTriangleVertices(const cv::Subdiv2D& sub_div, const std::vector<cv::Point2f>& points, std::vector<cv::Vec3i>& triangle_vertices)
{
	std::vector<cv::Vec6f> triangles;
	sub_div.getTriangleList(triangles);

	int num_triangles = triangles.size();
	triangle_vertices.clear();
	triangle_vertices.reserve(num_triangles);
	for(int i=0; i<num_triangles; i++){
		std::vector<cv::Point2f>::const_iterator vert1, vert2, vert3;
		vert1 = std::find(points.begin(), points.end(), cv::Point2f(triangles[i][0],triangles[i][1]));
		vert2 = std::find(points.begin(), points.end(), cv::Point2f(triangles[i][2],triangles[i][3]));
		vert3 = std::find(points.begin(), points.end(), cv::Point2f(triangles[i][4],triangles[i][5]));

		cv::Vec3i vertex;
		if(vert1 != points.end() && vert2 != points.end() && vert3 != points.end()){
			vertex[0] = vert1 - points.begin();
			vertex[1] = vert2 - points.begin();
			vertex[2] = vert3 - points.begin();
			triangle_vertices.push_back(vertex);
		}
	}
}


void TransTrianglerPoints(const std::vector<cv::Vec3i>& triangle_vertices,
	const std::vector<cv::Point2f>& points,  
	std::vector<std::vector<cv::Point2f>>& triangler_pts)
{
	int num_triangle = triangle_vertices.size();
	triangler_pts.resize(num_triangle);
	for(int i=0; i<num_triangle; i++){
		std::vector<cv::Point2f> triangle;
		for(int j=0; j<3; j++){
			triangle.push_back(points[triangle_vertices[i][j]]);
		}
		triangler_pts[i] = triangle;
	}
}


void PaintTriangles(cv::Mat& img, const std::vector<std::vector<cv::Point2f>>& triangles)
{
	int num_triangle = triangles.size();

	for(int i=0; i<num_triangle; i++){
		std::vector<cv::Point> poly(3);

		for(int j=0;j<3;j++){
			poly[j] = cv::Point(cvRound(triangles[i][j].x), cvRound(triangles[i][j].y));
		}
		cv::fillConvexPoly(img, poly,  cv::Scalar(i+1));
	}
}

///// for debug /////
void DrawTriangles(cv::Mat& img, const std::vector<std::vector<cv::Point2f>>& triangles)
{
	int num_triangle = triangles.size();

	std::vector<std::vector<cv::Point>> polies;
	for(int i=0; i<num_triangle; i++){
		std::vector<cv::Point> poly(3);

		for(int j=0;j<3;j++){
			poly[j] = cv::Point(cvRound(triangles[i][j].x), cvRound(triangles[i][j].y));
		}
		polies.push_back(poly);
	}
	cv::polylines(img, polies, true, cv::Scalar(255,0,255));
}
//////////////////////

void SolveHomography(const std::vector<cv::Point2f>& src_pts1, const std::vector<cv::Point2f>& src_pts2, cv::Mat& H)
{
	assert(src_pts1.size() == src_pts2.size());

	H = PointVec2HomogeneousMat(src_pts2) * PointVec2HomogeneousMat(src_pts1).inv();
}


void SolveHomography(const std::vector<std::vector<cv::Point2f>>& src_pts1, 
	const std::vector<std::vector<cv::Point2f>>& src_pts2,
	std::vector<cv::Mat>& Hmats)
{
	assert(src_pts1.size() == src_pts2.size());
	
	int pts_num = src_pts1.size();
	Hmats.clear();
	Hmats.reserve(pts_num);
	for(int i=0; i<pts_num; i++){
		cv::Mat H;
		SolveHomography(src_pts1[i], src_pts2[i], H);
		Hmats.push_back(H);
	}
}


// Morph homography matrix
void MorphHomography(const cv::Mat& Hom, cv::Mat& MorphHom1, cv::Mat& MorphHom2, float blend_ratio)
{
	cv::Mat invHom = Hom.inv();
	MorphHom1 = cv::Mat::eye(3,3,CV_32FC1) * (1.0 - blend_ratio) + Hom * blend_ratio;
	MorphHom2 = cv::Mat::eye(3,3,CV_32FC1) * blend_ratio + invHom * (1.0 - blend_ratio);
}



// Morph homography matrix
void MorphHomography(const std::vector<cv::Mat>& Homs,
	std::vector<cv::Mat>& MorphHoms1,
	std::vector<cv::Mat>& MorphHoms2, 
	float blend_ratio)
{
	int hom_num = Homs.size();
	MorphHoms1.resize(hom_num);
	MorphHoms2.resize(hom_num);
	for(int i=0; i<hom_num; i++){
		MorphHomography(Homs[i], MorphHoms1[i], MorphHoms2[i], blend_ratio);
	}
}


// create a map for cv::remap()
void CreateMap(const cv::Mat& TriangleMap, const std::vector<cv::Mat>& HomMatrices, cv::Mat& map_x, cv::Mat& map_y)
{
	assert(TriangleMap.type() == CV_32SC1);

	// Allocate cv::Mat for the map
	map_x.create(TriangleMap.size(), CV_32FC1);
	map_y.create(TriangleMap.size(), CV_32FC1);

	// Compute inverse matrices
	std::vector<cv::Mat> invHomMatrices(HomMatrices.size());
	for(int i=0; i<HomMatrices.size(); i++){
		invHomMatrices[i] = HomMatrices[i].inv();
	}

	for(int y=0; y<TriangleMap.rows; y++){
		for(int x=0; x<TriangleMap.cols; x++){
			int idx = TriangleMap.at<int>(y,x)-1;
			if(idx >= 0){
				cv::Mat H = invHomMatrices[TriangleMap.at<int>(y,x)-1];
				float z = H.at<float>(2,0) * x + H.at<float>(2,1) * y + H.at<float>(2,2);
				if(z==0)
					z = 0.00001;
				map_x.at<float>(y,x) = (H.at<float>(0,0) * x + H.at<float>(0,1) * y + H.at<float>(0,2)) / z;
				map_y.at<float>(y,x) = (H.at<float>(1,0) * x + H.at<float>(1,1) * y + H.at<float>(1,2)) / z;
			}
			else{
				map_x.at<float>(y,x) = x;
				map_y.at<float>(y,x) = y;
			}
		}
	}
}


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
{
	// Input Images
	cv::Mat SrcImg[2];
	SrcImg[0] = src_img1;
	SrcImg[1] = src_img2;

	// Input Points
	std::vector<cv::Point2f> SrcPoints[2];
	SrcPoints[0].insert(SrcPoints[0].end(), src_points1.begin(), src_points1.end());
	SrcPoints[1].insert(SrcPoints[1].end(), src_points2.begin(), src_points2.end());

	// Add 4 corner points of image to the points
	cv::Size img_size[2];
	for(int i=0; i<2; i++){
		img_size[i] = SrcImg[i].size();
		float w = img_size[i].width - 1;
		float h= img_size[i].height - 1;
		SrcPoints[i].push_back(cv::Point2f(0,0));
		SrcPoints[i].push_back(cv::Point2f(w,0));
		SrcPoints[i].push_back(cv::Point2f(0,h));
		SrcPoints[i].push_back(cv::Point2f(w,h));
	}

	// Morph points
	std::vector<cv::Point2f> MorphedPoints;
	MorphPoints(SrcPoints[0], SrcPoints[1], MorphedPoints, shape_ratio);

	// Generate Delaunay Triangles from the morphed points
	int num_points = MorphedPoints.size();
	cv::Size MorphedImgSize(MorphedPoints[num_points-1].x+1,MorphedPoints[num_points-1].y+1);
	cv::Subdiv2D sub_div(cv::Rect(0,0,MorphedImgSize.width,MorphedImgSize.height));
	sub_div.insert(MorphedPoints);

	// Get the ID list of corners of Delaunay traiangles.
	std::vector<cv::Vec3i> triangle_indices;
	GetTriangleVertices(sub_div, MorphedPoints, triangle_indices);

	// Get coordinates of Delaunay corners from ID list
	std::vector<std::vector<cv::Point2f>> triangle_src[2], triangle_morph;
	TransTrianglerPoints(triangle_indices, SrcPoints[0], triangle_src[0]);
	TransTrianglerPoints(triangle_indices, SrcPoints[1], triangle_src[1]);
	TransTrianglerPoints(triangle_indices, MorphedPoints, triangle_morph);

	// Create a map of triangle ID in the morphed image.
	cv::Mat triangle_map = cv::Mat::zeros(MorphedImgSize, CV_32SC1);
	PaintTriangles(triangle_map, triangle_morph);

	// Compute Homography matrix of each triangle.
	std::vector<cv::Mat> homographyMats, MorphHom[2];
	SolveHomography(triangle_src[0], triangle_src[1], homographyMats);
	MorphHomography(homographyMats, MorphHom[0], MorphHom[1], shape_ratio);

	cv::Mat trans_img[2];
	for(int i=0; i<2; i++){
		// create a map for cv::remap()
		cv::Mat trans_map_x, trans_map_y;
		CreateMap(triangle_map, MorphHom[i], trans_map_x, trans_map_y);

		// remap
		cv::remap(SrcImg[i], trans_img[i], trans_map_x, trans_map_y, cv::INTER_LINEAR); 
	}

	// Blend 2 input images
	float blend = (color_ratio < 0) ? shape_ratio : color_ratio;
	dst_img = trans_img[0] * (1.0 - blend) + trans_img[1] * blend;

	dst_points.clear();
	dst_points.insert(dst_points.end(), MorphedPoints.begin(), MorphedPoints.end() - 4);

}