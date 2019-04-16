#include<iostream>
#include<vector>
#include<fstream>
#include<algorithm>
#include"Eigen/Core"
#include"ItkImageIO.h"
#include"point_matching.h"
#include"read_csv.h"
#include"cubic_interpolate.h"

#define MAX_NUM_CASE 20
#define MAX_NUM_LANDMARK 100
#define NDIMS 3


int main(int argc, char **argv) {

	if (argc != 5) {
		std::cerr << "Usage : " << std::endl;
		std::cerr << argv[0] << std::endl;
		std::cerr << " inputFile(.mhd) randmarkFile(.csv) outputFile(.mhd) outputCsvFile(.csv)" << std::endl;
		exit(1);
	}

	// temporary array
	double A2[MAX_NUM_CASE][MAX_NUM_LANDMARK][NDIMS];

	/*********************************************************************/
	/// Read csv file to A2 array as int
	/*********************************************************************/
	std::string filename = std::string(argv[2]);
	vector<vector<string>> table;
	if (!GetContents(filename, table)) {
		std::cout << "csv read error" << std::endl;
		return -1;
	}
	// Assume first row in csv file is number of landmark
	const int num_landmark = std::stod(table[0][0]);
	std::cout << "num_landmark: " << num_landmark << std::endl;

	int num_case = 0;
	int check_count = 0;
	for (int row = 1; row < table.size(); row++) {
		vector<string> record; // 1 row
		record = table[row];
		for (int column = 0; column < record.size(); column++) {
			A2[num_case][check_count][column] = std::stod(record[column]);
		}
		check_count++;
		if (check_count == num_landmark) {
			num_case++;
			check_count = 0;
		}
	}
	std::cout << "Input # Case: " << num_case << std::endl;

	/*********************************************************************/
	/// Procrustes
	/*********************************************************************/
	Eigen::MatrixXd A = Eigen::MatrixXd::Zero(num_landmark, NDIMS);/// Moving matrix, 
	Eigen::MatrixXd B = Eigen::MatrixXd::Zero(num_landmark, NDIMS);/// fixed matrix
	Eigen::MatrixXd T(NDIMS, NDIMS); // rotation matrix
	Eigen::MatrixXd t(NDIMS, 1);     // translation vector
	Eigen::MatrixXd j = Eigen::MatrixXd::Ones(num_landmark, 1);
	Eigen::MatrixXd I = Eigen::MatrixXd::Identity(num_landmark, num_landmark); // identity matrix

	static const double low_reso[NDIMS] = { 0.429688, 0.429688, 0.5 };
	static const double high_reso[NDIMS] = { 0.07, 0.066, 0.07 };

	/*static const double low_reso[NDIMS] = { 2, 2, 2 };
	static const double high_reso[NDIMS] = { 1, 1, 1};*/

	/// Assume that Case 0 is Fixed image's landmark(high resolustion)
	for (int i = 0; i < num_landmark; i++) {
		for (int j = 0; j < NDIMS; j++) {
			B(i, j) = A2[0][i][j] * high_reso[j];
		}
	}
	/// Assume that Case 1 is Moving image's landmark(low resolution)
	for (int i = 0; i < num_landmark; i++) {
		for (int j = 0; j < NDIMS; j++) {
			A(i, j) = A2[1][i][j] * low_reso[j];
		}
	}

	/// calculate squared error between landmark before registration
	double SSD = 0.;
	for (int i = 0; i < num_landmark; i++) {
		for (int j = 0; j < NDIMS; j++) {
			SSD += (A(i, j) - B(i, j)) * (A(i, j) - B(i, j));
		}
	}
	std::cout << "squared error per randmark between landmark before registration: " << SSD / (double)num_landmark << std::endl;

	/// start registaration
	point_matching(A, B, T, t);

	std::cout << "Rotation matrix: T:" << std::endl;
	std::cout << T << std::endl;
	std::cout << "Translation vector: t:" << std::endl;
	std::cout << t << std::endl;

	/// calculate squared error between landmark after registration
	std::ofstream fs("SSD_per_landmark.csv");
	fs << "landmark_number" << "," << "Squared error" << std::endl;
	SSD = 0.;
	for (int i = 0; i < num_landmark; i++) {
		Eigen::RowVector3d temp = A.row(i);
		temp = temp * T + t.transpose();

		double sum = 0.;
		for (int j = 0; j < NDIMS; j++) {
			sum += (temp(j) - B(i, j)) * (temp(j) - B(i, j));
			SSD += (temp(j) - B(i, j))*(temp(j) - B(i, j));
		}
		fs << i + 1 << "," << sum << std::endl;
	}
	fs.close();
	std::cout << "squared error per randmark between landmark after registration: " << SSD / (double)num_landmark << std::endl;

	std::ofstream fs1(argv[4]);
	fs1 << "SSD: ," << SSD / (double)num_landmark << std::endl;
	fs1 << "Rotation Matrix:," << std::endl;
	for (int j = 0; j < T.rows(); j++) {
		for (int i = 0; i < T.cols(); i++) {
			fs1 << T(j, i) << std::endl;
		}
	}
	fs1 << "Translation Vector:," << std::endl;
	fs1 << t << std::endl;
	fs1.close();

	/*********************************************************************/
	/// Conversion process
	/*********************************************************************/
	std::vector<short> img;
	ImageIO<3> mhdi;
	mhdi.Read(img, argv[1]);
	int xe = mhdi.Size(0);
	int ye = mhdi.Size(1);
	int ze = mhdi.Size(2);
	int se = xe * ze * ye;
	std::cout << "Input Size: ";
	std::cout << xe << " " << ye << " " << ze << " " << se << std::endl;
	std::cout << "Output Size: ";
	std::cout << xe << " " << ye << " " << ze << " " << se << std::endl;

	std::vector<short> output(se, 0);
	Eigen::RowVector3d output_point, input_point;

	for (int z = 0; z < ze; z++) {
		output_point(2) = (double)z*low_reso[2];

		for (int y = 0; y < ye; y++) {
			output_point(1) = (double)y*low_reso[1];

			for (int x = 0; x < xe; x++) {
				output_point(0) = (double)x*low_reso[0];

				input_point = (output_point - t.transpose()) * T.transpose();
				//input_point = output_point*T + t.transpose(); //miss
				input_point(0) /= low_reso[0];
				input_point(1) /= low_reso[1];
				input_point(2) /= low_reso[2];
				//std::cout << input_point << std::endl;
				// cubic interpolate
				double cubic[4][4][4]; //p[x][y][z]
				for (int k = -1; k < 3; k++) {
					for (int j = -1; j < 3; j++) {
						for (int i = -1; i < 3; i++) {
							int xi = (int)(input_point(0)) + i;
							int yi = (int)(input_point(1)) + j;
							int zi = (int)(input_point(2)) + k;
							int ss = xe * (std::max(0, std::min(ze - 1, zi)) * ye
								+ std::max(0, std::min(ye - 1, yi)))
								+ std::max(0, std::min(xe - 1, xi));
							cubic[i + 1][j + 1][k + 1] = (double)img[ss];
							//std::cout << cubic[i + 1][j + 1][k + 1] << std::endl;
						}
					}
				}
				input_point(0) = input_point(0) >= 0 ? std::fabs(input_point(0)) - std::floor(std::fabs(input_point(0))) : std::ceil(std::fabs(input_point(0))) - std::fabs(input_point(0));
				input_point(1) = input_point(1) >= 0 ? std::fabs(input_point(1)) - std::floor(std::fabs(input_point(1))) : std::ceil(std::fabs(input_point(1))) - std::fabs(input_point(1));
				input_point(2) = input_point(2) >= 0 ? std::fabs(input_point(2)) - std::floor(std::fabs(input_point(2))) : std::ceil(std::fabs(input_point(2))) - std::fabs(input_point(2));
				//std::cout << input_point << std::endl;
				double var = tricubicInterpolate(cubic, input_point(0), input_point(1), input_point(2));
				output[z * xe * ye + y * xe + x] = (short)(var + 0.5);
			}
		}
	}
	mhdi.Write(output, argv[3]);

	return 0;

}