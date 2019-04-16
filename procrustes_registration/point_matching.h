#include <iostream> 
#include "Eigen/Core"
#include "Eigen/SVD"
#include "Eigen/Eigenvalues"

template<typename TYPE>
int point_matching(const Eigen::Matrix<TYPE, Eigen::Dynamic, Eigen::Dynamic>& A, \
	const Eigen::Matrix<TYPE, Eigen::Dynamic, Eigen::Dynamic>& B, \
	Eigen::Matrix < TYPE, Eigen::Dynamic, Eigen::Dynamic> &T, \
	Eigen::Matrix < TYPE, Eigen::Dynamic, Eigen::Dynamic> &t) {
	/*
	A: Moving matrix (num_landmark x ndims)
	B: Fixed matrix (num_landmark x ndims)
	T: rotation matrix (ndims x ndims)
	t: translation matrix(vector) (ndims x 1)
	*/

	static const int num_landmark = (int)A.rows();
	static const int ndims = (int)A.cols();

	Eigen::MatrixXd S(ndims, ndims);
	Eigen::MatrixXd j = Eigen::MatrixXd::Ones(num_landmark, 1);
	Eigen::MatrixXd I = Eigen::MatrixXd::Identity(num_landmark, num_landmark);

	/// calc S
	S = A.transpose() * (I - (j*j.transpose() / (double)num_landmark)) * B;
	
	/// singular value decomposition
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);	

	T = svd.matrixU() * svd.matrixV().transpose();

	t = (1 / (double)num_landmark) * (B - A*T).transpose() * j;

	/// decision mirror image
	std::cout << "decision mirror image" << std::endl;
	std::cout << "det = " << (svd.matrixU() * svd.matrixV().transpose()).determinant() << std::endl;
	if ((svd.matrixU() * svd.matrixV().transpose()).determinant() < 0) {
		return 0;
	}
	else {
		return 1;
	}
}