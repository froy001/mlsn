#include <Eigen/Dense>
#include <cmath>

void gradientDescent(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& X, const Eigen::Matrix<double, Eigen::Dynamic, 1>& y, Eigen::Matrix<double, Eigen::Dynamic, 1>& theta, double alpha, int iterations){

  int m = X.rows();
  int n = X.cols();

  for(int i = 0; i < iterations; ++i){
    /*    double temp0 = theta(0)-alpha*(X*theta-y).sum()/m;
    Eigen::Matrix<double, Eigen::Dynamic,1> temp = -alpha*((X*theta-y).transpose()*X.col(1))/m;
    double temp1 = theta(1)+temp(0);
    */

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> thetaTemp = theta;

    for(int j = 0; j < n; ++j)
      thetaTemp(j) += alpha*((y-X*theta).array()*X.col(j).array()).sum()/m;

    theta = thetaTemp;
  }
}

void standardDev(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& X, Eigen::Matrix<double, 1, Eigen::Dynamic>& mu, Eigen::Matrix<double, 1, Eigen::Dynamic>& sigma){
  int m = X.rows();
  int n = X.cols();
  mu.resize(1, n);
  sigma.resize(1, n);
  for(int i = 0; i < n; ++i){
    mu(i) = X.col(i).sum()/m;
    Eigen::Array<double, Eigen::Dynamic, 1> Xi = X.col(i).array()-mu(i);
    sigma(i) = sqrt(((Xi.array()*Xi.array())/(m-1)).sum());
  }
}

void featureNormalize(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& X, Eigen::Matrix<double, 1, Eigen::Dynamic>& mu, Eigen::Matrix<double, 1, Eigen::Dynamic>& sigma){
  for(int i = 0; i < X.cols(); ++i)
    X.col(i) = (X.col(i).array()-mu(i))/sigma(i);
}
