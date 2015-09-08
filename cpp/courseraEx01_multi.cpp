#include <iostream>
#include <fstream>
#include <vector>

#include <Eigen/Dense>

#include "funcsEx01.h"

using namespace std;

int main(){

  ifstream myfile;
  myfile.open("../ex1data2.txt");

  vector<double> vX{};
  vector<double> vy{};

  double d1, d2, d3;
  char c1, c2;

  if(myfile.is_open()){
    while(myfile>>d1>>c1>>d2>>c2>>d3){
      vX.push_back(d1);
      vX.push_back(d2);
      vy.push_back(d3);
    }
  }

  int m = vy.size();

  cout<<"Running Gradient Descent ...\n";

  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> X;
  Eigen::Matrix<double,Eigen::Dynamic, 1> y;
  Eigen::Matrix<double,1,Eigen::Dynamic> mu, sigma;

  X.resize(m,2);
  y.resize(m,1);

  for(int i = 0; i < m; ++i){
    X(i,0) = vX[2*i];
    X(i,1) = vX[2*i+1];
    y(i) = vy[i];
  }

  standardDev(X, mu, sigma);
  featureNormalize(X, mu, sigma);

  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Xcopy = X;
  X.resize(m,3);
  X<<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>::Constant(m,1,1.), Xcopy;


  Eigen::Matrix<double, Eigen::Dynamic, 1> theta = Eigen::MatrixXd::Zero(3,1);

  // Choose some alpha value
  double alpha = 1;
  int iterations = 200;


  gradientDescent(X, y, theta, alpha, iterations);

  cout<<"Theta found by gradient descent: "<<endl;
  cout<<theta<<endl<<endl;

  Eigen::Matrix<double, 1, 3> predVec;
  predVec<<1, (1650-mu(0))/sigma(0), (3-mu(1))/sigma(1);
  double price = predVec*theta;

  cout<<"Predicted price of a 1650 sq-ft, 3 br house using gradient descent"<<endl<<price<<endl;

  return 0;
}
