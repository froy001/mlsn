#include <iostream>
#include <fstream>
#include <vector>

#include <Eigen/Dense>

#include "funcsEx01.h"

using namespace std;

int main(){

  ifstream myfile;
  myfile.open("../ex1data1.txt");

  vector<double> vX{};
  vector<double> vy{};

  double d1, d2;
  char c;

  if(myfile.is_open()){
    while(myfile>>d1>>c>>d2){
      vX.push_back(d1);
      vy.push_back(d2);
    }
  }

  int m = vy.size();

  cout<<"Running Gradient Descent ...\n";

  Eigen::Matrix<double,Eigen::Dynamic,2> X;
  Eigen::Matrix<double,Eigen::Dynamic,1> y;

  X.resize(m,2);
  y.resize(m,1);

  for(int i = 0; i < m; ++i){
    X(i,0) = 1;
    X(i,1) = vX[i];
    y(i) = vy[i];
  }

  Eigen::Matrix<double, Eigen::Dynamic, 1> theta = Eigen::MatrixXd::Zero(2,1);

  // Some gradient descent settings
  int iterations = 1500;
  double alpha = 0.01;

  //  gradientDescent(X, y, theta, alpha, iterations);
  gradientDescent(X, y, theta, alpha, iterations);

  cout<<"Theta found by gradient descent: "<<endl;
  cout<<theta<<endl<<endl;

  Eigen::Matrix<double, 1, Eigen::Dynamic> predVec;
  predVec.resize(1,theta.rows());
  predVec<<1, 3.5;
  double predict1 = predVec*theta;

  predVec(1) = 7;
  double predict2 = predVec*theta;

  cout<<"For population = 35,000, we predict a profit of "<<predict1*10000<<endl;
  cout<<"For population = 70,000, we predict a profit of "<<predict2*10000<<endl<<endl;
  return 0;
}
