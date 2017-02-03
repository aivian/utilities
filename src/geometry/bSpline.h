#ifndef BSPLINE_H
#define BSPLINE_H

// random C++ stuff
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <math.h>
#include <iostream>

// ros stuff
#include <ros/ros.h>
#include <ros/console.h>

// eigen
#include <Eigen/Dense.h>

class bSpline {
private:
    Matrix<double, Dynamic, 1> interior_knots;
    Matrix<double, Dynamic, 1> boundary_knots;
    Matrix<double, Dynamic, 1> knots;
    double order;
    Matrix<double, Dynamic, 1> coordinates;

public:
    bSpline();
    virtual Matrix<double, Dynamic, Dynamic> 
        basis(Matrix<double, Dynamic, 1> x);
    virtual Matrix<double, Dynamic, 1> 
        eval(Matrix<double, Dynamic, 1> x);
}

class TensorProductSpline : public bSpline{
private:
    bSpline* splines;
    
public:
    TensorProductSpline();
    Matrix<double, Dynamic, Dynamic> 
        basis(Matrix<double, Dynamic, Dynamic> x);
    Matrix<double, Dynamic, Dynamic>
        eval(Matrix<double, Dynamic, Dynamic> x);
}
#endif
