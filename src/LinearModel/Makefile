Test: test.cc LinearRegression.o
	g++ -std=c++14  LinearRegression.o test.cc -o Test -O2 -larmadillo
LinearRegression.o: LinearRegression.cc LinearRegression.h
	g++ -std=c++14 -c LinearRegression.cc -o LinearRegression.o -O2 -larmadillo
