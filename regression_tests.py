#usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model # for inbuilt regression function
from sklearn import cross_validation
from sklearn.cross_validation import KFold

def multi_variate_dual_solution(val,m):
	Mat_x=np.column_stack((val[0:m,0],val[0:m,1]))
	Mat_x=np.asmatrix(Mat_x)	
#	type(Mat_x)
#	print len(Mat_x)
#	print len(Mat_x[0])
	Mat_g= (Mat_x * Mat_x.transpose())
	#print len(Mat_g)
	#print len(Mat_g[0])
	print np.linalg.inv(Mat_x * Mat_x.transpose())		
	
#	vec_alpha=np.linalg.solve(Mat_X,vec_y.transpose())
#	print len(Mat_g)
#	print len(Mat_g[0])
	return();

def multi_variate_primal_solution(val,m):
	z=np.ones((m,1))
	sq_err=0
###################multiple feature linear model#################################	
	Mat_z=np.column_stack((z,val[0:m,0],val[0:m,1])) #slices 0-m elements from col 0 and col 1 of val and stacks it with the vector of ones to get a z matrix of m values
	#print vec_z[m-1]
	vec_y=val[0:m,2]
	vec_th=np.dot(np.linalg.pinv(Mat_z),vec_y)	
	#print vec_th
	#print vec_th[0]
	#print val[1][0]
	#print (vec_th[0]*val[1][0])
##################Calculating training efficiency############################
	print "training error"
	for i in range (0,m):
		#print val[i][0]
		y_hat=(vec_th[0]+(vec_th[1]*val[i][0]) +(vec_th[2]*val[i][1]) )
		sq_err+=(y_hat - val[i][2])*(y_hat - val[i][2])
	mean_sq_error=sq_err/m
	print mean_sq_error
##################Calculating testing efficiency############################
	sq_err=0
	print "testing error"
	for i in range (m,2500):
#		print (val[i][0]*vec_th[1])		
		y_hat=(vec_th[0]+(vec_th[1]*val[i][0]) +(vec_th[2]*val[i][1]) )
		sq_err+=(y_hat - val[i][2])*(y_hat - val[i][2])
	mean_sq_error=sq_err/m
	print mean_sq_error

	return();
	





def single_variate_linear_model(val,m):
	sq_err,sum_x , sum_y,sum_xy,  sum_x2= 0,0,0,0,0;
	for i in range(0,m):
		x=val[i][0]
		y=val[i][1]
		plt.scatter(x,y)
		
		sum_x+=x
		sum_y+=y
		sum_xy+=(x*y)
		sum_x2+=(x*x)
	#print sum_y
	#print sum_xy	
	plt.show()


###################single feature linear model#################################
	Mat_X=np.matrix( ((m, sum_x), (sum_x, sum_x2)) )
	vec_y=np.matrix( ((sum_y ), (sum_xy) ) )
	#print Mat_X
	#print vec_y
	vec_th=np.linalg.solve(Mat_X,vec_y.transpose())
	#print vec_th
##################Calculating training efficiency############################
	for i in range(0,m):
		x=val[i][0]
		y=val[i][1]
		y_hat=(vec_th[0]+(vec_th[1]*x))
		plt.scatter(x,y_hat)
		sq_err+=(y_hat - y)*(y_hat - y)
	plt.plot(x,y_hat)
	plt.show()
	print "training error " 
	mean_sq_error=sq_err/m
	print mean_sq_error
	
##################Calculating testing efficiency############################
	sq_err=0
	print "testing error"
	for i in range(m,200):
		x=val[i][0]
		y=val[i][1]
		y_hat=(vec_th[0]+(vec_th[1]*x))
		sq_err+=(y_hat - y)*(y_hat - y)
	mean_sq_error=sq_err/(200-m)
	print mean_sq_error
####################Linear regression with inbuilt function ########################
	x_train=val[0:m,0]
	y_train=val[0:m,1]
	x_test=val[m:200,0]
	y_test=val[m:200,1]

	regr = linear_model.LinearRegression()
	regr.fit(x_train[:, np.newaxis], y_train)

	print("Mean squares error with inbuilt function: %.2f" % np.mean((regr.predict(x_test[:, np.newaxis]) - y_test) ** 2))

	plt.scatter(x_test, y_test,  color='blue')
	plt.plot(x_test[:, np.newaxis], regr.predict(x_test[:, np.newaxis]), color='red')

	plt.show()

	return();

####################single feature polynomial model (degree 2) ########################
def single_variate_polynomial_model(val,m):
	sq_err,sum_x , sum_y,sum_xy,  sum_x2= 0,0,0,0,0;
	print("single feature polynomial model of degree 2")
	z=np.ones((m,1))
	Mat_z=np.column_stack((z,val[0:m,0],val[0:m,0]*val[0:m,0])) #slices 0-m elements from col 0 and col 1 of val and stacks it with the vector of ones to get a z matrix of m values
	#print Mat_z[0]
	vec_y=val[0:m,1]
	vec_th=np.dot(np.linalg.pinv(Mat_z),vec_y)	
	print vec_th
##################Calculating training efficiency############################
	for i in range(0,m):
		x=val[i][0]
		y=val[i][1]
		y_hat=(vec_th[0]+(vec_th[1]*x)+(vec_th[2]*x*x))
		plt.scatter(x,y_hat)
		sq_err+=(y_hat - y)*(y_hat - y)
	plt.plot(x,y_hat)
	plt.show()
	print "training error " 
	mean_sq_error=sq_err/m
	print mean_sq_error
	
##################Calculating testing efficiency############################
	sq_err=0
	print "testing error"
	for i in range(m,200):
		x=val[i][0]
		y=val[i][1]
		y_hat=(vec_th[0]+(vec_th[1]*x)+(vec_th[2]*x*x))
		sq_err+=(y_hat - y)*(y_hat - y)
	mean_sq_error=sq_err/(200-m)
	print mean_sq_error
	

####################single feature polynomial model (degree 3) ########################
	print ("single feature polynomial model of degree 3")	
	Mat_z=np.column_stack((z,val[0:m,0],val[0:m,0]*val[0:m,0],val[0:m,0]*val[0:m,0]*val[0:m,0])) #slices 0-m elements from col 0 and col 1 of val and stacks it with the vector of ones to get a z matrix of m values
	#print Mat_z[0]
	vec_y=val[0:m,1]
	vec_th=np.dot(np.linalg.pinv(Mat_z),vec_y)	
	print vec_th
##################Calculating training efficiency############################
	for i in range(0,m):
		x=val[i][0]
		y=val[i][1]
		y_hat=(vec_th[0]+(vec_th[1]*x)+(vec_th[2]*x*x)+(vec_th[3]*x*x*x))
		plt.scatter(x,y_hat)
		sq_err+=(y_hat - y)*(y_hat - y)
	plt.plot(x,y_hat)
	plt.show()
	print "training error " 
	mean_sq_error=sq_err/m
	print mean_sq_error
	
##################Calculating testing efficiency############################
	sq_err=0
	print "testing error"
	for i in range(m,200):
		x=val[i][0]
		y=val[i][1]
		y_hat=(vec_th[0]+(vec_th[1]*x)+(vec_th[2]*x*x)+(vec_th[3]*x*x*x))
		sq_err+=(y_hat - y)*(y_hat - y)
	mean_sq_error=sq_err/(200-m)
	print mean_sq_error
	

	return();
	
#"""
def single_variate_linear_validation(val):
	X=val[:,0]
	y=val[:,1]
	#X = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
	#y = np.array([1, 2, 3, 4])
	kf = KFold(200, n_folds=10)
	#print(kf)
	sq_err,avg_mean_sq_error=0,0  
	for train_index, test_index in kf:
		#print("TRAIN:", train_index, "TEST:", test_index)
		x_train, x_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		#print x_test
		sum_x=np.sum(x_train)
		sum_y=np.sum(y_train)
		sum_x2=np.sum(np.square(x_train))
		sum_xy=np.sum(x_train*y_train)
		vec_y=np.matrix( ((sum_y ), (sum_xy) ) )	
		Mat_X=np.matrix( ((len(x_train), sum_x), (sum_x, sum_x2)) )
		vec_th=np.linalg.solve(Mat_X,vec_y.transpose())
		y_hat=(vec_th[0]+(vec_th[1]*x_test))
	#	print y_hat
	#	print y_test
		sq_err=np.sum(np.square(np.subtract(y_hat, y_test)))
		#print sq_err
		#print "testing error " 
		mean_sq_error=sq_err/len(y_test)
		#print mean_sq_error
		avg_mean_sq_error+=mean_sq_error
	avg_mean_sq_error/=10
	print avg_mean_sq_error
	return();


###############_____MAIN____#######################
print "This is Single variable dataset 1"
val=np.loadtxt('svar-set1.dat', skiprows=5, delimiter=' ', usecols=(1,2))
single_variate_linear_model(val,160)
single_variate_linear_validation(val)
single_variate_polynomial_model(val,160)


print "Single variable dataset 2"
val=np.loadtxt('svar-set2.dat', skiprows=5, delimiter=' ', usecols=(1,2))
single_variate_linear_model(val,160)
single_variate_linear_validation(val)
single_variate_polynomial_model(val,160)


print "Single variable dataset 3"
val=np.loadtxt('svar-set3.dat', skiprows=5, delimiter=' ', usecols=(1,2))
single_variate_linear_model(val,160)
single_variate_polynomial_model(val,80)
print "Single variable dataset 4"
val=np.loadtxt('svar-set4.dat', skiprows=5, delimiter=' ', usecols=(1,2))
single_variate_linear_model(val,160)

"""
print "Multi variable dataset 1"
val=np.loadtxt('mvar-set1.dat', skiprows=5, delimiter=' ', usecols=(1,2,3))
#multi_variate_primal_solution(val,1150)
#multi_variate_dual_solution(val,1150)
#Mat_x=val[:2000,:-1]
#Mat_g= np.dot(Mat_x, Mat_x.transpose())
#print Mat_g.shape
#print np.linalg.inv(Mat_g)		

print "Multi variable dataset 2"
val=np.loadtxt('mvar-set2.dat', skiprows=5, delimiter=' ', usecols=(1,2,3))
multi_variate_primal_model(val,2000)

print "Multi variable dataset 3"
val=np.loadtxt('mvar-set3.dat', skiprows=5, delimiter=' ', usecols=(1,2,3,4,5,6))
multi_variate_primal_model(val,80000)

print "Multi variable dataset 4"
val=np.loadtxt('mvar-set3.dat', skiprows=5, delimiter=' ', usecols=(1,2,3,4,5,6))
#multi_variate_primal_model(val,80000)

"""



