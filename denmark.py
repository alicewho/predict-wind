import pandas as pd
import numpy as np
import tkinter 
import matplotlib.pyplot as plt
import re 
import pylab
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from scipy import stats
from numpy.linalg import norm
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.pyplot import quiver
from pandas.plotting import scatter_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score

# read 1 file:
def read_fn(fn):
	with open(fn, 'r') as f:
		data = f.readlines()
	return data

# parse a variable (from one file)
def parse_var(dat):
	# define the regular expression for replacing
	regex = re.compile('^a-zA-Z0-9.') 
	l = []
	for i in range(len(dat)): # iter through each day (one JSON array in {})
		subd = dat[i][1:-2] # gets rid of curly brackets and newline character at ends
		subd = subd.split(',')
		subl = []
		for j in range(len(subd)): # iter through each "timestamp" : "reading" entry, comma separated
			ssd = subd[j].split(': ') # only affects : btw "timestamp" and "reading"
			k = regex.sub('', ssd[0]).replace('\"','') # ssd[0] is "timestamp"
			v = regex.sub('', ssd[1]).replace('\"','') # ssd[1] is "reading"
			if(len(v) == 0): # no reading
			    v = np.nan # replace with nan
			else:
			    v = float(v) # convert real reading to float
			subl.append([np.datetime64(k), v]) # convert timestamp to np.datetime64 object
		subl = np.vstack(subl)
		l.append(subl)
	return np.vstack(l)
			

fns = ['/home/ahu/Documents/wind/dewptm.txt', '/home/ahu/Documents/wind/hum.txt', '/home/ahu/Documents/wind/pressurem.txt', '/home/ahu/Documents/wind/tempm.txt', '/home/ahu/Documents/wind/vism.txt','/home/ahu/Documents/wind/wspdm.txt','/home/ahu/Documents/wind/wdird.txt']


# iter thru fns:
finl = []
for i in range(len(fns)):
	fn = fns[i]
	# open file + parse:
	dat = read_fn(fn)
	dp = parse_var(dat)
	if(i==0): # only need datetime once; take from 0th file
		finl.append(dp[:,0]) # append datetime64 first
	finl.append(dp[:,1]) # then append readings
# repackage into big array:
finar = np.vstack(finl).T
dt_sorted = finar[finar[:,0].argsort()] # sort nparray by datetime64
print(dt_sorted,np.shape(dt_sorted))

# use pandas to create DataFrame
df = pd.DataFrame(dt_sorted,columns=['Timestamp','Dew_Point','Humidity','Pressure','Temperature','Visibility','Wind_Speed','Wind_Direction'])

# initial exploration
print(df.head())
print(df.info())
print(df.describe())

# scatter plot
# wind direction
x = dt_sorted[:,0]
y = dt_sorted[:,-1]
plt.scatter(x, y)
plt.show()

# wind speed
plt.figure()
plt.plot(dt_sorted[:,-2])
plt.show()

#### clean erroneous values ####
## wind speed ##

def countneg(col):
	neg = 0
	for i in range(len(col)):
		if col[i] < 0:
			neg += 1
	return neg
	
print(countneg(dt_sorted[:,-2])) 

def getind(col):
	ind = 0
	for i in range(len(col)):
		if col[i] < 0:
			ind = i
	return ind
	
print(getind(dt_sorted[:,-2])) 

# replace -9999 with NaN
dt_sorted[5509,-2] = np.nan 
print(dt_sorted[5509,-2])

## visibility ##

print(countneg(dt_sorted[:,-3])) 

# logical indexing
visboolarray = dt_sorted[:,-3] < 0 # numpy inequality returns bool array of same shape
print(visboolarray)
dt_sorted[visboolarray,-3] = np.nan
print(dt_sorted[7947,-3]) # check if worked

#### impute all nans ####

print(dt_sorted)
print(np.shape(dt_sorted))
ip = SimpleImputer()
imp = ip.fit_transform(dt_sorted[:,1:],y=None)
print(imp[5509,-2]) # check
print(imp)
print(np.shape(imp))

#### change pressure from mbar to psi (instead of scaling) ####

imp[:,2] = imp[:,2]*0.0145038
print(imp[:,2])

#### histograms ####
df_4hist = pd.DataFrame(imp,columns=['Dew Point','Humidity','Pressure','Temperature','Visibility','Wind Speed','Wind Direction'])
df_4hist.hist(bins=40,figsize=(10,7)) 
plt.show()

#### correlation matrix ####
print(df_4hist.corr())

#### scatter matrix ####

scatt = scatter_matrix(df_4hist, figsize=(15,10))
plt.show()

#### make 2-dim wind vector/target ####

def twodim(windspeed,winddir):
	rad = np.deg2rad(winddir) # convert degrees to radians
	x_comp = windspeed*np.cos(rad) 
	y_comp = windspeed*np.sin(rad)
	return x_comp, y_comp

targx, targy = twodim(imp[:,-2],imp[:,-1])

print(targx[:5])
print(targy[:5])

# plot x,y components of target
plt.figure() 
plt.subplot(2,1,1)
plt.plot(targx)
plt.subplot(2,1,2)
plt.plot(targy)
plt.show()

target = np.column_stack((targx,targy))
print(target)

#### make input matrix X ####

print(imp[:,:-2])

X = np.hstack((imp[:,:-2], target)) #X has target (rcos and rsin - not windspeed) 
print(np.shape(X))
print(X)


#### Make temporal offset #### 
offset = 1 
Xoff = X[:-1*offset,:]
targoff = target[offset:,:]


#### make test set ####

X_train, X_test, y_train, y_test = train_test_split(Xoff, targoff, test_size=0.2, random_state=42)

######################### One-step ahead OLS linear regression ##############################

#### train ####

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
ypred = lin_reg.predict(X_train)

# regularized version

enet = ElasticNet()
enet.fit(X_train,y_train)
ypred_enet = enet.predict(X_train)

## train plots ##

plt.figure()
plt.scatter(y_train[:,0], ypred[:,0])
slope, intercept, r_value0, p_value, std_err = stats.linregress(y_train[:,0], ypred[:,0])
plt.plot(np.unique(y_train[:,0]), np.poly1d(np.polyfit(y_train[:,0], ypred[:,0], 1))(np.unique(y_train[:,0])), c='k') # plot line of best fit
plt.legend(['r^2_train = 0.88'])
plt.show()
print(r_value0)

plt.figure()
plt.scatter(y_train[:,1], ypred[:,1])
slope, intercept, r_value1, p_value, std_err = stats.linregress(y_train[:,1], ypred[:,1])
plt.plot(np.unique(y_train[:,1]), np.poly1d(np.polyfit(y_train[:,1], ypred[:,1], 1))(np.unique(y_train[:,1])), c='k')
plt.legend(['r^2_train = 0.97'])
plt.show()
print(r_value1)

r2_train = r2_score(y_train,ypred)
print(r2_train) 

train_mse_xcomp = mean_squared_error(y_train[:,0], ypred[:,0])
train_mse_ycomp = mean_squared_error(y_train[:,1], ypred[:,1])

#### cross-validation ####

train_mse_xcomp_enet = mean_squared_error(y_train[:,0], ypred_enet[:,0])
train_mse_ycomp_enet = mean_squared_error(y_train[:,1], ypred_enet[:,1])

# Get cross-validation mse for desired model version and x/y-component
def cv_scores(model,component_data,target,no_folds):
    scores = cross_val_score(model,component_data,target,scoring = 'neg_mean_squared_error', cv = no_folds)
    return -scores

unreg_xcomp_cv_scores = cv_scores(lin_reg,X_train,y_train[:,0],10)    
print('unreg train mse x-comp = ',train_mse_xcomp)
print('unreg cv mse x-comp = ',unreg_xcomp_cv_scores.mean(),'+/-',unreg_xcomp_cv_scores.std(),'\n')

unreg_ycomp_cv_scores = cv_scores(lin_reg,X_train,y_train[:,1],10)    
print('unreg train mse y-comp = ',train_mse_ycomp)
print('unreg cv mse y-comp = ',unreg_ycomp_cv_scores.mean(),'+/-',unreg_ycomp_cv_scores.std(),'\n')

enet_xcomp_cv_scores = cv_scores(enet,X_train,y_train[:,0],10) 
print('elastic net train mse x-comp = ',train_mse_xcomp_enet)
print('elastic net cv mse x-comp = ',enet_xcomp_cv_scores.mean(),'+/-',enet_xcomp_cv_scores.std(),'\n')

enet_ycomp_cv_scores = cv_scores(enet,X_train,y_train[:,1],10) 
print('elastic net train mse y-comp = ',train_mse_ycomp_enet)
print('elastic net cv mse y-comp = ',enet_ycomp_cv_scores.mean(),'+/-',enet_ycomp_cv_scores.std())

#### test on test set ####

ypred_test = lin_reg.predict(X_test)
lin_mse = mean_squared_error(y_test, ypred_test)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse) 
print(max(y_test[:,0]) - min(y_test[:,0])) 
print(max(y_test[:,1]) - min(y_test[:,1])) 

r2_test = r2_score(y_test,ypred_test)
print(r2_test)

## test plots ##

plt.figure()
plt.scatter(y_test[:,0], ypred_test[:,0],c='#92e2ab')
slope, intercept, r_value3, p_value, std_err = stats.linregress(y_test[:,0], ypred_test[:,0])
plt.plot(np.unique(y_test[:,0]), np.poly1d(np.polyfit(y_test[:,0], ypred_test[:,0], 1))(np.unique(y_test[:,0])), c='k') # plot line of best fit
plt.legend(['r\u00b2 = 0.88'])
plt.title('Actual vs. Prediction, X-component')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()
print(r_value3) 

plt.figure()
plt.scatter(y_test[:,1], ypred_test[:,1],c='#92e2ab')
slope, intercept, r_value4, p_value, std_err = stats.linregress(y_test[:,1], ypred_test[:,1])
plt.plot(np.unique(y_test[:,1]), np.poly1d(np.polyfit(y_test[:,1], ypred_test[:,1], 1))(np.unique(y_test[:,1])), c='k')
plt.legend(['r\u00b2 = 0.97'])
plt.title('Actual vs. Prediction, Y-component')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()
print(r_value4) 

mean_r_value_test = (r_value3 + r_value4)/2 # get mean r^2 for test data
print(mean_r_value_test)

## get model coefficients ##

coeffs = lin_reg.coef_
print(np.shape(coeffs))

#### xy coordinates with magnitude (wndspd) ####

# scatter plot w/rotation for wind convention

u = ypred_test[:,1]
v = ypred_test[:,0]
plt.figure()
plt.scatter(v,u,c='#3339ff',linewidths=.0001)
plt.title('Predicted Wind Vector (test set)')
plt.xlabel('x-component')
plt.ylabel('y-component')
plt.grid(True,alpha=0.5)
plt.show()

# vector plot #

U = v # rotate to reflect wind convention
V = u
plt.figure()
plt.quiver(0,0,U,V,color='#3339ff',units='inches',scale=40,width=.005)
plt.title('Predicted Wind Vector')
plt.xlabel('x-component')
plt.ylabel('y-component')
plt.ylim([-.5,.5])
plt.xlim([-.02,.02])
plt.show()

viz = np.column_stack((x,y)) # with magnitude
print(x)
print('viz',viz)

## xy coords ON unit circle ##

x_hat = np.array([1, 0])
x_hat = x_hat.reshape(-1,1)
y_hat = np.array([0, 1])
y_hat = y_hat.reshape(-1,1)
x = ypred_test
num = np.dot(x,x_hat)
denom = np.reshape(np.linalg.norm(x,axis=1), (-1,1))
sig = np.sign(np.dot(x,y_hat))
theta = sig * np.arccos(num/denom)
theta_deg = np.rad2deg(theta)

# make all degrees positive
def degpos(data):
	for i in range(len(data)):
		if data[i] < 0:
			data[i] = 360 - abs(data[i])
	return data

print(theta_deg)

x_unit = np.sin(theta) 
y_unit = np.cos(theta)
viz2 = np.column_stack([x_unit,y_unit])
print(viz2)
plt.figure()
plt.scatter(x_unit, y_unit)
plt.show()

#### get wind speed back out ####

wndspd_pred_test = np.dot(ypred_test,x_hat)/y_unit

a2 = ypred_test[0,0]*ypred_test[0,0]
b2 = ypred_test[0,1]*ypred_test[0,1]
wndspd0 = np.sqrt(a2 + b2) 
print(wndspd0) 
print(wndspd_pred_test) # check if same

#### more plots ####

# input wind direction (angle) scatter plot

timestmp = dt_sorted[:,0]
angle = imp[:,-1]
plt.figure()
plt.scatter(timestmp,angle,c='#ff334f',linewidths=.0001)
plt.title('Input Wind Direction')
plt.xlabel('Time')
plt.ylabel('Angle (degrees)')
plt.show()


#### actual step ahead wind spd and direction (from y_test) ####

def gettheta(data):
	x_hat = np.array([1, 0])
	x_hat = x_hat.reshape(-1,1)
	y_hat = np.array([0, 1])
	y_hat = y_hat.reshape(-1,1)
	x = data
	num = np.dot(x,x_hat)
	denom = np.reshape(np.linalg.norm(x,axis=1), (-1,1))
	sig = np.sign(np.dot(x,y_hat))
	theta = sig * np.arccos(num/denom)
	theta_deg = np.rad2deg(theta)
	return theta_deg

print(gettheta(y_test))
actual_dir = gettheta(y_test)

actual_dir = degpos(actual_dir) 
pred_dir = degpos(theta_deg)

actual_dir_train = degpos(gettheta(y_train))
pred_dir_train = degpos(gettheta(ypred))

# plot actual vs. pred dir from test set
plt.figure()
plt.plot(pred_dir,label='Predicted')
plt.plot(actual_dir,label='Actual')
pylab.legend(loc='best')
plt.title('Actual vs. Predicted Wind Direction Test Set')
plt.xlabel('Time')
plt.ylabel('Wind Direction (deg)')
plt.xlim([0,72]) 
plt.show()

# plot actual vs. pred dir from training set
plt.figure()
plt.plot(pred_dir_train,label='Predicted')
plt.plot(actual_dir_train,label='Actual')
pylab.legend(loc='best')
plt.title('Actual vs. Predicted Wind Direction Training Set')
plt.xlabel('Time')
plt.ylabel('Wind Direction (deg)')
plt.xlim([0,72]) 
plt.show()

wndspd_test = abs(np.dot(y_test,x_hat)/y_unit)
print(wndspd_test)
print(imp[1:,-2])

actual = np.column_stack([wndspd_test,actual_dir])
print(actual)
print(np.shape(actual))

df_actual = pd.DataFrame(actual)
