import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import sys 
from integrator import *
import time
import pylab
import pylab as plt
from pylab import rc
from matplotlib import rc,rcParams
import matplotlib.gridspec as gridspec
label_size = 20
rc('text', usetex=True )
rc('font',family='serif',size= 15, weight = 'normal')
M_SIZE = 8



############## dealing with the argparser related stuff ##########################
## for th arguments with reuired=True do not used metavar=''!!
parser = argparse.ArgumentParser(description='Run the Code to simulate the timing noise of the pulsar population')

parser.add_argument('-f', metavar='', type=str, required =False, nargs='?', help='Supply an ascii file with PSR name, P and P_dot values in the col1, col2 and col3 respectively. Pass the file the other option is currently not available.', default='N')
parser.add_argument('-Nsamp', metavar='', type=int, required =False, help='Number of pulsars need to be sampled from the P - P dot diagram.')
parser.add_argument('-n', type=int, required=True, help='n can be 2 or 3. 2 implies 2 component fluid and 3 implies component fluid.')
parser.add_argument('-ainf', type=float, required=True, help='Alpha infity.')
parser.add_argument('-amf1', type=float, required=True, help='Alpha mutual friction of superfluid component 1.')
parser.add_argument('-amf2', metavar='', type=float, required=False, help='Alpha mutual friction of superfluid component 2.')
parser.add_argument('-B1', type=float, required=True, help='Coeff of mutual friction for superfluid component 1.')

parser.add_argument('-B2', metavar='', type=float, required=False, help='Coefficient of mutual friction for superfluid component 2.')
parser.add_argument('-x1', type=float, required=True, help='Fractional moment of inertia for the superfluid componenet 1.')
parser.add_argument('-x2',   metavar='', type=float, required=False, help='Fractional moment of inertia for the superfluid componenet 2.')
parser.add_argument('-nu0',  metavar='', type=float, required=False, help='Initial value of rotation frequency of the pulsar. You must pass this argument if you are not passing -f field.')
parser.add_argument('-nudot0', metavar='', type=float, required=False, help='Initial value of rotation frequency derivative of the pulsar. You must pass this argument if you are not passing -f field. You must pass the numerical value avoid writing the -ve sign.')
parser.add_argument('-dt',  metavar='', type=float, required=False, nargs='?', help='Time-steps in days. If not provided it will take 10 days.', default = 10.0)
parser.add_argument('-Ny',  metavar='', type=float, required=False, nargs='?', help='Time span over which the simulation will be performed in years. Therefore it will give Ny/dt data points. If not supplied then the default value is 20 yrs.', default = 20.0)
parser.add_argument('-e',  metavar='', type=int, required=False, nargs='?', help='Ensemble size, number of time the simulation will be repeated with the same parameters. If not supplied the default value is 1.', default = 1)

args = parser.parse_args()

TIMESTAMP = datetime.now().strftime("%Y:%m:%d-%H:%M:%S")

#############################################################
if args.f == 'N' and args.nu0 == None:
	print ('You must pass the value of -nu0:')
	sys.exit()
if args.f == 'N' and args.nudot0 == None:
	print ('You must pass the value of -nudot0')
	sys.exit()
	
############ Simulation over the pulsar population ############################
if args.f != 'N' and args.Nsamp == None:
	print ('You must provide the number of pulsars required to be sampled from the P-P_dot diagram.')
	sys.exit()

if args.f != 'N':
	PPDOT = np.loadtxt(args.f, dtype=str)

	PSR = PPDOT[:,0]
	P = PPDOT[:,1]
	P_dot = PPDOT[:,2]

	locs_log = np.logical_or(P =='*', P_dot=='*')
	locs = np.where (locs_log == True)[0]

	PSR_sel = np.delete(PSR, locs)
	P = (np.delete(P, locs)).astype(float)
	P_dot = (np.delete(P_dot, locs)).astype(float)

	sample = np.random.choice(len(P), size=args.Nsamp, replace=False)
	filename = 'Pulsar.sample.'+TIMESTAMP+'.comp.'+str(args.n)+'.txt'
	data_sampled = zip(PSR_sel[sample], P[sample], P_dot[sample])
	np.savetxt(filename, list(data_sampled), fmt="%s %s %s", delimiter='\t')

	plt.plot(P, P_dot, '*', alpha =0.4, color='grey', label='Full population')
	plt.plot(P[sample], P_dot[sample], 'P', color='blue', alpha=0.5, label = 'Sampled population')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel('Period')
	plt.ylabel('Period derivative')
	plt.legend()
	plt.savefig('ppdot.'+str(args.n)+'.pdf')
	

############################################################






################ Dealing with the nu, nu_dot and time span of simulation and component independent uantities ###################
if args.f == 'N':
	nu_init = args.nu0
	nu_dot_init = -args.nudot0
else:
	nu_init = 1.0/P[sample]
	nu_dot_init = -(1.0/(P[sample])**2.0)*P_dot[sample]
	
N_year = args.Ny
component = args.n
ensemble_size = args.e

############################## Computation part starts ##################################
################### Two component computation ##############################

def dt_rescale(tau, dt):
	## Testing the tau and dt ratio
	## And resetting the dt value.
	## Use the parameters such that you need to reset dt.
	## For very small value of dt and large N_year, it might take a very long time to compute.
	## The differencial equations get solved by the jit compilor hence that part is fast. 
	ratio_dt_by_tau = (dt*86400.0)/tau
	if ratio_dt_by_tau > 1.0:
		#print (" dt > Tau, so exiting!")
		print (ratio_dt_by_tau)
		new_dt = (dt*86400.0)/((ratio_dt_by_tau//1)+1)
		print (" Setting new dt =", new_dt/86400.0, 'days ')
		dtnew = new_dt/86400.0
	else:
		dtnew = dt
		
	return dtnew
	
	
	
def dt_rescale3(tau1, tau2, dt):
	## Testing the tau and dt ratio
	## And resetting the dt value.
	## Use the parameters such that you need to reset dt.
	## For very small value of dt and large N_year, it might take a very long time to compute.
	## The differencial equations get solved by the jit compilor hence that part is fast. 
	ratio_dt_by_tau1 = (dt*86400.0)/tau1
	ratio_dt_by_tau2 = (dt*86400.0)/tau2
	if ratio_dt_by_tau1 > 1.0 or ratio_dt_by_tau1 > 1.0:
		print (' Either Tau1 or tau2 is larger than dt. Re-defining time steps')
		min_tau = np.min([tau1, tau2])
		ratio_min_tau = (dt*86400.0)/min_tau
		new_dt = (dt*86400.0)/((ratio_min_tau//1)+1)
		print (" Setting new dt =", new_dt/86400.0, 'days ')
		dtnew = new_dt/86400.0
	else:
		dtnew = dt
	return dtnew




if component == 2:
	alpha_infinity = args.ainf
	alpha_mutual_friction = args.amf1
	B = args.B1
	x1 = args.x1
	xp = 1.0-x1
	t_init = 0.0
	dt = args.dt## use this value in days.
	print (' Input sampling time (in days): ', dt)
			
			
	if args.f !='N':
	
		psr_name =[]
		freq_s =[]
		freqd_s = []
		var = []
		Timespan = []
		tau_arr =[]
		for i in range(len(nu_init)):
			b = (2.0*(2.0*np.pi*nu_init[i])*B)
			tau = xp/b
			omega_dot_init_ps = (2.0*np.pi*nu_dot_init[i])
			omega_init_ps = (2.0*np.pi*nu_init[i])
			sigma_inf2 = np.abs(omega_dot_init_ps)*omega_init_ps*alpha_infinity**2.0
			sigma_T2 = ((alpha_mutual_friction*x1*omega_dot_init_ps)**2.0)/b
			lag_two_comp = np.abs(2.0*np.pi*nu_dot_init[i]*tau)*xp
			dt_rs = dt_rescale(tau, dt)
			if dt_rs > 0.005:
				var_array =[]
				for _ in range(ensemble_size):
					time_v, delta_rot_p, delta_rot_1, omega_p_secular = two_comp_int (t_init, N_year, x1, dt_rs, nu_dot_init[i], nu_init[i], B, alpha_infinity, alpha_mutual_friction, lag=lag_two_comp)
					
					delta_phi = integrate.cumtrapz(delta_rot_p/(2.0*np.pi), time_v, initial =0.0) # here phase is defined in 0 to 1.
					timing_residual = delta_phi/ (omega_p_secular/(2.0*np.pi))
					
					fit_straight_line = np.polyfit(time_v, timing_residual, 1)
					
					predict = np.poly1d(fit_straight_line)
					
					y_predict = predict (time_v)
					
					timing_residual = timing_residual - y_predict
					
					plt.plot(time_v/(86400*365), timing_residual)
					plt.xlabel('Time (yr)')
					plt.ylabel('Timing residual (sec)')
					
					variance = (float(len(timing_residual))**-1.0)*np.sum(timing_residual**2.0)
					var_array.append(variance)
				
				variance = np.median(var_array)
				freq_s.append(nu_init[i])
				freqd_s.append(nu_dot_init[i])
				var.append(np.sqrt(variance))
				Timespan.append(N_year)
				psr_name.append(PSR_sel[i])
				tau_arr.append(tau)
				plt.show()
						
			else:
				print('Skipped this pulsar as the dt required is ', dt_rs, 'simulations are performed for the pulsars with dt > 0.005')
				
				
		plt.subplot(1,2,2)	
		plt.loglog(np.abs(freqd_s), var, 'o', color='k', alpha=0.5)
		plt.xlabel(r'$\dot \nu$ (Hz/s)')
		plt.subplot(1,2,1)	
		plt.loglog(freq_s, var, 'o', color='k', alpha=0.5)
		plt.xlabel(r'$\nu$ (Hz)')
		plt.ylabel(r'$\sigma_{TN}$ (s)')
		plt.savefig('Population.plot.pdf')
		
		data = zip(freq_s, freqd_s, var, tau_arr, Timespan)
		np.savetxt('Pulsar.pop.'+TIMESTAMP+'.comp.'+str(args.n)+'.solution.dat', list(data), header=' col1 = F0 (Hz) col2 = F1 (Hz/s) col3 = sigma_p(s) col4 = tau (sec) col5= Timespan(Yr)', comments='# x1='+str(x1)+' B='+str(B)+' alpha_inf='+str(alpha_infinity)+' alpha_mf ='+str(alpha_mutual_friction))
	
	
	
	
if component == 3:
	if args.amf2 == None:
		print ('Need to pass the -amf2 argument')
		sys.exit()
	if args.B2 == None:
		print ('Need to pass the -B2 argument')
		sys.exit()
	if args.x2 == None:
		print ('Need to pass the -x2 argument')
		sys.exit()


	t_init = 0.0
	alpha_infinity = args.ainf
	alpha_mutual_friction1 = args.amf1
	alpha_mutual_friction2 = args.amf2
	B1 = args.B1
	B2 = args.B2
	x1 = args.x1
	x2 = args.x2
	dt = args.dt## use this value in days.

	if args.f !='N':
		psr_name =[]
		freq_s =[]
		freqd_s = []
		var = []
		Timespan = []
		for i in range(len(nu_init)):
			b1 = 2.0*np.pi*nu_init[i]*B1
			b2 = 2.0*np.pi*nu_init[i]*B2
			xp = 1.0-x1-x2
			LAG1 = (np.abs(2.0*np.pi*nu_dot_init[i]))/(2.0*np.pi*nu_init[i]*B1)
			LAG2 = (np.abs(2.0*np.pi*nu_dot_init[i]))/(2.0*np.pi*nu_init[i]*B2)
			tau1 = 1.0/(2.0*(2.0*np.pi*nu_init[i])*B1)
			tau2 = 1.0/(2.0*(2.0*np.pi*nu_init[i])*B2)
			print ('Solving for pulsar number:',i+1,' tau1 =', tau1, 'tau2 =', tau2)
			dt_rs = dt_rescale3(tau1,tau2, dt)
			if dt_rs > 0.005:
				var_array =[]
				for _ in range(ensemble_size):
					time_v, delta_rot_p, delta_rot_1, delta_rot_2, omega_p_secular = three_comp_int(t_init, N_year, x1, x2, dt, nu_dot_init[i], nu_init[i], B1, B2, alpha_infinity, alpha_mutual_friction1,  alpha_mutual_friction2, lag1=LAG1, lag2=LAG2)
					
					delta_phi = integrate.cumtrapz(delta_rot_p/(2.0*np.pi), time_v, initial =0.0) # here phase is defined in 0 to 1.
					timing_residual = delta_phi/ (omega_p_secular/(2.0*np.pi))
					
					variance = (float(len(timing_residual))**-1.0)*np.sum(timing_residual**2.0)
					var_array.append(variance)
				variance = np.median(var_array)
				freq_s.append(nu_init[i])
				freqd_s.append(nu_dot_init[i])
				var.append(np.sqrt(variance))
				Timespan.append(N_year)
				psr_name.append(PSR_sel[i])
				
					
			else:
				print('Skipped this pulsar as the dt required is ', dt_rs, 'simulations are performed for the pulsars with dt > 0.005')
				
				
		plt.subplot(1,2,2)	
		plt.loglog(np.abs(freqd_s), var, 'o', color='k', alpha=0.5)
		plt.xlabel(r'$\dot \nu$ (Hz/s)')
		plt.subplot(1,2,1)	
		plt.loglog(freq_s, var, 'o', color='k', alpha=0.5)
		plt.xlabel(r'$\nu$ (Hz)')
		plt.ylabel(r'$\sigma_{TN}$ (s)')
		plt.savefig('Population.plot.pdf')
		
		data = zip(freq_s, freqd_s, var, Timespan)
		np.savetxt('Pulsar.pop.'+TIMESTAMP+'.comp.'+str(args.n)+'.solution.dat', list(data), header=' col1 = F0 (Hz) col2 = F1 (Hz/s) col3 = sigma_p(s) col4 = Timespan (Yr)', comments='# x1='+str(x1)+' x2='+str(x2)+' B1='+str(B1)+' B2='+str(B2)+' alpha_inf='+str(alpha_infinity)+' alpha_mf1 ='+str(alpha_mutual_friction1)+' alpha_mf2 ='+str(alpha_mutual_friction2))







