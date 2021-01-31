import numpy as np
from matplotlib import pyplot as plt

J1 = 0.015       # maximum excitatory strength
J2 = 0.02        # other strengthes
l = 0.15         # decay constant of excitatory interactions
Sigma = 1.8      # weights factor

V_res = 0.85     # resting state of membrane potential
V_th = 1.        # spike threshold
tau_mem = 20.    # time constant of membrane potential
tau_ex = 6.      # time constant of excitatory synaptic current
tau_in = 4.      # time constant of inhibitory synaptic current
S_ex = 0.2       # reliability of E synaptic transmission
S_in = 0.7       # reliability of I synaptic transmission
lambda_e = 0.03    # degree of place-specific E input modulation
lambda_i = 0.02    # degree of place-specific I input modulation

I0_ex = 1.02         # input current baseline
I0_in = 1.02        # input current baseline
T = 1000/9       # period of inhibitory input

num_E = 800      # num of E neurons 
num_I = 200      # num of I neurons 
num_ALL = num_E+num_I # num of all neurons 

time_MOVING = 4000 # time of rat moving
dt = 2             # time step for simulation

rand_choices = [0, 0.25,0.5, 0.75,1]

def get_ee_w(sigma, xi, xj):
    return sigma*J1*np.exp(-np.abs(xi-xj)/l)

def fr_sigmoid(v):
    return 1 / (1 + np.exp(-x))

def get_avg_mem_dv(v, i):
    return (-v+i)/tau_mem

def get_syn_di(i, syn_tau, s, Jij, spike_j):
    i_decay = -i/syn_tau 
    i_rec =  s*np.dot(Jij, spike_j)
    return i_decay + i_rec

def get_ext_e_i(xi, x0_t):
    return I0_ex*(1+lambda_e*np.exp(-np.abs(xi-x0_t)/l))

def get_ext_i_i(t, period):
    return I0_in*(1+lambda_i*np.cos(2*np.pi*t/period))

def get_x0(t, tot_time):
    return t/tot_time

def get_position_labels(max_p=1):
    return np.linspace(0,max_p,num_E)

def init_rec_w(x_labels):
    W_rec = np.ones((num_ALL, num_ALL))*J2

    for i in range(num_E):
        for j in range(num_E):
            sigma_ij = 1 if i<j else Sigma
            W_rec[i][j] = get_ee_w(sigma=sigma_ij, xi=x_labels[i],xj=x_labels[j])

    np.fill_diagonal(W_rec, 0.)

    return W_rec

class ThetaRhythmSpikeNetwork():
    def __init__(self,tot_time, ext_t=T):
        self.tot_time = tot_time
        self.x_labels = get_position_labels(1)
        self.W_rec = init_rec_w(self.x_labels)
        self.time_map = np.arange(0, tot_time, dt)
        self.step_num = len(self.time_map)
        self.Vs = np.zeros((num_ALL, self.step_num))
        self.Vs[:,0] = V_res
        self.Is = np.zeros((num_ALL, self.step_num))
        self.I_exs = np.zeros((num_ALL, self.step_num))
        self.I_ins = np.zeros((num_ALL, self.step_num))
        self.Ss = np.zeros((num_ALL, self.step_num))

        self.ext_t = ext_t

    def simulation(self,):
        for curr_ts in range(1, self.step_num):
            curr_x0 = get_x0(self.time_map[curr_ts], self.tot_time)

            for n in range(0, num_ALL):
                prev_iex = self.I_exs[n, curr_ts-1]
                prev_iin = self.I_ins[n, curr_ts-1]
                prev_In = self.Is[n, curr_ts-1]
                prev_v = self.Vs[n, curr_ts-1]
                prev_s = self.Ss[:, curr_ts-1]

                curr_s_ex = np.random.choice(rand_choices)*S_ex
                curr_s_in = np.random.choice(rand_choices)*S_in

                if n < num_E:
                    curr_i_ext = get_ext_e_i(xi=self.x_labels[n], x0_t=curr_x0)
                else:
                    curr_i_ext = get_ext_i_i(self.time_map[curr_ts],period=self.ext_t)

                curr_i_ex = dt*get_syn_di(prev_iex, tau_ex, curr_s_ex, 
                                       self.W_rec[n,:num_E], prev_s[:num_E])+prev_iex
                curr_i_in = dt*get_syn_di(prev_iin, tau_in, curr_s_in, 
                                       self.W_rec[n,num_E:], prev_s[num_E:])+prev_iin

                curr_i = curr_i_ex-curr_i_in+curr_i_ext

                self.Is[n,curr_ts]=curr_i
                self.I_exs[n, curr_ts] = curr_i_ex
                self.I_ins[n, curr_ts] = curr_i_in

                curr_v = dt*get_avg_mem_dv(v=prev_v, i=curr_i)+prev_v
                
                if curr_v >= V_th:
                    self.Vs[n, curr_ts] = V_res
                    self.Ss[n, curr_ts] = 1.
                else:
                    self.Vs[n, curr_ts] = curr_v
                    self.Ss[n, curr_ts] = 0.

class ThetaRhythmRefSpikeNetwork(ThetaRhythmSpikeNetwork):
    def __init__(self, tot_time=time_MOVING, ext_t=T, ref_t=6):
        super().__init__(tot_time=tot_time, ext_t=ext_t)
        self.refs = np.zeros(num_ALL)
        self.ref_ts = int(ref_t/dt)
        
    def simulation(self,):
        for curr_ts in range(1, self.step_num):
            curr_x0 = get_x0(self.time_map[curr_ts], self.tot_time)

            for n in range(0, num_ALL):
                prev_iex = self.I_exs[n, curr_ts-1]
                prev_iin = self.I_ins[n, curr_ts-1]
                prev_In = self.Is[n, curr_ts-1]
                prev_v = self.Vs[n, curr_ts-1]
                prev_s = self.Ss[:, curr_ts-1]

                curr_s_ex = np.random.choice(rand_choices)*S_ex
                curr_s_in = np.random.choice(rand_choices)*S_in

                if n < num_E:
                    curr_i_ext = get_ext_e_i(xi=self.x_labels[n], x0_t=curr_x0)
                else:
                    curr_i_ext = get_ext_i_i(self.time_map[curr_ts],period=self.ext_t)

                curr_i_ex = dt*get_syn_di(prev_iex, tau_ex, curr_s_ex, 
                                       self.W_rec[n,:num_E], prev_s[:num_E])+prev_iex
                curr_i_in = dt*get_syn_di(prev_iin, tau_in, curr_s_in, 
                                       self.W_rec[n,num_E:], prev_s[num_E:])+prev_iin

                curr_i = curr_i_ex-curr_i_in+curr_i_ext

                self.Is[n,curr_ts]=curr_i
                self.I_exs[n, curr_ts] = curr_i_ex
                self.I_ins[n, curr_ts] = curr_i_in

                curr_v = dt*get_avg_mem_dv(v=prev_v, i=curr_i)+prev_v
                
                if curr_v >= V_th:
                    if self.refs[n] == 0.:
                        self.Ss[n, curr_ts] = 1.
                        self.refs[n] = self.ref_ts
                    else:
                        self.refs[n] = self.refs[n]-1
                        
                    self.Vs[n, curr_ts] = V_res
                else:
                    self.Vs[n, curr_ts] = curr_v
                    self.Ss[n, curr_ts] = 0.



# tot_times = list(range(3200, 4200, 200))
tot_times = list(range(3000, 4100, 200))
# tot_times = list(range(time_MOVING+200, time_MOVING+2200, 200))
itr_num = 200

for tot_t in tot_times:
    print('tot_t',tot_t)
    curr_spike_recording = []
    for t in range(itr_num):
        print('record itr',t)
        snn = ThetaRhythmRefSpikeNetwork(tot_t)#ThetaRhythmSpikeNetwork(tot_t)
        snn.simulation()
        # spike_recording.append(snn.Ss)
        with open('/home/opc/disk/0130/spike_recording_totT'+str(tot_t)+'_itr'+str(t)+'.npy', 'wb') as f:
             np.save(f, snn.Ss)