import graphix.command
from graphix.random_objects import Circuit
from graphix.states import BasicStates
import numpy as np
from client_vqa import Client, Secrets
import random as rd
from graphix.sim.statevec import StatevectorBackend
import pandas as pd 


class e2e():
    def __init__(self,shots,t,angle):
        self.d=4*shots 
        self.t=t
        self.N=self.d+t
        self.shots=shots
        self.angle=angle
        self.angle_noise_max = self.angle
        self.angle_noise_min = self.angle
        self.init_params = [.3, .9] 
        self.secrets = Secrets(r=True, a=True, theta=True)
        self.states = [BasicStates.PLUS for _ in range(2)]
        self.Z = np.array([[1, 0], [0, -1]])
        self.X = np.array([[0, 1], [1, 0]])
        self.hamiltonian= np.kron(self.Z, self.Z) + .2*np.kron(self.X, np.eye(2)) + .2*np.kron(np.eye(2), self.X)


    def tool(self,i,d_dict):
        if i<=d_dict[0]:
            return 0
        elif i>d_dict[0] and i<=d_dict[1]:
            return 1
        elif i>d_dict[1] and i<=d_dict[2]:
            return 2
        elif i>d_dict[2] and i<=d_dict[3]:
            return 3

    def prepare_computations(self,params,angle):
        circuit = Circuit(2)
        circuit.ry(0,params[0])
        circuit.ry(1,params[1])
        circuit.cnot(0, 1)
        pattern = circuit.transpile().pattern 
        pattern.standardize()

        classical_output = pattern.output_nodes
        for onode in classical_output:
            pattern.add(graphix.command.M(node=onode,angle=angle))
        pattern.standardize()

        client = Client(pattern=pattern, secrets=self.secrets, input_state=self.states)
        return pattern,client,classical_output

    def parameter_shift_angles(self,params):
        shift = .5
        shifted=[]
        for i in range(len(params)):
            # Copy of the original parameters
            shifted_plus = np.copy(params)
            shifted_minus = np.copy(params)

            shifted_plus[i] += shift
            # Backward shift
            shifted_minus[i]-= shift
            shifted.append([shifted_plus,shifted_minus])

        return shifted
    def exp_value(self,counts):
        total_shots = sum(counts.values())
        prob_vector = np.zeros(4)
        bitstring_to_index = {'00': 0, '01': 1, '10': 2, '11': 3}

        for bitstring, count in counts.items():
            index = bitstring_to_index[bitstring]
            prob_vector[index] = count / total_shots
        rho = np.diag(prob_vector)
        return np.trace(self.hamiltonian @ rho).real

    def grad_computation(self,all_histograms):
        gradients = np.zeros_like(self.init_params)
        for i in range(0,4,2):
            forward = self.exp_value(all_histograms[i])
            backward = self.exp_value(all_histograms[i+1])
            gradients[int(i/2)] = 0.5 * (np.array(forward) - np.array(backward))
        return gradients


    def run(self,p_values,show=True):
        params_psr=self.parameter_shift_angles(self.init_params)
        data_run={}
        cnt=0
        for i in params_psr:
            for el in i:
                data_run[cnt]=el
                cnt+=1
        #picking one of the clients, is the same
        _,client,_=self.prepare_computations(data_run[0],0)
        test_runs = client.create_test_runs()

        number_of_traps = sum([len(run.trap_qubits) for run in test_runs])
        n_nodes = len(client.graph[0])
        if show:
            print(f"There are {number_of_traps} traps in total. (VBQC uses single-qubit traps)")

        d_dict={i-1:i*self.shots for i in range(1,5)}

        rounds = list(range(self.N))
        rd.shuffle(rounds)
        all_histograms = {}
        failed_traps_histograms = {}

        df=pd.DataFrame()
        noise_flag=False
        for p in p_values:
            if p==0:
                if show:
                    print("Computing noise free case..")
            else:
                if show:
                    print(f"Attempt with noise model p={p}")
            outcomes_dicts={i:dict() for i in range(4)}

            n_failed_traps = 0
            # Iterating through rounds
            for i in rounds:
                backend = StatevectorBackend()
                if np.random.random()<p:
                    angle_noise=rd.uniform(self.angle_noise_min, self.angle_noise_max)
                    noise_flag=True 

                if i<=self.d:
                    j=self.tool(i,d_dict)
                    if noise_flag:
                        pattern,client,classical_output=self.prepare_computations(data_run[j],angle=angle_noise)
                    else:
                        pattern,client,classical_output=self.prepare_computations(data_run[j],angle=0)

                    client.refresh_randomness()
                    client.delegate_pattern(backend=backend)

                    computation_outcome = ""
                    for onode in classical_output:
                        computation_outcome += str(int(client.results[onode]))
                    if computation_outcome not in outcomes_dicts[j]:
                        outcomes_dicts[j][computation_outcome] = 1
                    else:
                        outcomes_dicts[j][computation_outcome] += 1   
                else:
                    # Test round
                    run = rd.choice(test_runs)
                    client.refresh_randomness()
                    if noise_flag:
                        trap_outcomes,_ = client.delegate_test_run(outnodes=[5,11],angles_noise=angle_noise,run=run, backend=backend)
                    else:
                        trap_outcomes,_ = client.delegate_test_run(outnodes=[5,11],angles_noise=0,run=run, backend=backend)

                    if sum(trap_outcomes) != 0:
                        n_failed_traps += 1
                noise_flag=False

            if p==0:
                noisless_grad=self.grad_computation(outcomes_dicts)
                #print(np.linalg.norm(noisless_grad,ord=1))
            else:
                out=self.grad_computation(outcomes_dicts)
                grad_abs_err=np.linalg.norm(out - noisless_grad,ord=1)
                grad_rel_err=np.linalg.norm(out - noisless_grad,ord=1)/np.linalg.norm(noisless_grad,ord=1)
                if show:
                    print(f"    Noise: {p} Gradient Abs error {grad_abs_err}")
                    print(f"                Gradient Rel error {grad_rel_err}")

            # Combine results
            all_histograms[p] = outcomes_dicts
            if self.t != 0:
                failed_traps_histograms[p] = n_failed_traps / (self.t)
            if p!=0:
                if show:
                    print(f"    Traps detected: {failed_traps_histograms[p]*self.t} Perentage: {failed_traps_histograms[p]}")
                new_row={"noise":p,
                        "grad_abs_err": grad_abs_err,
                        "grad_rel_err": grad_rel_err,
                        "traps":failed_traps_histograms[p]*self.t}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)  
        return df
