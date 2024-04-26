from sympy.parsing.sympy_parser import parse_expr
import sympy
from sympy.solvers import solve
from scipy.optimize import fsolve
import numpy as np
sympy.init_printing(pretty_print=False)

class WheatstoneBridge:
    '''      
    A class for definition of various strain gauge setups, for a variety of Wheatstone bridge variations.

    Attributes
    --------------
    GF, k :
    poisson, v :
    V :
    R1, R2, R3, R4 : functions, optional
        functions of variables `eps` and `v`
    gauge_resistance : float, default=350
        nominal resistance of strain gauge
    
    Notes
    ----------
    The Wheatstone brige circuit is defined as follows:

              x ---------
            .  .        |
          R1     R4     |
        .          .    |
       x --- Vo --- x   V
        .          .    |   
         R2      R3     |
           .    .       |
             x ----------

    Here, all resistances R1-R4 are either given by strain gauges or fixed resistors. V is the input
    voltage (excitation) and Vo the output voltage.

    '''

    resistor_names = ['R1', 'R2', 'R3', 'R4']

    def __init__(self, V, GF=2.0, poisson=0.3, gauge_resistance=350.0,
                 R1=None, R2=None, R3=None, R4=None):
        '''
        
        Parameters
        ------------
        V : 
        GF : 
        poisson :
        R1, R2, R3, R4 : functions, optional
            strings defining functions of variables `eps` and `v`
        gauge_resistance : float, default=350
            nominal resistance of strain gauge
        balanced : boolean, default=True
            whether or not the bridge is designed to be balanced at no strain (used to predict values for resistors
            not being strain gauges); otherwise, non-gauge resistors must be explicitly defined with float values

        Example
        -----------
        A quarter bridge setup (using R1) with an input voltage of 5V, GF=2.0 and poisson ratio of 0.3,
        would be defined as follows:
            
            `StrainGauge(5.0, R1='eps')`

        Alternatively, a half bridge setup used for a bending type of strain where strain gauges 
        (let's say R1 and R2) are placed on each side of a bending beam can be defined as follows:

            `StrainGauge(5.0, R1='eps', R2='-eps')`
        
        As the strains from both tension and compression is added (inverse signs), the bridge factor is 2 in this case.
        (Note that this is taken care of automatically).

        Finally, a full bridge setup for tension and compression can designed by placing two gauges on the tension side and
        two gauges on the compression side of a beam. One of the gauges on each side will be placed in the transverse direction 
        (the other in the principal direction) to compensate for temperature effects and bending. Assume R1 and R2 (transverse) is placed
        on the tension side and R3 (transverse) and R4 on the compression side. This setup is described as follows:

            `StrainGauge(5.0, R1='eps', R2= '-v*eps', R3='-v*eps', R4='eps')`

        '''
        self.GF = self.k = GF
        self.V = V  #input voltage/excitation
        self.poisson = self.v = poisson     # poisson ratio of material mounted to 
        self.R0 = gauge_resistance

        # Define symbolic expressions
        eps, v, Vo = sympy.symbols('eps v Vo')   #initiate variables from sympy
        self.sym = dict(eps=eps, v=v, Vo=Vo)

        self.assign_R('R1', R1)
        self.assign_R('R2', R2)
        self.assign_R('R3', R3)
        self.assign_R('R4', R4)

        # Assign Rs not defined
        self.find_unset_R()
        
        # Prepare for strain prediction
        self.calc()

    def calc(self):
        eps = self.sym['eps']
        Vo = self.sym['Vo']
        self.Vout = (self.R3/(self.R3 + self.R4) - self.R2/(self.R1 + self.R2))*self.V
        self.get_Vout = sympy.lambdify(eps, self.Vout)
        self.strain = solve(self.Vout - Vo, eps)

    def find_unset_R(self):
        if len(self.defined_resistances)<4:
            finished = False
            while not finished:
                r1, r2, r3, r4 = sympy.symbols('r1 r2 r3 r4')

                if 'R1' in self.defined_resistances:
                    r1 = r1.subs(r1, self.get_nominal_resistance('R1'))
                if 'R2' in self.defined_resistances:
                    r2 = r2.subs(r2, self.get_nominal_resistance('R2'))
                if 'R3' in self.defined_resistances:
                    r3 = r3.subs(r3, self.get_nominal_resistance('R3'))
                if 'R4' in self.defined_resistances:
                    r4 = r4.subs(r4, self.get_nominal_resistance('R4'))
                
                self.sym.update({'R1':r1, 'R2':r2, 'R3':r3, 'R4':r4})

                f = (r4/r3 - r2/r1)**2
                sols = solve(f, r1, r2, r3, r4)
                if len(sols)>=1:
                    sols = sols[0]
                else:
                    raise ValueError('No valid solution for specified resitances/gauge combo found.')
                
                all_var = [r1, r2, r3, r4]
                undetermined = [str(r).upper() for r in all_var if isinstance(r, sympy.Symbol) ]
                
                if type(sols) is dict:
                    add_dict = {}
                    for a in sols:
                        add_dict = add_dict | {str(a).upper(): sols[a]}
                else:
                    add_dict = {}
                
                # print(add_dict)
                self.sol_dict = dict(zip(self.resistor_names, all_var))
                self.sol_dict = self.sol_dict | add_dict

                masters = [name for name in self.resistor_names if (self.sol_dict[name]==self.sym[name]) and name in undetermined]     #needs to be set

                if len(masters)>0:
                    self.assign_R(masters[0], self.R0)
                else:
                    self.assign_R(undetermined[0], self.sol_dict[undetermined[0]])
                    finished = True

    @property
    def is_balanced(self):
        R1 = self.eval_R('R1', val=0.0)
        R2 = self.eval_R('R2', val=0.0)
        R3 = self.eval_R('R3', val=0.0)
        R4 = self.eval_R('R4', val=0.0)
        return R2/R1==R4/R3
    
    def eval_R(self, name, val=0.0):
        eps = self.sym['eps']

        R = getattr(self, name)
        if isinstance(R, sympy.Expr):
            return float(R.subs(eps, val))
        else:
            return R
        
        
    def get_strain_fun(self):
        if len(self.strain)==1:
            Vo = self.sym['Vo']
            return sympy.lambdify(Vo, self.strain[0])
        else:
            return lambda __: np.nan

    def get_strain(self, Vout):
        eps = self.sym['eps']
        fun = sympy.lambdify(eps, self.Vout - Vout)
        sol = fsolve(fun, 0.0)
        if len(sol)==1:
            return sol[0]
        else:
            return np.nan

    def assign_R(self, name, R):
        v = self.sym['v']

        if R is not None and isinstance(R, str):
            eps_exp = sympy.symbols('eps_exp') 
            Rfun = (self.R0*(1 + eps_exp*self.GF))
            assign_val = Rfun.subs(eps_exp, parse_expr(R).subs(v, self.poisson))
        else:
            assign_val = R

        setattr(self, name, assign_val)

    @property
    def R1(self):
        if hasattr(self, 'calc_R1'):
            return self.calc_R1
        else:
            return self._R1
    
    @R1.setter
    def R1(self, val):
        self._R1 = val

    @property
    def R2(self):
        if hasattr(self, 'calc_R2'):
            return self.calc_R2
        else:
            return self._R2
    
    @R2.setter
    def R2(self, val):
        self._R2 = val

    @property
    def R3(self):
        if hasattr(self, 'calc_R3'):
            return self.calc_R3
        else:
            return self._R3
    
    @R3.setter
    def R3(self, val):
        self._R3 = val

    @property
    def R4(self):
        if hasattr(self, 'calc_R4'):
            return self.calc_R4
        else:
            return self._R4
    
    @R4.setter
    def R4(self, val):
        self._R4 = val

    @property
    def strain_gauges(self):
        return [name for name in self.resistor_names if isinstance(getattr(self, name), sympy.Expr)]
         
    @property
    def resistors(self):
        return [name for name in self.resistor_names if not isinstance(getattr(self, name), sympy.Expr)]
    
    @property
    def defined_resistors(self):
        return [name for name in self.resistor_names if type(getattr(self, '_' + name)) in [int, float]]   
    
    @property
    def undefined_resistors(self):
        return [name for name in self.resistor_names if not type(getattr(self, '_' + name)) in [int, float]]
    
    @property
    def defined_resistances(self):
        return list(set(self.defined_resistors + self.strain_gauges)) 
    
    def get_nominal_resistance(self, name):
        if name not in self.defined_resistances:
            return np.nan
        else:
            return self.eval_R(name, val=0.0)



