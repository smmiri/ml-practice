#Import the relevant libraries 
from __future__ import division
import os
from pyomo.environ import *
from pyomo.opt import SolverFactory

for i in range(964,1000):
    #Define an abstract model
    model = AbstractModel()

    #Define the sets and parameters of the abstract model
    model.t = Set() #time periods
    model.g = Set() #generation tecnologies

    model.ccost = Param(model.g) # capital cost of generating units
    model.vomcost = Param(model.g) # variable operation & maintenance cost of generating units
    model.fomcost = Param(model.g) # fixed operation & maintenance cost of generating units
    model.cap = Param(model.g) #Installed capacity of individual plants (by generation type)
    model.dem = Param(model.t) #electricity demand

    #Define the variables of the abstract model
    model.gen = Var(model.g, model.t, domain=NonNegativeReals) #generation level of plants
    model.n = Var(model.g, domain=IntegerSet)

    #Define the objective function of the abstract model
    def obj_expression(model):
      return   sum(sum(model.ccost[g]*model.cap[g]*model.n[g] for g in model.g) +
                   sum(model.fomcost[g]*model.cap[g]*model.n[g] for g in model.g) +
                   sum(model.vomcost[g]*model.gen[g,t]  for g in model.g)
                   for t in model.t)

    model.OBJ = Objective(rule=obj_expression)

    #Power balance constraint
    def balance_rule(model, t):
        return sum(model.gen[g,t] for g in model.g) == model.dem[t]

    model.balance = Constraint(model.t, rule=balance_rule)

    # Notice the two arguments passed to the Constraint() function:
    #    i) <model.t> : this is the set over which the constraint is indexed over.
    #   ii) <rule=balance_rule>: this is the function we defined the constraint in.

    #Maximum generation constraint
    def max_gen_rule(model, g, t):
        return model.gen[g, t] <= model.cap[g]*model.n[g]
    model.max_gen = Constraint(model.g, model.t, rule=max_gen_rule)

    # Load in the data

    #Open a DataPortal
    data = DataPortal()
    input_data = os.getcwd()
    os.chdir(input_data)

    data.load(filename = input_data+'/capacity_expansion/dem_data.csv', format='set', set='t')
    data.load(filename = input_data+'/capacity_expansion/dem_data.csv', index ='t', param=['dem'])
    data.load(filename = input_data+'/capacity_expansion/gen_data_'+str(i)+'.csv', format='set', set='g')
    data.load(filename = input_data+'/capacity_expansion/gen_data_'+str(i)+'.csv', index='g', param=['ccost', 'fomcost','vomcost', 'cap'])

    #Create an instance of your model
    instance = model.create_instance(data)

    #We define the optimization solver. You can also use cplex, gurobi, etc
    opt = SolverFactory('cplex')

    ''' Extra
    import pandas as pd
    from IPython.display import display, HTML
    
    df_gen = pd.read_csv('capacity_expansion/gen_data.csv')
    df_dem = pd.read_csv('capacity_expansion/dem_data.csv')
    
    display(HTML(df_gen.to_html()))
    display(HTML(df_dem.head().to_html()))
    
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))
    
    df_dem.plot(x='t', y = 'dem', ax=ax1)
    ax1.set_ylim(ymin=0)
    ax1.set_ylabel('Demand [MW]')
    ax1.set_xlabel('Hour')
    ax1.set_title('Annual Demand')
    
    df_dem[df_dem.t <= 24].plot(x='t', y = 'dem', ax=ax2)
    ax2.set_ylim(ymin=0)
    ax2.set_ylabel('Demand [MW]')
    ax2.set_xlabel('Hour')
    ax2.set_title('Daily Demand')
    
    '''


    #Solve the optimization problem
    results = opt.solve(instance, symbolic_solver_labels = True, tee=True)

    print('The model built:')
    for g in instance.g.value:
        print('\t- %d %s plants' % (instance.n[g].value, g))

    print('\nThe total system cost is %.2d dollars.' % value(instance.OBJ))

    ''' Extra
    import matplotlib.pyplot as plt
    
    # Create a dictionary where the keys are the generator types and the values
    # are the corresponding number of generation plants built.
    
    n_generators_dict = {g:instance.n[g].value for g in instance.g.value if 
                     instance.n[g].value > 0}
    capacity_dict = {g:instance.n[g].value * instance.cap[g]
                     for g in instance.g.value if instance.n[g].value > 0}
    
    energy_dict = {}
    for g in instance.g.value:
        temp_list = []
        for t in instance.t.value:
            temp_list.append(instance.gen[g,t].value)
        if sum(temp_list) > 0:
            energy_dict[g] = temp_list
            
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8,8))
    axes[0,0].bar(n_generators_dict.keys(), n_generators_dict.values())
    axes[0,0].set_ylabel('Number of Generators')
    axes[0,0].set_title('Number of Generators by Type', fontsize=14)
    
    axes[0,1].bar(capacity_dict.keys(), capacity_dict.values())
    axes[0,1].set_ylabel('Generation Capacity [MW]')
    axes[0,1].set_title('Generation Capacities by Type', fontsize=14)
    
    for key in energy_dict:
        axes[1,0].plot(energy_dict[key], label = key)
    axes[1,0].set_xlabel('Hour')
    axes[1,0].set_ylabel('Generated Power [MW]')
    axes[1,0].set_title('Generation Plot', fontsize=14)
    axes[1,0].legend()
    
    capacity_factors = [ sum(x) / (y * 8760) * 100 for x, y in 
                        zip(energy_dict.values(), capacity_dict.values())]
    axes[1,1].bar(capacity_dict.keys(), capacity_factors)
    axes[1,1].set_ylabel('Capacity Factor [%]')
    axes[1,1].set_title('Capacity Factors', fontsize=14)
    
    plt.subplots_adjust(wspace = 0.3, hspace=0.3)
    
    '''


    #Write the results into a csv file

    #Write out generation data:
    f = open('result_gen_'+str(i)+'.csv', 'w')
    f.write('g,t,gen'+'\n')
    for g in instance.g.value:
        for t in instance.t.value:
            f.write(str(g)+','+str(t)+','+str(instance.gen[g,t].value)+'\n')


    #Write out capacity data:
    f = open('result_cap_'+str(i)+'.csv', 'w')
    f.write('g,cap'+'\n')
    for g in instance.g.value:
        f.write(str(g)+','+str(instance.n[g].value)+'\n')