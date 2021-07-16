# import icecream as ic

from numpy.random import uniform

import pandas as pd



#Read all the data from different files
#data.load(filename='economic_dispatch/scalars.dat')
gen_data = pd.read_csv(r"gen_data.csv",header=0)
gen_data_rand = pd.read_csv(r"gen_data.csv", header=0, index_col=0)
dem_data = pd.read_csv(r"dem_data.csv",header=0)


# gendata = pd.read_excel (r'Generation_type_data_SMR_CCS.xlsx',header=0)
# gendata_fix = pd.read_excel (r'Generation_type_data_SMR_CCS.xlsx',header=0, index_col=0)
# fixed_o_m=dict(zip(list(gendata.iloc[:]['Type']),list(gendata.iloc[:]['fixed_o_m'])))#(fixedom.values)
# variable_o_m=dict(zip(list(gendata.iloc[:]['Type']),list(gendata.iloc[:]['variable_o_m'])))#dict(variableom.values)
# capitalcost=dict(zip(list(gendata.iloc[:]['Type']),list(gendata.iloc[:]['capitalcost'])))#dict(capital_cost.values)
# fuelprice=dict(zip(list(gendata.iloc[:]['Type']),list(gendata.iloc[:]['fuelprice'])))#dict(fuel_price.values)

gen_type = gen_data['g'].tolist()
dem = dict(zip(list(dem_data.iloc[:]['t']),list(dem_data.iloc[:]['dem'])))
ccost = dict(zip(list(gen_data.iloc[:]['g']),list(gen_data.iloc[:]['ccost'])))
fomcost = dict(zip(list(gen_data.iloc[:]['g']),list(gen_data.iloc[:]['fomcost'])))
vomcost = dict(zip(list(gen_data.iloc[:]['g']),list(gen_data.iloc[:]['vomcost'])))
cap = dict(zip(list(gen_data.iloc[:]['g']),list(gen_data.iloc[:]['cap'])))

# configuration = pd.read_excel (r'COPPER_configuration.xlsx',header=0)
# config=dict(zip(list(configuration.iloc[:]['Parameter']),list(configuration.iloc[:]['Value'])))
# ctax=int(config['carbon price'])

max_dem = 1000
min_dem = 100
max_cos = 100
min_cos = 10
max_cap = 700
min_cap = 400


# for type in gendata.iloc[:]['Type']:
#     locals()["r_fixed_"+type] = uniform(0, 1, size=5)
#     locals()["r_var_"+type] = uniform(0, 1, size=5)
#     locals()["r_fp_" + type] = uniform(0, 1, size=5)
#     locals()["r_cap_" + type] = uniform(0, 1, size=5)

# r_ctax = uniform(0, 1, size=5)

for g in gen_data.iloc[:]['g']:
    locals()["ccost_"+g] = uniform(0, 1, size=1000)
    locals()["fomcost_"+g] = uniform(0, 1, size=1000)
    locals()["vomcost_"+g] = uniform(0, 1, size=1000)
    locals()["cap_"+g] = uniform(0, 1, size=1000)



#for t in dem_data.iloc[:]['t']:
#    locals()["dem_"+str(t)] = uniform(0, 1, size=1000)*1000


for i in range(0,1000):
#   for t in dem_data.iloc[:]['t']:
#       dem_data.iloc[t-1,1] = locals()["dem_"+str(t)][i]
    for g in gen_type:
        gen_data_rand.loc[g, 'ccost'] = ccost[g] * 0.8 + ccost[g] * 0.4 * locals()["ccost_" + g][i]
        gen_data_rand.loc[g, 'fomcost'] = fomcost[g] * 0.8 + fomcost[g] * 0.4 * locals()["fomcost_" + g][i]
        gen_data_rand.loc[g, 'vomcost'] = vomcost[g] * 0.8 + vomcost[g] * 0.4 * locals()["vomcost_" + g][i]
        gen_data_rand.loc[g, 'cap'] = cap[g] * 0.8 + cap[g] * 0.4 * locals()["cap_" + g][i]

#   dem_data.to_csv('dem_data_'+str(i)+'.csv', index=False)
    gen_data_rand.to_csv('gen_data_' + str(i) + '.csv', index=gen_type)

# for i in range(1,5):
#     for type in gendata.iloc[:]['Type']:
#         for thermal in gendata.iloc[:]['Is thermal?']:
#             if thermal==True:
#                 gendata_fix.loc[type, 'fixed_o_m'] = fixed_o_m[type] * min_th + locals()["r_fixed_" + type][i] * (
#                             fixed_o_m[type] * max_th - fixed_o_m[type] * min_th)
#                 gendata_fix.loc[type, 'variable_o_m'] = variable_o_m[type] * min_th + locals()["r_var_" + type][i] * (
#                         variable_o_m[type] * max_th - variable_o_m[type] * min_th)
#                 gendata_fix.loc[type, 'fuelprice'] = fuelprice[type] * min_fp + locals()["r_fp_" + type][i] * (
#                         fuelprice[type] * max_fp - fuelprice[type] * min_fp)
#                 gendata_fix.loc[type, 'capitolcost'] = capitalcost[type] * min_th + locals()["r_cap_" + type][i] * (
#                         capitalcost[type] * max_th - capitalcost[type] * min_th)
#             else:
#                 gendata_fix.loc[type, 'fixed_o_m'] = fixed_o_m[type] * min_vre + locals()["r_fixed_" + type][i] * (
#                             fixed_o_m[type] * max_vre - fixed_o_m[type] * min_vre)
#                 gendata_fix.loc[type, 'variable_o_m'] = variable_o_m[type] * min_vre + locals()["r_var_" + type][i] * (
#                         variable_o_m[type] * max_vre - variable_o_m[type] * min_vre)
#                 gendata_fix.loc[type, 'capitolcost'] = capitalcost[type] * min_vre + locals()["r_cap_" + type][i] * (
#                         capitalcost[type] * max_vre - capitalcost[type] * min_vre)

    # configuration.loc[4, 'Value'] = 200 * r_ctax[i]
    # configuration.to_excel('COPPER_Configuration_' + str(i) + '.xlsx', index=False)

    # gendata_fix.to_excel('Generation_type_data_SMR_CCS_'+ str(i)+'.xlsx', index=True)


