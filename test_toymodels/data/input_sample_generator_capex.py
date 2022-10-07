# import icecream as ic

from numpy.random import uniform

import pandas as pd



#Read all the data from different files
#data.load(filename='economic_dispatch/scalars.dat')
gen_data = pd.read_csv(r"gen_data.csv",header=0)
dem_data = pd.read_csv(r"dem_data.csv",header=0)


# gendata = pd.read_excel (r'Generation_type_data_SMR_CCS.xlsx',header=0)
# gendata_fix = pd.read_excel (r'Generation_type_data_SMR_CCS.xlsx',header=0, index_col=0)
# fixed_o_m=dict(zip(list(gendata.iloc[:]['Type']),list(gendata.iloc[:]['fixed_o_m'])))#(fixedom.values)
# variable_o_m=dict(zip(list(gendata.iloc[:]['Type']),list(gendata.iloc[:]['variable_o_m'])))#dict(variableom.values)
# capitalcost=dict(zip(list(gendata.iloc[:]['Type']),list(gendata.iloc[:]['capitalcost'])))#dict(capital_cost.values)
# fuelprice=dict(zip(list(gendata.iloc[:]['Type']),list(gendata.iloc[:]['fuelprice'])))#dict(fuel_price.values)

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
    locals()["ccost_"+g] = uniform(0, 1, size=1000)*6
    locals()["fomcost_"+g] = uniform(0, 1, size=1000)*0.11
    locals()["vomcost_"+g] = uniform(0, 1, size=1000)*7.1
    locals()["cap_"+g] = uniform(0, 1, size=1000)*10089



#for t in dem_data.iloc[:]['t']:
#    locals()["dem_"+str(t)] = uniform(0, 1, size=1000)*1000


for i in range(0,1000):
#    for t in dem_data.iloc[:]['t']:
#        dem_data.iloc[t-1,1] = locals()["dem_"+str(t)][i]
    for g in gen_data.iloc[:]['g']:
        gen_data.loc[0,'ccost'] = ccost_combined_cycle[i]
        gen_data.loc[1,'ccost'] = ccost_combustion_turbine[i]
        gen_data.loc[2, 'ccost'] = ccost_nuclear[i]
        gen_data.loc[3, 'ccost'] = ccost_coal_ccs[i]
        gen_data.loc[4, 'ccost'] = ccost_biomass[i]
        gen_data.loc[5, 'ccost'] = ccost_wind_onshore[i]
        gen_data.loc[6, 'ccost'] = ccost_solar_pv[i]
        gen_data.loc[0, 'fomcost'] = fomcost_combined_cycle[i]
        gen_data.loc[1, 'fomcost'] = fomcost_combustion_turbine[i]
        gen_data.loc[2, 'fomcost'] = fomcost_nuclear[i]
        gen_data.loc[3, 'fomcost'] = fomcost_coal_ccs[i]
        gen_data.loc[4, 'fomcost'] = fomcost_biomass[i]
        gen_data.loc[5, 'fomcost'] = fomcost_wind_onshore[i]
        gen_data.loc[6, 'fomcost'] = fomcost_solar_pv[i]
        gen_data.loc[0, 'vomcost'] = vomcost_combined_cycle[i]
        gen_data.loc[1, 'vomcost'] = vomcost_combustion_turbine[i]
        gen_data.loc[2, 'vomcost'] = vomcost_nuclear[i]
        gen_data.loc[3, 'vomcost'] = vomcost_coal_ccs[i]
        gen_data.loc[4, 'vomcost'] = vomcost_biomass[i]
        gen_data.loc[5, 'vomcost'] = vomcost_wind_onshore[i]
        gen_data.loc[6, 'vomcost'] = vomcost_solar_pv[i]
        gen_data.loc[0, 'cap'] = cap_combined_cycle[i]
        gen_data.loc[1, 'cap'] = cap_combustion_turbine[i]
        gen_data.loc[2, 'cap'] = cap_nuclear[i]
        gen_data.loc[3, 'cap'] = cap_coal_ccs[i]
        gen_data.loc[4, 'cap'] = cap_biomass[i]
        gen_data.loc[5, 'cap'] = cap_wind_onshore[i]
        gen_data.loc[6, 'cap'] = cap_solar_pv[i]

#    dem_data.to_csv('dem_data_'+str(i)+'.csv', index=False)
    gen_data.to_csv('gen_data_'+str(i)+'.csv', index=False)

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


