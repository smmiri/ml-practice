import pandas as pd
from scipy.stats import truncnorm

gendata = pd.read_excel (r'Generation_type_data_SMR_CCS.xlsx',header=0)
gendata_fix = pd.read_excel (r'Generation_type_data_SMR_CCS.xlsx',header=0, index_col=0)
fixed_o_m=dict(zip(list(gendata.iloc[:]['Type']),list(gendata.iloc[:]['fixed_o_m'])))#(fixedom.values)
variable_o_m=dict(zip(list(gendata.iloc[:]['Type']),list(gendata.iloc[:]['variable_o_m'])))#dict(variableom.values)
capitalcost=dict(zip(list(gendata.iloc[:]['Type']),list(gendata.iloc[:]['capitalcost'])))#dict(capital_cost.values)
fuelprice=dict(zip(list(gendata.iloc[:]['Type']),list(gendata.iloc[:]['fuelprice'])))#dict(fuel_price.values)

configuration = pd.read_excel (r'COPPER_configuration.xlsx',header=0)
config=dict(zip(list(configuration.iloc[:]['Parameter']),list(configuration.iloc[:]['Value'])))
ctax=int(config['carbon price'])

max_th = 1.1
min_th = 0.9
max_vre = 1
min_vre = 0.6
max_fp = 2
min_fp = 0.9

for type in gendata.iloc[:]['Type']:
    locals()["r_fixed_"+type] = truncnorm.rvs(0, 1, size=2000)
    locals()["r_var_"+type] = truncnorm.rvs(0, 1, size=2000)
    locals()["r_fp_" + type] = truncnorm.rvs(0, 1, size=2000)
    locals()["r_cap_" + type] = truncnorm.rvs(0, 1, size=2000)

r_ctax = truncnorm.rvs(0, 1, size=2000)

for i in range(1,2000):
    for type in gendata.iloc[:]['Type']:
        for thermal in gendata.iloc[:]['Is thermal?']:
            if thermal==True:
                gendata_fix.loc[type, 'fixed_o_m'] = fixed_o_m[type] * min_th + locals()["r_fixed_" + type][i] * (
                            fixed_o_m[type] * max_th - fixed_o_m[type] * min_th)
                gendata_fix.loc[type, 'variable_o_m'] = variable_o_m[type] * min_th + locals()["r_var_" + type][i] * (
                        variable_o_m[type] * max_th - variable_o_m[type] * min_th)
                gendata_fix.loc[type, 'fuelprice'] = fuelprice[type] * min_fp + locals()["r_fp_" + type][i] * (
                        fuelprice[type] * max_fp - fuelprice[type] * min_fp)
                gendata_fix.loc[type, 'capitolcost'] = capitalcost[type] * min_th + locals()["r_cap_" + type][i] * (
                        capitalcost[type] * max_th - capitalcost[type] * min_th)
            else:
                gendata_fix.loc[type, 'fixed_o_m'] = fixed_o_m[type] * min_vre + locals()["r_fixed_" + type][i] * (
                            fixed_o_m[type] * max_vre - fixed_o_m[type] * min_vre)
                gendata_fix.loc[type, 'variable_o_m'] = variable_o_m[type] * min_vre + locals()["r_var_" + type][i] * (
                        variable_o_m[type] * max_vre - variable_o_m[type] * min_vre)
                gendata_fix.loc[type, 'capitolcost'] = capitalcost[type] * min_vre + locals()["r_cap_" + type][i] * (
                        capitalcost[type] * max_vre - capitalcost[type] * min_vre)

    configuration.loc[4, 'Value'] = 200 * r_ctax[i]
    configuration.to_excel('COPPER_Configuration_' + str(i) + '.xlsx', index=False)

    gendata_fix.to_excel('Generation_type_data_SMR_CCS_'+ str(i)+'.xlsx', index=True)


