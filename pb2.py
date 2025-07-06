
import pandapower as pp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyswarm import pso

def load_data(filename='15-100%renewable.xlsx', sheet_name='Sheet1'):#change file name and sheet name #'16-100%renewable.xlsx'
    try:
        df = pd.read_excel(filename, sheet_name=sheet_name)
        data_columns = ['Electricity Consumption (MW)', 'PV Generation (MW)', 'Wind Generation (MW)']
        df_data = df[data_columns]
        df_hourly = df_data.groupby(np.arange(len(df_data)) // 4).mean()
        if 'Timestamp' in df.columns and len(df) > 0:
            try:
                first_original_timestamp_str = str(df['Timestamp'].iloc[0]).split(' - ')[0].strip()
                start_datetime = pd.to_datetime(first_original_timestamp_str, format='%d.%m.%Y %H:%M')
            except Exception:
                start_datetime = pd.to_datetime('2024-01-01 00:00') # Fallback if parsing fails
        else:
            start_datetime = pd.to_datetime('2024-01-01 00:00') # Default fallback

        timestamps = pd.date_range(start=start_datetime, periods=len(df_hourly), freq='H')

        demand_ts = df_hourly['Electricity Consumption (MW)']
        pv_ts = df_hourly['PV Generation (MW)']
        wind_ts = df_hourly['Wind Generation (MW)']

        print(f"Data loaded and manually aggregated. Original 15-min data points: {len(df)}")
        print(f"New 60-min data points: {len(df_hourly)}")
        return timestamps, demand_ts, pv_ts, wind_ts
    except FileNotFoundError:
        print(f"Error: Excel file '{filename}' not found.")
        exit()
    except KeyError as e:
        print(f"Error: Column '{e.args[0]}' not found in the Excel sheet. Please check column names.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        print("Please ensure your Excel file structure is as expected with numerical columns for generation/consumption.")
        exit()

# 2. Create Network (No changes needed from previous version where BESS_Fixed was added)
def create_network():
    net = pp.create_empty_network()
    bus2 = pp.create_bus(net, vn_kv=220., name="Bus 2")
    pp.create_ext_grid(net, bus=bus2, name="Grid Connection", vm_pu=1.0)
    pp.create_load(net, bus=bus2, p_mw=0, name="Load")
    pp.create_sgen(net, bus=bus2, p_mw=0, name="PV Plant")
    pp.create_sgen(net, bus=bus2, p_mw=0, name="Wind Farm")
    pp.create_sgen(net, bus=bus2, p_mw=0, name="BESS_Opt") # Optimizable BESS
    pp.create_sgen(net, bus=bus2, p_mw=0, name="BESS_Fixed") # New Fixed BESS

    pv_index = net.sgen.index[net.sgen.name == "PV Plant"].tolist()[0]
    wind_index = net.sgen.index[net.sgen.name == "Wind Farm"].tolist()[0]
    bess_opt_index = net.sgen.index[net.sgen.name == "BESS_Opt"][0]
    bess_fixed_index = net.sgen.index[net.sgen.name == "BESS_Fixed"][0]
    grid_index = net.ext_grid.index[0]
    load_index = net.load.index[net.load.name == "Load"][0]
  
    return net, pv_index, wind_index, bess_opt_index, bess_fixed_index, grid_index, load_index

# 3. Battery Parameters function 
def get_optimizable_battery_default_params(): 
    return {
        'opt_bess_soc_max': 1.0,
        'opt_bess_soc_min': 0.1,
        'opt_bess_soc_init': 0.5,
        'opt_bess_charge_eff': 0.6324,
        'opt_bess_discharge_eff': 0.6324
    }

# 4. Time-Series Simulatio
def run_simulation(net, demand_ts, pv_ts, wind_ts,
                   # Optimizable BESS parameters
                   opt_bess_capacity_MWh, opt_bess_power_MW,
                   opt_bess_soc_max, opt_bess_soc_min, opt_bess_soc_init,
                   opt_bess_charge_eff, opt_bess_discharge_eff,
                   # Fixed BESS parameters
                   fixed_bess_capacity_MWh, fixed_bess_power_MW,
                   fixed_bess_soc_max, fixed_bess_soc_min, fixed_bess_soc_init,
                   fixed_bess_charge_eff, fixed_bess_discharge_eff,
                   # Indices
                   pv_idx, wind_idx, bess_opt_idx, bess_fixed_idx, grid_idx, load_idx,
                   is_pso_run=False):
    results = []
    opt_battery_soc = opt_bess_soc_init
    fixed_battery_soc = fixed_bess_soc_init

    opt_bess_throughput_energy_MWh = 0 # For optimizable battery utilization
    time_step_hours = 1.0

    for t in range(len(demand_ts)):
        net.load.at[load_idx, 'p_mw'] = demand_ts.iloc[t]
        net.sgen.at[pv_idx, 'p_mw'] = pv_ts.iloc[t]
        net.sgen.at[wind_idx, 'p_mw'] = wind_ts.iloc[t]

        initial_imbalance_MW = pv_ts.iloc[t] + wind_ts.iloc[t] - demand_ts.iloc[t]
        
        # --- Fixed Battery (BESS_Fixed) Dispatch Logic (ACTS FIRST) ---
        # Max power limited by SoC for fixed battery
        max_charge_power_soc_limit_fixed_MW = (fixed_bess_soc_max - fixed_battery_soc) * fixed_bess_capacity_MWh / time_step_hours if fixed_bess_capacity_MWh > 0 else 0
        max_discharge_power_soc_limit_fixed_MW = (fixed_battery_soc - fixed_bess_soc_min) * fixed_bess_capacity_MWh / time_step_hours if fixed_bess_capacity_MWh > 0 else 0
        
        fixed_bess_dispatch_MW = 0
        if initial_imbalance_MW > 0: # Excess generation, try to charge BESS_Fixed
            charge_power_fixed_MW = min(initial_imbalance_MW, fixed_bess_power_MW, max_charge_power_soc_limit_fixed_MW)
            fixed_bess_dispatch_MW = -charge_power_fixed_MW # Negative for charging sgen
        elif initial_imbalance_MW < 0: # Deficit, try to discharge BESS_Fixed
            discharge_power_fixed_MW = min(abs(initial_imbalance_MW), fixed_bess_power_MW, max_discharge_power_soc_limit_fixed_MW)
            fixed_bess_dispatch_MW = discharge_power_fixed_MW
            
        net.sgen.at[bess_fixed_idx, 'p_mw'] = fixed_bess_dispatch_MW
        actual_fixed_bess_terminal_power_MW = fixed_bess_dispatch_MW

        # --- Optimizable Battery (BESS_Opt) Dispatch Logic 
        imbalance_after_fixed_bess_MW = initial_imbalance_MW + actual_fixed_bess_terminal_power_MW # actual_fixed_bess is sgen convention

        # Max power limited by SoC for optimizable battery
        max_charge_power_soc_limit_opt_MW = (opt_bess_soc_max - opt_battery_soc) * opt_bess_capacity_MWh / time_step_hours if opt_bess_capacity_MWh > 0 else 0
        max_discharge_power_soc_limit_opt_MW = (opt_battery_soc - opt_bess_soc_min) * opt_bess_capacity_MWh / time_step_hours if opt_bess_capacity_MWh > 0 else 0
        
        opt_bess_dispatch_MW = 0
        if imbalance_after_fixed_bess_MW > 0: # Still excess, try to charge BESS_Opt
            charge_power_opt_MW = min(imbalance_after_fixed_bess_MW, opt_bess_power_MW, max_charge_power_soc_limit_opt_MW)
            opt_bess_dispatch_MW = -charge_power_opt_MW
        elif imbalance_after_fixed_bess_MW < 0: # Still deficit, try to discharge BESS_Opt
            discharge_power_opt_MW = min(abs(imbalance_after_fixed_bess_MW), opt_bess_power_MW, max_discharge_power_soc_limit_opt_MW)
            opt_bess_dispatch_MW = discharge_power_opt_MW
        
        net.sgen.at[bess_opt_idx, 'p_mw'] = opt_bess_dispatch_MW
        actual_opt_bess_terminal_power_MW = opt_bess_dispatch_MW

        # --- Power Flow TEST--
        curtailed_at_t = 0
        try:
            pp.runpp(net, algorithm='nr', calculate_voltage_angles=True)
            generator_power = net.res_ext_grid.at[grid_idx, 'p_mw']
            if generator_power < 0:
                curtailed_at_t = abs(generator_power)
                generator_power = 0
        except Exception as e:
            if not is_pso_run:
                print(f"Power flow did not converge at time step {t}: {e}")
            # Fallback: calculate generator power based on net balance if PF fails
         
            net_gen_after_batteries = pv_ts.iloc[t] + wind_ts.iloc[t] + actual_opt_bess_terminal_power_MW + actual_fixed_bess_terminal_power_MW
            generator_power = demand_ts.iloc[t] - net_gen_after_batteries
            if generator_power < 0: # Grid would be absorbing excess -> curtailment
                 curtailed_at_t = abs(generator_power)
                 generator_power = 0
            else: # Grid is supplying deficit -> import
                 curtailed_at_t = 0


        # --- SoC Update for Fixed Battery (BESS_Fixed) ---
        fixed_energy_change_MWh = 0
        if actual_fixed_bess_terminal_power_MW > 0: # Discharging
            energy_removed_fixed = (actual_fixed_bess_terminal_power_MW * time_step_hours) / fixed_bess_discharge_eff if fixed_bess_discharge_eff > 0 else float('inf')
            fixed_energy_change_MWh = -energy_removed_fixed
        elif actual_fixed_bess_terminal_power_MW < 0: # Charging
            energy_stored_fixed = (abs(actual_fixed_bess_terminal_power_MW) * time_step_hours) * fixed_bess_charge_eff
            fixed_energy_change_MWh = energy_stored_fixed

        if fixed_bess_capacity_MWh > 0:
            fixed_battery_soc += fixed_energy_change_MWh / fixed_bess_capacity_MWh
        fixed_battery_soc = np.clip(fixed_battery_soc, fixed_bess_soc_min, fixed_bess_soc_max)

        # --- SoC Update for Optimizable Battery (BESS_Opt) ---
        opt_energy_change_MWh = 0
        if actual_opt_bess_terminal_power_MW > 0: # Discharging
            energy_removed_opt = (actual_opt_bess_terminal_power_MW * time_step_hours) / opt_bess_discharge_eff if opt_bess_discharge_eff > 0 else float('inf')
            opt_energy_change_MWh = -energy_removed_opt
            opt_bess_throughput_energy_MWh += actual_opt_bess_terminal_power_MW * time_step_hours
        elif actual_opt_bess_terminal_power_MW < 0: # Charging
            energy_stored_opt = (abs(actual_opt_bess_terminal_power_MW) * time_step_hours) * opt_bess_charge_eff
            opt_energy_change_MWh = energy_stored_opt
            opt_bess_throughput_energy_MWh += abs(actual_opt_bess_terminal_power_MW) * time_step_hours
        
        if opt_bess_capacity_MWh > 0:
            opt_battery_soc += opt_energy_change_MWh / opt_bess_capacity_MWh
        opt_battery_soc = np.clip(opt_battery_soc, opt_bess_soc_min, opt_bess_soc_max)

        results.append({
            'time': t * 60,
            'load': demand_ts.iloc[t],
            'pv_gen': pv_ts.iloc[t],
            'wind_gen': wind_ts.iloc[t],
            'opt_battery_power': actual_opt_bess_terminal_power_MW,
            'opt_battery_soc': opt_battery_soc,
            'fixed_battery_power': actual_fixed_bess_terminal_power_MW,
            'fixed_battery_soc': fixed_battery_soc,
            'generator_power': generator_power,
            'curtailment_MW': curtailed_at_t
        })

    results_df = pd.DataFrame(results)
    total_curtailment_energy_MWh = results_df['curtailment_MW'].sum() * time_step_hours
    total_generator_import_energy = results_df['generator_power'].sum() * time_step_hours
    
    opt_bess_utilization_ratio = 0
    if opt_bess_capacity_MWh > 0:
        opt_bess_utilization_ratio = opt_bess_throughput_energy_MWh / opt_bess_capacity_MWh

    total_simulation_duration_hours = len(demand_ts) * time_step_hours
    total_simulation_duration_days = total_simulation_duration_hours / 24

    if not is_pso_run:
        print(f"Optimizable Battery Throughput Energy: {opt_bess_throughput_energy_MWh:.2f} MWh")
        print(f"Optimizable Battery Utilization Ratio (Throughput / Capacity): {opt_bess_utilization_ratio:.4f}")
        print(f"Total Simulation Duration: {total_simulation_duration_days:.2f} days")

    return total_curtailment_energy_MWh, total_generator_import_energy, \
           opt_bess_utilization_ratio, total_simulation_duration_days, results_df


# 6.Analysis and Visualization-------------------------------------------------------------
def analyze_and_plot(results_df, opt_battery_capacity_MWh, opt_battery_power_MW, fixed_bess_capacity_MWh, fixed_bess_power_MW):
    total_load_energy = results_df['load'].sum()
    total_pv_energy = results_df['pv_gen'].sum()
    total_wind_energy = results_df['wind_gen'].sum()
    total_renewable_generation_MWh = total_pv_energy + total_wind_energy
    total_generator_import_energy = results_df['generator_power'].sum()
    total_opt_battery_discharge_energy = results_df['opt_battery_power'].apply(lambda x: max(0, x)).sum()
    total_opt_battery_charge_energy = results_df['opt_battery_power'].apply(lambda x: min(0, x)).sum()
    total_fixed_battery_discharge_energy = results_df['fixed_battery_power'].apply(lambda x: max(0, x)).sum()
    total_fixed_battery_charge_energy = results_df['fixed_battery_power'].apply(lambda x: min(0, x)).sum()
    total_curtailment_energy_MWh = results_df['curtailment_MW'].sum()
    print(f"\n--- Simulation Parameters ---")
    print(f"Optimized Battery Energy Capacity: {opt_battery_capacity_MWh:.2f} MWh")
    print(f"Optimized Battery Power Rating: {opt_battery_power_MW:.2f} MW")
    print(f"Fixed Battery Energy Capacity: {fixed_bess_capacity_MWh:.2f} MWh")
    print(f"Fixed Battery Power Rating: {fixed_bess_power_MW:.2f} MW")
    print("\n--- Energy Balance (MWh) ---")
    print(f"Total Load Energy: {total_load_energy:.2f}")
    print(f"Total PV Energy: {total_pv_energy:.2f}")
    print(f"Total Wind Energy: {total_wind_energy:.2f}")
    print(f"Total Renewable Generation: {total_renewable_generation_MWh:.2f}")
    print(f"Total Fossil fuel Energy (Grid Import): {total_generator_import_energy:.2f}")
    print(f"Total Optimizable Battery Discharge Energy: {total_opt_battery_discharge_energy:.2f}")
    print(f"Total Optimizable Battery Charge Energy: {total_opt_battery_charge_energy:.2f}") 
    print(f"Total Fixed Battery Discharge Energy: {total_fixed_battery_discharge_energy:.2f}")
    print(f"Total Fixed Battery Charge Energy: {total_fixed_battery_charge_energy:.2f}") 
    print(f"Total Curtailment Energy: {total_curtailment_energy_MWh:.2f}")
    print("\n--- Battery Stats ---")
    print(f"Optimizable Battery Max SoC: {results_df['opt_battery_soc'].max():.2f}")
    print(f"Optimizable Battery Min SoC: {results_df['opt_battery_soc'].min():.2f}")
    print(f"Fixed Battery Max SoC: {results_df['fixed_battery_soc'].max():.2f}")
    print(f"Fixed Battery Min SoC: {results_df['fixed_battery_soc'].min():.2f}")
    print(f"\nMax Fossil fuel Power : {results_df['generator_power'].max():.2f} MW")

    percentage_renewable_wasted = 0
    if total_renewable_generation_MWh > 0:
        percentage_renewable_wasted = (total_curtailment_energy_MWh / total_renewable_generation_MWh) * 100
    print(f"Percentage of Renewable Energy Wasted: {percentage_renewable_wasted:.2f} %")
    percentage_renewable_utilization = 100 - percentage_renewable_wasted
    print(f"Percentage of Renewable Energy Utilization: {percentage_renewable_utilization:.2f} %")
    renewable_energy_penetration = 0
    if total_load_energy > 0:
        renewable_energy_penetration = (total_pv_energy + total_wind_energy) / total_load_energy * 100
    print(f"Renewable Energy Penetration (Gross Generation): {renewable_energy_penetration:.2f} %")
    fig, ax1 = plt.subplots(figsize=(15, 8))
    ax1.plot(results_df['time'], results_df['load'], label='Load (MW)', color='blue')
    ax1.plot(results_df['time'], results_df['pv_gen'] + results_df['wind_gen'], label='Renewables (MW)', color='green')
    ax1.plot(results_df['time'], results_df['opt_battery_power'], label='Opt BESS  (MW)', color='purple')
    ax1.plot(results_df['time'], results_df['fixed_battery_power'], label='Fixed BESS  (MW)', color='blue', linestyle=':')
    ax1.plot(results_df['time'], results_df['generator_power'], label='Fossil fuel (MW)', color='red')
    ax1.plot(results_df['time'], results_df['curtailment_MW'], label='Curtailment (MW)', color='black', linestyle='--')
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Power (MW)')
    ax1.grid(True)
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(results_df['time'], results_df['opt_battery_soc'], label='Opt BESS SoC', color='red')
    ax2.plot(results_df['time'], results_df['fixed_battery_soc'], label='Fixed BESS SoC', color='orange', linestyle=':')
    ax2.set_ylabel('Battery SoC')
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc='upper right')
    plt.title("Power Balance and Battery SoC Over Time (Fixed BESS First)")
    plt.tight_layout()
    plt.show()
    print("Simulation Analysis Complete")

# PSO LOOP-------------------------------------
class BatteryOptimizer:
    def __init__(self, net, demand_ts, pv_ts, wind_ts, 
                 opt_bess_params_static, 
                 fixed_bess_full_params,
                 network_indices):
        self.net = net
        self.demand_ts = demand_ts
        self.pv_ts = pv_ts
        self.wind_ts = wind_ts
        self.opt_bess_params_static = opt_bess_params_static
        self.fixed_bess_full_params = fixed_bess_full_params
        self.network_indices = network_indices

        self.min_capacity = 1000.0
        self.max_capacity = 30000000
        self.min_power_MW = 1000.0
        self.max_power_MW = 30000.0

        self.min_utilization_threshold = 0.05
        self.penalty_for_under_utilization = 1e6

        self.weight_curtailment = 80000
        self.weight_grid_import = 6000

        self.cost_per_mwh_opt = 5000
        self.cost_per_mw_opt = 1750000
        self.cost_penalty_weight = 2

    def evaluate_fitness(self, x):
        opt_bess_capacity_MWh = float(x[0])
        opt_bess_power_MW = float(x[1])
        return self._fitness_function(opt_bess_capacity_MWh, opt_bess_power_MW)

    def _fitness_function(self, opt_bess_capacity_MWh, opt_bess_power_MW):
        curtailment, grid_import, opt_bess_utilization_ratio, sim_days, _ = run_simulation(
            self.net, self.demand_ts, self.pv_ts, self.wind_ts,
            opt_bess_capacity_MWh, opt_bess_power_MW,
            self.opt_bess_params_static['opt_bess_soc_max'],
            self.opt_bess_params_static['opt_bess_soc_min'],
            self.opt_bess_params_static['opt_bess_soc_init'],
            self.opt_bess_params_static['opt_bess_charge_eff'],
            self.opt_bess_params_static['opt_bess_discharge_eff'],
            self.fixed_bess_full_params['fixed_bess_capacity_MWh'],
            self.fixed_bess_full_params['fixed_bess_power_MW'],
            self.fixed_bess_full_params['fixed_bess_soc_max'],
            self.fixed_bess_full_params['fixed_bess_soc_min'],
            self.fixed_bess_full_params['fixed_bess_soc_init'],
            self.fixed_bess_full_params['fixed_bess_charge_eff'],
            self.fixed_bess_full_params['fixed_bess_discharge_eff'],
            self.network_indices['pv'], self.network_indices['wind'],
            self.network_indices['bess_opt'], self.network_indices['bess_fixed'],
            self.network_indices['grid'], self.network_indices['load'],
            is_pso_run=True)

        fitness = (self.weight_curtailment * curtailment) + \
                  (self.weight_grid_import * grid_import)
        fitness += self.cost_penalty_weight * (
            opt_bess_capacity_MWh * self.cost_per_mwh_opt +
            opt_bess_power_MW * self.cost_per_mw_opt
        )
        if opt_bess_utilization_ratio < self.min_utilization_threshold:
            fitness += self.penalty_for_under_utilization * (self.min_utilization_threshold - opt_bess_utilization_ratio)
        return float(fitness)

    def optimize(self, n_particles, max_iter):
        swarmsize = n_particles
        omega = 0.7
        phip = 2.0
        phig = 2.0
        maxiter = max_iter
        minstep = 1e-8
        minfunc = 1e-8

        print("\n--- Starting pyswarm PSO Optimization (2D) for Long-Term Optimizable Storage--")
        print(f"Searching Optimizable BESS Energy Capacity between {self.min_capacity:.2f} MWh and {self.max_capacity:.2f} MWh")
        print(f"Searching Optimizable BESS Power Rating between {self.min_power_MW:.2f} MW and {self.max_power_MW:.2f} MW")
        print(f"Cost per MWh (Optimizable BESS): {self.cost_per_mwh_opt} EUR/MWh")
        print(f"Cost per MW (Optimizable BESS): {self.cost_per_mw_opt} EUR/MW")
        print(f"Fixed BESS: {self.fixed_bess_full_params['fixed_bess_capacity_MWh']:.0f} MWh, {self.fixed_bess_full_params['fixed_bess_power_MW']:.0f} MW")

        xopt, fopt = pso(
            func=self.evaluate_fitness,
            lb=[self.min_capacity, self.min_power_MW],
            ub=[self.max_capacity, self.max_power_MW],
            swarmsize=swarmsize, omega=omega, phip=phip, phig=phig,
            maxiter=maxiter, minstep=minstep, minfunc=minfunc)

        print("\n--- Optimization Complete ---")
        print(f"Optimal Optimizable BESS Energy Capacity: {xopt[0]:.2f} MWh")
        print(f"Optimal Optimizable BESS Power Rating: {xopt[1]:.2f} MW")
        print(f"Minimum Fitness Value: {fopt:.2f}")
        return xopt[0], xopt[1]

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Loading data and creating network...")
    timestamps, demand_ts, pv_ts, wind_ts = load_data()
    net, pv_idx, wind_idx, bess_opt_idx, bess_fixed_idx, grid_idx, load_idx = create_network()
    network_indices = {
        'pv': pv_idx, 'wind': wind_idx, 'bess_opt': bess_opt_idx, 
        'bess_fixed': bess_fixed_idx, 'grid': grid_idx, 'load': load_idx
    }

    optimizable_bess_params_static = {
        'opt_bess_soc_max': 1.0,
        'opt_bess_soc_min': 0.1,
        'opt_bess_soc_init': 0.5,
        'opt_bess_charge_eff': 0.6324,
        'opt_bess_discharge_eff': 0.6324
    }

    fixed_bess_full_params = {
        'fixed_bess_capacity_MWh': 80000.0,
        'fixed_bess_power_MW': 20000.0,
        'fixed_bess_soc_max': 1.0,
        'fixed_bess_soc_min': 0.1,
        'fixed_bess_soc_init': 0.0,
        'fixed_bess_charge_eff': 0.8944,
        'fixed_bess_discharge_eff': 0.8944
    }

    optimizer = BatteryOptimizer(net, demand_ts, pv_ts, wind_ts, 
                                 optimizable_bess_params_static, 
                                 fixed_bess_full_params,
                                 network_indices)
    
    # Using slightly higher PSO params for a more meaningful, yet still quick, test.
    optimal_opt_bess_capacity, optimal_opt_bess_power = optimizer.optimize(n_particles=2, max_iter=2) 

    print("\nRunning final detailed simulation with optimal (BESS_Opt) and fixed (BESS_Fixed) parameters (Fixed BESS acts first)...")
    final_curtailment, final_grid_import, final_opt_utilization, final_sim_days, final_results_df = run_simulation(
        net, demand_ts, pv_ts, wind_ts,
        optimal_opt_bess_capacity, optimal_opt_bess_power,
        optimizable_bess_params_static['opt_bess_soc_max'],
        optimizable_bess_params_static['opt_bess_soc_min'],
        optimizable_bess_params_static['opt_bess_soc_init'],
        optimizable_bess_params_static['opt_bess_charge_eff'],
        optimizable_bess_params_static['opt_bess_discharge_eff'],
        fixed_bess_full_params['fixed_bess_capacity_MWh'],
        fixed_bess_full_params['fixed_bess_power_MW'],
        fixed_bess_full_params['fixed_bess_soc_max'],
        fixed_bess_full_params['fixed_bess_soc_min'],
        fixed_bess_full_params['fixed_bess_soc_init'],
        fixed_bess_full_params['fixed_bess_charge_eff'],
        fixed_bess_full_params['fixed_bess_discharge_eff'],
        network_indices['pv'], network_indices['wind'],
        network_indices['bess_opt'], network_indices['bess_fixed'],
        network_indices['grid'], network_indices['load'],
        is_pso_run=False
    )
    
    opt_bess_throughput = final_opt_utilization * optimal_opt_bess_capacity
    opt_bess_cycles_per_day = 0
    if final_sim_days > 0 and optimal_opt_bess_capacity > 0:
        total_equivalent_cycles_opt = opt_bess_throughput / (2 * optimal_opt_bess_capacity) if optimal_opt_bess_capacity > 0 else 0
        opt_bess_cycles_per_day = total_equivalent_cycles_opt / final_sim_days

    print(f"\n--- Final Optimizable Battery Cycling ---")
    print(f"Cycles per day (Optimizable BESS): {opt_bess_cycles_per_day:.2f}")

    total_capital_cost_optimized_system = (optimal_opt_bess_capacity * optimizer.cost_per_mwh_opt) + \
                                           (optimal_opt_bess_power * optimizer.cost_per_mw_opt)
    print(f"Total Estimated Capital Cost for Optimized BESS: {total_capital_cost_optimized_system:,.2f} EUR")
    analyze_and_plot(final_results_df, 
                     optimal_opt_bess_capacity, optimal_opt_bess_power,
                     fixed_bess_full_params['fixed_bess_capacity_MWh'], 
                     fixed_bess_full_params['fixed_bess_power_MW'])