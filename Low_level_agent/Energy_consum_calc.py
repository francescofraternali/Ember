import Pible_func

total_min_experiments = 60*1 # A day
SC_volt = 3.2
avg_light_per_experim = 63

PIR_on_off = 1; PIR_event = 2
thpl_on_off = 0; thpl_event = 0

next_wake_up_time = 60 # min
energy_prod_tot = 0
energy_used_tot = 0

while total_min_experiments > 0:
    SC_volt, energy_prod, energy_used = Pible_func.Energy(SC_volt, avg_light_per_experim, PIR_on_off, thpl_on_off, next_wake_up_time, PIR_event, thpl_event)
    total_min_experiments -= next_wake_up_time
    PIR_event =-1
    thpl_event =-1
    PIR_event = 0 if PIR_event < 0 else PIR_event
    thpl_event = 0 if thpl_event < 0 else thpl_event
    #print(energy_prod)
    energy_prod_tot += energy_prod
    energy_used_tot += energy_used

#print("energy prod", energy_prod_tot)
print("energy used", energy_used_tot)
print("rew normalized", -0.1*energy_used_tot)

energy_prod_tot = 0
energy_used_tot = 0

PIR_on_off = 1; PIR_event = 3; next_wake_up_time = 3
SC_volt, energy_prod, energy_used = Pible_func.Energy(SC_volt, avg_light_per_experim, PIR_on_off, thpl_on_off, next_wake_up_time, PIR_event, thpl_event)
energy_used_tot += energy_used

PIR_on_off = 0; PIR_event = 0; next_wake_up_time = 57
SC_volt, energy_prod, energy_used = Pible_func.Energy(SC_volt, avg_light_per_experim, PIR_on_off, thpl_on_off, next_wake_up_time, PIR_event, thpl_event)
energy_used_tot += energy_used

print("energy used", energy_used_tot)
