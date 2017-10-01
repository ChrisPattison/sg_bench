import numpy as np
import tempfile
import bondfile
import propanelib
import localrun

def get_tts(instances):
    results = []

    # make schedule
    schedule = propanelib.make_schedule(256, np.linspace(18, 0, 32), instances[0]['bondscale'])
    with tempfile.NamedTemporaryFile('w') as sch_file:
        # write schedule
        sch_file.write(schedule)
        sch_file.flush()

        for i in instances:
            with tempfile.NamedTemporaryFile('w') as bonds_file:
                # write bond file
                bondfile.write_bondfile(i['bonds'], bonds_file)
                bonds_file.flush()
                
                # run restarts
                i['results'] = localrun.get_data(sch_file.name, bonds_file.name)

    tts = []
    for i in instances:
        success_prob = np.mean(np.isclose(i['ground_energy'], i['results']['E_MIN']))
        tts.append(np.mean(i['results']['Total_Sweeps'])*np.log(1-.99)/np.log(1. - success_prob))
    
    return tts

