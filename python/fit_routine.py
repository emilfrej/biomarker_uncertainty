import numpy as np
import pandas as pd
from cmdstanpy import model

def fit_patient(patient_id):
    synth_df = pd.read_csv(synth_dir / 'synth_patients.csv')
    patient_df = synth_df[synth_df['PatientId'] == patient_id]
    stan_data = {**make_stan_data(patient_df, patient_id=patient_id, priors=priors), 'fit_u': 1, 'u_fixed_val': 0}
    post_fit = stan_model.sample(data=stan_data, chains=4, seed=4, inits=[u_init() for _ in range(4)],
                                 adapt_delta=0.99, show_progress=False)
    idata = az.from_cmdstanpy(posterior=post_fit)
    return idata



if __name__ == 'main':
