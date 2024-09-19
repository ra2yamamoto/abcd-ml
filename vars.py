import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cognition = [
  "tb_picvocab", 
  "tb_flanker", 
  "tb_list", 
  "tb_cardsort", 
  "tb_pattern", 
  "tb_picture", 
  "tb_reading", 
  "tb_fluid", 
  "tb_cryst", 
  "tb_total", 
  "lmt_accuracy", 
  "lmt_efficiency", 
  "str_accuracy", 
  "str_stroop_acc", 
  "str_stroop_mrt", 
  "ddt_mrt", 
  "correctRT_singlearith", 
  "accu_mixeddigitarith", 
  "totalcorrect_arith"
]

health = [
  "fitbit_sedentary_mins", 
  "fitbit_lightlyactive_mins", 
  "fitbit_fairlyactive_mins", 
  "fitbit_veryactive_mins", 
  "fitbit_steps", 
  "fitbit_resting_hr", 
  "height", 
  "weight", 
  "waist", 
  "blood_pressure_sys", 
  "blood_pressure_dia", 
  "saliva_DHEA", 
  "saliva_estradiol", 
  "saliva_testosterone", 
  "pain_last_month_k", 
  "pain_scale_worst_k", 
  "medhx_p", 
  "puberty_k"]

technology = [
  "screentime_weekday_ss_k", 
  "screentime_weekend_ss_k", 
  "socialmedia_daysperweek_k", 
  "socialmedia_hoursperday_k", 
  "videogames_daysperweek_k", 
  "instagram_account_k"]

personality = [
  "up_negative_urgency_ss_k", 
  "up_lackofplanning_ss_k", 
  "up_sensationseeking_ss_k", 
  "up_positiveurgency_ss_k", 
  "up_lackperseverance_ss_k", 
  "bis_behav_inhibition_ss_k", 
  "bis_reward_responsive_ss_k", 
  "bis_drive_ss_k", 
  "bis_funseeking_ss_k", 
  "emoreg_sup_ss_k", 
  "emoreg_reapp_ss_k", 
  "mania_7up_ss_k", 
  "easily_offended_p", 
  "blames_others_p", 
  "sociable_p", 
  # "rarely_sad_p", 
  "school_excitement_p", 
  "not_critical_others_p", 
  # "procrastination_p", 
  "friendly_p", 
  "scared_dark_p", 
  "disagreeable_p", 
  "goal_continuity_p", 
  "avoids_eyecontact_p", 
  "difficulty_making_friends_p", 
  "regarded_weird_p", 
  "bad_conversational_flow_p", 
  "narrow_interests_p", 
  "sensory_sensitivity_p", 
  "concentration_on_parts_p", 
  "bdefs_calm_down_p", 
  "bdefs_consequences_p", 
  "bdefs_distract_upset_p", 
  "bdefs_impulsive_action_p", 
  "bdefs_inconsistant_p", 
  "bdefs_lazy_p", 
  "bdefs_rechannel_p", 
  "bdefs_sense_time_p", 
  "bdefs_stop_think_p"]

social = [
  "close_boy_friends_k", 
  "close_girl_friends_k", 
  "peer_net_protective_ss_k", 
  "peers_beh_prosocial_ss_k", 
  "peers_beh_delinquent_ss_k", 
  "feels_leftout_k", 
  "not_invited_k", 
  "excluded_k", 
  "otherkids_spreadneg_rumors_k", 
  "otherkids_gossip_k", 
  "feels_threatned_k", 
  "saysmeanthings_others_k", 
  "otherkids_saymeanthings_k", 
  "discrimination_ss_k", 
  "feels_discriminated_k", 
  "senses_racism_k", 
  "doesnt_feel_accepted_k", 
  "bullied_on_internet_k", 
  "bullied_internet_frequency_k", 
  "not_liked_p", 
  "doesnt_get_along_p"]

parental_environmental = [
  "family_not_talk_aboutfeelings_p",
  "family_peaceful_p",
  "family_open_discussing_anything_p",
  "family_lose_temper_rare_p",
  "family_believe_not_raise_voice_p",
  "frequent_family_conflict_p",
  "parent_education", 
  "parent_income", 
  # "struggle_food_expenses", 
  "family_conflict_ss_k", 
  "family_conflict_ss_p", 
  "family_expression_ss_p", 
  "family_intellectual_ss_p", 
  "family_activities_ss_p", 
  "family_organisation_ss_p", 
  "parent_monitoring_ss_k", 
  "parent_cares_ss_k", 
  "y_acceptance_ss_p_crpbi", 
  "y_acceptance_ss_caregiver_crpbi", 
  "marital_status"]

residential = [
  "area_deprivation_idx", 
  "neighborhood_safety_ss_p", 
  "resid_density", 
  "resid_walkability", 
  "resid_prox_roads", 
  "resid_crime_tot", 
  "resid_crime_violent", 
  "resid_crime_drug", 
  "resid_crime_dui", 
  "resid_lead_risk_poverty", 
  "resid_lead_risk_houses_perc", 
  "resid_lead_risk", 
  "resid_no2_avg", 
  "resid_pm25_avg", 
  "resid_sexism", 
  "resid_sex_orient_bias", 
  "resid_immigrant_bias", 
  "resid_racism"]

neural = [
  "rsfmri_DMN_intra", 
  "rsfmri_SAN_intra", 
  "rsfmri_FPN_intra", 
  "rsfmri_CON_intra", 
  "rsfmri_VAN_intra", 
  "rsfmri_DAN_intra", 
  "rsfmri_DMN_FPN_inter", 
  "rsfmri_DMN_CON_inter", 
  "rsfmri_c_ngd_dla_ngd_fo", 
  "rsfmri_c_ngd_dla_ngd_sa", 
  "rsfmri_c_ngd_dla_ngd_vta", 
  "rsfmri_c_ngd_fo_ngd_sa", 
  "rsfmri_c_ngd_fo_ngd_vta", 
  "rsfmri_c_ngd_sa_ngd_vta", 
  "rsfmri_c_ngd_dt_ngd_dla", 
  "rsfmri_c_ngd_dt_ngd_sa", 
  "rsfmri_c_ngd_dt_ngd_vta", 
  "rsfmri_cor_ngd_df_scs_aalh", 
  "rsfmri_cor_ngd_df_scs_aglh", 
  "rsfmri_cor_ngd_df_scs_bs", 
  "rsfmri_cor_ngd_df_scs_cdelh", 
  "rsfmri_cor_ngd_df_scs_cderh", 
  "rsfmri_cor_ngd_df_scs_crcxlh", 
  "rsfmri_cor_ngd_df_scs_crcxrh", 
  "rsfmri_cor_ngd_df_scs_hplh", 
  "rsfmri_cor_ngd_df_scs_pllh", 
  "rsfmri_cor_ngd_df_scs_ptlh", 
  "rsfmri_cor_ngd_df_scs_ptrh", 
  "rsfmri_cor_ngd_df_scs_thplh", 
  "rsfmri_cor_ngd_df_scs_thprh", 
  "rsfmri_cor_ngd_df_scs_vtdclh"]

school_identity = [
  "repeated_grade", 
  "grades_dropped", 
  "school_detension_suspension", 
  "sex_orient_y", 
  "trans_id_y"
]

within_categories = [
  ("cognition", cognition),
  ("health", health),
  ("technology", technology),
  ("personality", personality),
  ("social", social),
  ("parental_environmental", parental_environmental),
  ("residential", residential),
  ("neural", neural),
  ("school_identity", school_identity)
]

across_categories = [
  "tb_flanker",
  "tb_list",
  "tb_cardsort",
  "tb_fluid",
  "tb_cryst",
  "fitbit_sedentary_mins",
  "fitbit_veryactive_mins",
  "fitbit_steps",
  "fitbit_resting_hr",
  "weight",
  "saliva_testosterone",
  "pain_last_month_k",
  "puberty_k",
  "medhx_p",
  "socialmedia_daysperweek_k",
  "socialmedia_hoursperday_k",
  "up_negative_urgency_ss_k",
  "up_sensationseeking_ss_k",
  "up_positiveurgency_ss_k",
  "up_lackperseverance_ss_k",
  "bis_behav_inhibition_ss_k",
  "bis_reward_responsive_ss_k",
  "easily_offended_p",
  "blames_others_p",
  "bdefs_impulsive_action_p",
  "close_boy_friends_k",
  "close_girl_friends_k",
  "feels_leftout_k",
  "feels_threatned_k",
  "saysmeanthings_others_k",
  "doesnt_feel_accepted_k",
  "not_liked_p",
  "doesnt_get_along_p",
  "parent_education",
  "parent_income",
  "area_deprivation_idx",
  "neighborhood_safety_ss_p",
  "family_conflict_ss_k",
  "family_conflict_ss_p",
  "family_organisation_ss_p",
  "parent_cares_ss_k",
  "y_acceptance_ss_p_crpbi",
  "y_acceptance_ss_caregiver_crpbi",
  "family_not_talk_aboutfeelings_p",
  "family_peaceful_p",
  "frequent_family_conflict_p",
  "family_open_discussing_anything_p",
  "worries_p",
  "nervous_general_p",
  "fears_school_p",
  "fears_being_bad_p",
  "wishes_other_sex_p",
  "frequent_headaches_p",
  "frequent_stomachaches_p",
  "cant_concentrate_p",
  "impulsive_p",
  "demands_attention_p",
  "argues_p",
  "fights_p",
  "lying_p",
  "compulsions_p",
  "strange_ideas_p",
  "strange_behavior_p",
  "sex"
]

def save_plot(model, name, X, y, tp):
  # Calculate R² and RMSE
  r_sq = model.score(X, y)
  rmse = np.sqrt(np.mean((model.predict(X) - y) ** 2))

  # Get sorted coefficients
  coefs = pd.Series(model.coef_, index=X.columns)
  sorted_coefs = coefs.reindex(coefs.abs().sort_values(ascending=False).index)

  # Create the plot
  plt.figure(figsize=(8, 6))
  sns.barplot(x=sorted_coefs.values, y=sorted_coefs.index, palette="gray")
  plt.title(f"Full sample: LASSO Coefficients depression ~ {name} (R² = {r_sq:.4f}, RMSE = {rmse:.4f}), t = {tp}")
  plt.xlabel("Coefficient")
  plt.tight_layout()

  plt.yticks(fontsize=7)

  # Save the plot
  filename = f"plots/dep_{name}.png"
  plt.savefig(filename, dpi=300)
  plt.close()

def save_plot_2(coefs, name, metric, metric_score, tp):
  # Create the plot
  plt.figure(figsize=(8, 6))
  sns.barplot(x=coefs.values, y=coefs.index, palette="gray")
  plt.title(f"{name} ({metric} = {metric_score:.4f}, t = {tp})")
  plt.xlabel("Importance")
  plt.ylabel("Feature")
  plt.tight_layout()

  plt.yticks(fontsize=7)

  # Save the plot
  filename = f"plots/dep_{name}.png"
  plt.savefig(filename, dpi=300)
  plt.close()