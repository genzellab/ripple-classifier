## Pipeline (In Order)

**1) ‘Sri Lanka’ computer**

   F:/OSF/swr_analysys

   preprocessing_spec.m 
   clean_dataset_spec.m 
   align_dataset_spec.m 

   Download the ‘detections’ and ‘processed_data’ folders to the local computer.


**2) Local Computer** 

   Data Processing/Generation (NREM) 

   dataprocessing_signal2detections2.m 
   remove_events_with_nan2.m 
   dataprocessing_eventtype_rat_HPC.m (manual - regarding rat number)
   data_4by3601_ratIDX.m (where X is ratID number) 
   add_features.m (Cyprus) 
   add_unfiltered_data.m (Cyprus): 6-by-3601 data
   remove_extraEvents_fromUnfiltered.m
   replaceInstaFreq.m
      compute-insta-freq.py
