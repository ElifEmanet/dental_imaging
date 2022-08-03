# dental_imaging
Repository for Carotid Calcification Detection in OPGs: Bachelor Thesis of Elif Emanet.

This repository provides code to detect anomalies in the dataset consisting of
dental images, which are mainly normal. 
For this, one can use either the Autoencoder (AE) or the Variational Autoencoder (VAE).
Also, one can decide among three methods:
1. the modified z-score of mean squared error of the test images on the trained model compared with a threshold.
2. using the average mean squared error of the training images on the trained model as the threshold.
3. fitting a multivariate Gaussian distribution to the latent representations of the training 
   images on the trained model and computing probability density function of each test image and
   comparing this value with a threshold.
   
In order to use one of these three methods, uncomment the corresponding line and 
comment the other two methods' lines. One can find the corresponding lines on l. 
194-201 in src/models/opg_module_ae.py and on l. 214-221 in in src/models/opg_module_ae.py.

To run this code, you should do:
1. Copy the files 
    1. data/new_all_images_train.csv
    2. data/new_all_images_val.csv
    3. data/new_all_images_test.csv
    4. data/new_all_images_train_clf.csv
    
    to a directory you prefer and change the lines 90, 92 and 96 in 
src/datamodules/opg_datamodule to their direct path accordingly for the
first three csv files, respectively and for the last one, change the
line 36 in src/compute_threshold.py accordingly.


2. On line 99 in src/training_pipeline.py, 
   give a direct path to a folder of your choice. This is where the trained
   model and (its corresponding score) is stored. 
   Write this same direct path on lines 165 and 168 of src/models/opg_module_ae.py
   and on line 185 and 188 of src/models/opg_module_vae.py. These are the lines 
   where the trained model is called for the classification of test images.
   

3. Create a directory for saving the resulting arrays 
   (e.g. a directory called "test_results"). 
   Change the paths of the saved arrays (all the lines with np.save) 
   in files src/models/opg_module_ae.py and src/models/opg_module_vae.py.

4. To set the name of each run to be able to distinguish different runs 
   on weights and biases, change the line 66 in src/models/opg_module_ae.py and
   the line 70 in src/models/opg_module_vae.py.
   

5. On configs/experiment/example.yaml:
    - l. 7: opg_ae.yaml if you are using the AE, otherwise opg_vae.yaml
    - l. 22: uncomment this line (and change the value of beta if you want) if you are using the VAE.
    - you can change different hyperparameters of this whole model on this file: 
      e.g. if you want to set the latent dimension to 10, set encoded_space_dim
      on line 24 to 10 or e.g. if you want to use 5 epochs at most, set 
      max_epochs on l. 29 to 5.
      

6. The last changes will be on src/utils/computation.py: on l. 100, 
   type in your own Euler address. 
   If you want to use something different than "dental_imaging", 
   you can also change the project path on l. 56.


7. Run send_single_job.py on your local machine to run this code on the cluster.
