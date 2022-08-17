pytorch rounformer
currently, the model is updated with the Conv1d network which tenfold faster than the original MLP model for each epoch. However, the loss can be very trouble in the Loss_x and Loss_Y.
The model final step will realize the prediciton of entropy as well as the traj sampling by using the E(n)-transformer as encoder 

the model also need to compare with current Conv GMM model released on ICLR 2022

git@github.com:Xinchunran/rounformer.git
