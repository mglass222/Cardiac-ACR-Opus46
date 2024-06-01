	
def main(mode):

	if mode == 'train_weighted_sampling':
		
		################## DIRECTORIES FOR WEIGHTED SAMPLING #######################

		MODEL_DIR = BASE_DIR + "Saved_Models\\Weighted_Sampler\\"
		SAVED_DATABASE_DIR = BACKEND_DIR + "Saved_Databases\\Weighted_Sampler\\"
		SLIDE_DX_DIR = BACKEND_DIR + "Slide_Dx\\Weighted_Sampler\\"
		TEST_SLIDE_PREDICTIONS_DIR = BACKEND_DIR + "Test_Slide_Predictions\\Weighted_Sampler\\"
		TEST_SLIDE_ANNOTATIONS_DIR = WSI_DIR + "TEST_SLIDE_ANNOTATIONS\\Weighted_Sampler\\"
		ANNOTATED_PNG_DIR = BACKEND_DIR + "Annotated_Test_Slides\\Weighted_Sampler\\"


	elif mode == 'train_weighted_loss':
		   
		################### DIRECTORIES FOR BACKEND - WEIGHTED LOSS FXN #############

		MODEL_DIR = BASE_DIR + "Saved_Models\\Weighted_Loss\\"
		SAVED_DATABASE_DIR = BACKEND_DIR + "Saved_Databases\\Weighted_Loss\\"
		SLIDE_DX_DIR = BACKEND_DIR + "Slide_Dx\\Weighted_Loss\\"
		TEST_SLIDE_PREDICTIONS_DIR = BACKEND_DIR + "Test_Slide_Predictions\\Weighted_Loss\\"
		TEST_SLIDE_ANNOTATIONS_DIR =  WSI_DIR + "TEST_SLIDE_ANNOTATIONS\\Weighted_Loss\\"
		ANNOTATED_PNG_DIR = BACKEND_DIR + "Annotated_Test_Slides\\Weighted_Loss\\"


	elif mode == 'train_unweighted':

		##################### ORIGINAL CODE (NON_WEIGHTED) ##########################

		MODEL_DIR = BASE_DIR + "Saved_Models\\Non_Weighted\\"
		SAVED_DATABASE_DIR = BACKEND_DIR + "Saved_Databases\\Non_Weighted\\"
		SLIDE_DX_DIR = BACKEND_DIR + "Slide_Dx\\Non_Weighted\\"
		TEST_SLIDE_PREDICTIONS_DIR = BACKEND_DIR + "Test_Slide_Predictions\\Non_Weighted\\"
		TEST_SLIDE_ANNOTATIONS_DIR =  WSI_DIR + "TEST_SLIDE_ANNOTATIONS\\Non_Weighted\\"
		ANNOTATED_PNG_DIR = BACKEND_DIR + "Annotated_Test_Slides\\Non_Weighted\\"


	elif mode == 'analyze_results_train':


		####### DIRECTORIES FOR WEIGHTED LOSS FXN - TRAINING SET ANALYSIS #########

		MODEL_DIR = BASE_DIR + "Saved_Models\\Weighted_Loss\\"
		SAVED_DATABASE_DIR = BACKEND_DIR + "Saved_Databases\\Weighted_Loss_Training_Set\\"
		SLIDE_DX_DIR = BACKEND_DIR + "Slide_Dx\\Weighted_Loss_Training_Set_Analysis\\"
		TEST_SLIDE_PREDICTIONS_DIR = BACKEND_DIR + "Test_Slide_Predictions\\Weighted_Loss_Training_Set\\"
		TEST_SLIDE_ANNOTATIONS_DIR = None
		ANNOTATED_PNG_DIR = None


	elif mode == 'analyze_results_test':

		####### DIRECTORIES FOR WEIGHTED LOSS FXN - TEST SET ANALYSIS #########

		MODEL_DIR = BASE_DIR + "Saved_Models\\Weighted_Loss\\"
		SAVED_DATABASE_DIR = BACKEND_DIR + "Saved_Databases\\Weighted_Loss\\"
		SLIDE_DX_DIR = BACKEND_DIR + "Slide_Dx\\Weighted_Loss_Test_Set_Analysis\\"
		TEST_SLIDE_PREDICTIONS_DIR = BACKEND_DIR + "Test_Slide_Predictions\\Weighted_Loss\\"
		TEST_SLIDE_ANNOTATIONS_DIR = None
		ANNOTATED_PNG_DIR = None



	return [MODEL_DIR,SAVED_DATABASE_DIR,SLIDE_DX_DIR,TEST_SLIDE_PREDICTIONS_DIR,TEST_SLIDE_ANNOTATIONS_DIR,ANNOTATED_PNG_DIR]


if __name__ == '__main__': main()