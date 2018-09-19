==================== This is an example AutoML3 starting kit ====================

ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS".
UNIVERSITE PARIS SUD, INRIA, CHALEARN, AND/OR OTHER ORGANIZERS 
OR CODE AUTHORS DISCLAIM ANY EXPRESSED OR IMPLIED WARRANTIES.

===== Usage:
Zip the contents of AutoML3_sample_code_submission (without the directory structure)

	zip mysubmission.zip AutoML3_sample_code_submission/*

and submit to Codalab competition "Participate>Submit/View results".


===== Local development and testing:

Contents:
AutoML3_ingestion_program/: The code and libraries used on Codalab to run your submmission.
AutoML3_scoring_program/: The code and libraries used on Codalab to score your submmission.
AutoML3_sample_code_submission/: An example of code submission you can use as template.
AutoML3_sample_data/: Some sample data to test your code before you submit it.
AutoML3_sample_ref/: Reference data required to evaluate your submission.

To make your own submission, modify AutoML3_sample_code_submission/. You can then 
test it in the exact same environment as the Codalab environment using docker.

If you are new to docker, install docker from https://docs.docker.com/get-started/.
Then, at the shell, run:

	docker run -it -u root -v $(pwd):/app/codalab ckcollab/codalab-legacy:latest bash

You will then be able to run the ingestion program (to produce predictions) and the
scoring program (to evaluate your predictions) on toy sample data.
1) Ingestion program (using default directories):
	python AutoML3_ingestion_program/ingestion.py
	 
Eventually, substitute AutoML3_sample_data with other public data. The full call is:
	python AutoML3_ingestion_program/ingestion.py AutoML3_sample_data AutoML3_sample_predictions AutoML3_ingestion_program AutoML3_sample_submission

2) Scoring program (using default directories):
	python AutoML3_scoring_program/score.py

The full call is:
	python AutoML3_scoring_program/score.py AutoML3_sample_data AutoML3_sample_predictions AutoML3_scoring_output
