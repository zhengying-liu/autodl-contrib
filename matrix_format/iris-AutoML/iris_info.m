% Example of info meta data for sample data

% General public information
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
info.public.usage='Sample dataset Iris data';                      % You can leave this "as is"
info.public.name='iris';                                   % Fill out the name of the dataset
info.public.task='multiclass.classification';                       % A choice of 'binary.classification',
                                                                % 'multiclass.classification',
                                                                % 'multilabel.classification',
                                                                % 'categorical.regression', 'regression'
info.public.target_type='Numerical';                               % A choice of 'Numerical', 'Categorical',
                                                                % or 'Binary' (no mixing)
info.public.feat_type='Numerical';                              % A choice of 'Numerical', 'Categorical',
                                                                % 'Binary', or 'Mixed'
info.public.metric='auc_metric'; % You can leave "as is"

% General private information
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
info.private.title='Sample dataset';
info.private.keywords='pattern.recognition';                    % Any useful keyword(s)
info.private.authors='R.A. Fisher ';                                     % Original authors
info.private.resource_url='http://archive.ics.uci.edu/ml/datasets/Iris';                                % URL of the web page of the original data
info.private.contact_name='Michael Marshall';                                % Donnor name
info.private.contact_url='https://www.linkedin.com/in/michael-marshall-26647b68';                         % Web site of donnor (or donnor institution)
info.private.license='unknown';                                 % Give name or URL or license terms
info.private.date_created='1988-07-01';                         % Date the dataset was created
info.private.past_usage='Classical benchmark, see ref. on UCI repository';                                  % Data used before (e.g. in other challenges)?
info.private.description='Predicted attribute: class of iris plant. This is perhaps the best known database to be found in the pattern recognition literature. Fisher''s paper is a classic in the field and is referenced frequently to this day. (See Duda & Hart, for example.) The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2. The latter are NOT linearly separable from each other.';                                 % Describe the data and task(s)
info.private.preparation='This data differs from the data presented in Fishers article (identified by Steve Chadwick, spchadwick@espeedaz.net ). The 35th sample should be: 4.9,3.1,1.5,0.2,"Iris-setosa" where the error is in the fourth feature. The 38th sample: 4.9,3.6,1.4,0.1,"Iris-setosa" where the errors are in the second and third features.';                                 % Data collection and preprocessing
info.private.representation='Numerical features.';                              % Describe the type of features