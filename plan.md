From here, I think the project should be broken into 3 parts:

## image PROCESSING:
#### Gathering Data:
- web scraping 
- getting scans and labels from retailers
- getting data from the deckdrafterprod.json files (what we have been using)
#### Processing
- augmentation of images (rotation, blur, skew, filter, etc.)
- generating a unique validation set for internal testing and metrics
#### Reshaping 
- tossing all of this into the structure that the TRAINING stage will be looking for
- image folders for the data
- csv for the labels
- have different type (master, queue, validation)
- pushing to the queue for the next "interval" of training

## TRAINING 
- this is arguably the hardest part: knowing what to train when... we must be very organized or things will get confusing 
#### Interval Training
- data will be "popped" from the image and label queues that were previously mentioned
#### Validation 
- test on the validation data
- maybe a vetting process should be done here
    - if the model does not meet the requirements, it should not be used in production
#### Saving the Models
    - save the models in the directory with the desired format (datae, version/interval, which submodel)
    - save a json with all the model metadata (TODO: possibly not needed?)
        - validation data accuracy, version, # of outputs/ # of submodels, training start date, training end date, average speed to scan a card (averaged off of the validation set)
    - a buffer of 3-7 models could be saved so if there is a horrible mistake, then we are able to roll back

## USING the model
- setting up an api (DONE)
- loading the models 
    - loading simple models for smaller datasets (DONE)
    - loading multi-models (TODO)
- run certain model versions
- keep in mind different product lines 

