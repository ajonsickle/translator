# transformers work by receiving an entire sequence and processing each token in parallel to preserve context better than RNNs
from transformers import AutoModelForSeq2SeqLM as automodeller, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
# pickle allows trained ml models to be serialized for future use 
import pickle
import numpy
# pytorch import for tensor computation using gpu acceleration
import torch

# using the pretrained mt5 transformer create a tokeniser and initiate an instance of the model (https://arxiv.org/pdf/2010.11934.pdf)
model_repo = 'google/mt5-small'
tokeniser = AutoTokenizer.from_pretrained(model_repo)
model = automodeller.from_pretrained(model_repo)
# send model to gpu
model = model.cuda()
# default max number of tokens is 20 so increase this to 50
model.config.max_length = 50

# actual translate function
def runTranslate(input, targetLang):
    # deserialize model from model.pkl using pickle
    pickled_model = pickle.load(open('model.pkl', 'rb'))
    # convert input to tokens
    inputIDs = tokeniseInput(input, tokeniser, targetLang)
    # use pytorch to reduce tensor dimensions and send to gpu
    inputIDs = inputIDs.unsqueeze(0).cuda()
    # call generate function from transformers to load tokenised input into model and return output tokens 
    newTokens = pickled_model.generate(inputIDs, num_beams=75)
    # decode output tokens into actual text, skip any special char tokens 
    return tokeniser.decode(newTokens[0], skip_special_tokens=True)


# train model to perform translation
def train():
    # load asian language treebank dataset for japanese (https://huggingface.co/datasets/alt)
    dataset = load_dataset('alt', 'alt-parallel')
    # take 'train' split of the data, ignoring validation and test 
    data = dataset['train']

    # 1 epoch = 1 pass of the dataset through the algorithm. too many epochs can cause overfitting
    numOfEpochs = 5
    # keep batch size relatively small so gpu doesn't run out of memory, 4 samples per batch
    batchSize = 4
    # apparently 3e-4 is the best learning rate for AdamW
    lr = 3e-4
    # calculate total batch number
    numOfBatches = int(numpy.ceil(len(data) / batchSize))
    # for each epoch, pass this number of batches 
    totalSteps = numOfEpochs * numOfBatches
    # create warmup period
    numOfWarmupSteps = int(totalSteps * 0.01)
    # model.parameters() in pytorch returns an iterator over all the parameters. adamw is an optimizer (https://www.fast.ai/images/adam_charts.png)
    optimizer = AdamW(model.parameters(), lr=lr)
    # scheduler will first enter a warmup period where it increases linearly from 0 to the set learning rate before linearly decreasing to 0 
    scheduler = get_linear_schedule_with_warmup(optimizer, numOfWarmupSteps, totalSteps)
    for x in range(numOfEpochs):
        # returns batches of data, pairs of random languages
        data_generator = createDataGenerator(data, tokeniser, batchSize)
        # loop through every batch returned by the generator
        for batch in data_generator:
            # set gradients of all tensors to 0
            optimizer.zero_grad()
            # send batch to model
            model_out = model.forward(input_ids = batch)     
            # loss = penalty for a bad prediction, backpropagate through the network in preparation to update the weights

            loss = model_out.loss
            loss.backward()
            # apply new gradients to update weights
            optimizer.step()
            # will change learning rate
            scheduler.step()
        
    # dump trained model into new file using pickle
    pickle.dump(model, open('model.pkl', 'wb'))


# encode input with tokeniser 
def tokeniseInput(input, tokeniser, langToken):
    # ensure the text contains the output language surrounded by arrows at the start so the model learns what language to translate to
    # use pytorch instead of tensorflow for tensors, and if the text is less than 50 tokens add padding characters for the rest
    # if greater than 50 truncate the rest of the text
    ids = tokeniser.encode(text = '<' + langToken + '>' + input, return_tensors = 'pt', padding = 'max_length', truncation = True, max_length = 50)
    # return the token list only
    return ids[0]

# encode output with tokeniser
def tokeniseOutput(text, tokeniser):
    # same as input except language token is not required because the output is not being translated to anything
    ids = tokeniser.encode(text = text, return_tensors = 'pt', padding = 'max_length', truncation = True, max_length = 50)
    return ids[0]

# create tokenised versions of input and output data
def encodeData(translations, tokeniser):
    # select random language pair (ja->en or en->ja)
    languages = list(('en', 'ja'))
    inputLang, outputLang = numpy.random.choice(languages, size=2, replace=False)
    input = translations[inputLang]
    output = translations[outputLang]

    if input == None or output == None:
        return None
    
    inputTokens = tokeniseInput(input, tokeniser, outputLang)
    outputTokens = tokeniseOutput(output, tokeniser)

    return inputTokens, outputTokens

# go through each example in the batch and call the encodeData function to return tokenised inputs and outputs in the form of tensors
def createTensors(batch, tokeniser):
    inputs = []
    outputs = []
    # for each parallel sentence set in the batch
    for x in batch['translation']:
        # encode a random pair of languages
        formatted_data = encodeData(x, tokeniser)
        # if no data returned just leave it 
        if formatted_data == None:
            continue
        inputIDs, outputIDs = formatted_data
        # add to input and output arrays. unsqueeze the input and output ids to make a 1D array
        inputs.append(inputIDs.unsqueeze(0))
        outputs.append(outputIDs.unsqueeze(0))
    # concatenate inputs and outputs into respective individual tensors
    batchInputIDs = torch.cat(inputs).cuda()
    batchOutputIDs = torch.cat(outputs).cuda()

    return batchInputIDs, batchOutputIDs

# generator which yields a new set of tensors for each batch 
def createDataGenerator(dataset, tokeniser, batchSize):
    # randomize order of dataset
    dataset = dataset.shuffle()
    # up until the length of the dataset, read in chunks of batchSize
    for i in range(0, len(dataset), batchSize):
        # get the batch using the batchSize
        batch = dataset[i:i+batchSize]
        # yield rather than return 
        yield createTensors(batch, tokeniser)
