#pipelines.py

from nltk import sent_tokenize
from typing import Optional
import torch
from transformers import(
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

class SpanExtPipeline:
    def __init__(
        self,
        span_model: PreTrainedModel,
        span_tokenizer: PreTrainedTokenizer,
        use_cuda: bool
    ):
        
        self.tokenizer = span_tokenizer
        self.span_model = span_model
        self.span_tokenizer = span_tokenizer

        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
  
        self.span_model.to(self.device)


        assert self.span_model.__class__.__name__ in ["T5ForConditionalGeneration", "BartForConditionalGeneration"]
        
        if "T5ForConditionalGeneration" in self.span_model.__class__.__name__:
            self.model_type = "t5"
        else:
            self.model_type = "bart"

    #To call using the object directly.
    #Returns list of unique answer spans
    def __call__(self, inputs: str):
        inputs = " ".join(inputs.split())
        sents, spans = self._extract_spans(inputs)
        sent_spans=[]
        #return list containing sentence and answers
        for i in zip(sents,spans):
          sent_spans.append([i[0],i[1]])
        return list(sent_spans)
        '''
        flat_answers = list(itertools.chain(*answers))
        flat_answers = [(i.strip("<pad>")).strip() for i in flat_answers]
        return list(set(flat_answers))
        '''

    #Given a context sentence, extract answer spans
    #Context is annotated using <hl> tags
    #Answer spans separated by <sep> tokens
    def _extract_spans(self, context):
        sents, inputs = self._prepare_inputs_for_span_extraction(context)
        inputs = self._tokenize(inputs, padding=True, truncation=True)
        outs = self.span_model.generate(
            input_ids=inputs['input_ids'].to(self.device), 
            attention_mask=inputs['attention_mask'].to(self.device), 
            max_length=32,
        )
        
        dec = [self.span_tokenizer.decode(ids, skip_special_tokens=False) for ids in outs]
        spans = [item.split('<sep>') for item in dec]
        spans = [i[:-1] for i in spans]
        
        return sents, spans
    
    def _tokenize(self,
        inputs,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        max_length=512
    ):
        inputs = self.span_tokenizer.batch_encode_plus(
            inputs, 
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt"
        )
        return inputs
    
    def _prepare_inputs_for_span_extraction(self, text):
        sents = sent_tokenize(text)

        inputs = []
        for i in range(len(sents)):
            source_text = "extract answers:"
            for j, sent in enumerate(sents):
                if i == j:
                    sent = "<hl> %s <hl>" % sent
                source_text = "%s %s" % (source_text, sent)
                source_text = source_text.strip()
            #t5 based models need sentence to end with </s> tag, hence appending it
            if self.model_type == "t5":
                source_text = source_text + " </s>"
            inputs.append(source_text)

        return sents, inputs


SUPPORTED_TASKS = {
    "span-extraction": {
        "impl": SpanExtPipeline,
        "default":{
            "model": "valhalla/t5-small-qg-hl",
            "span_model": "valhalla/t5-small-qa-qg-hl",
        }
    }
}

def span_pipeline(
    task: str,
    span_model,
    span_tokenizer,
    use_cuda: Optional[bool] = True,
    **kwargs,
):
    # Retrieve the task
    

    targeted_task = SUPPORTED_TASKS[task]
    task_class = targeted_task["impl"]

    # Use default model/config/tokenizer for the task if no model is provided
   
    # Instantiate tokenizer if needed
   

    if isinstance(span_tokenizer,str):
        span_tokenizer = AutoTokenizer.from_pretrained(span_tokenizer)
    '''if isinstance(span_tokenizer, (str, tuple)):
        if isinstance(span_tokenizer, tuple):
            # For tuple we have (tokenizer name, {kwargs})
            ans_tokenizer = AutoTokenizer.from_pretrained(span_tokenizer[0], **span_tokenizer[1])
        else:
            ans_tokenizer = AutoTokenizer.from_pretrained(span_tokenizer, use_fast=False)'''
    
    # Instantiate model if needed
    if isinstance(span_model, str):
        span_model = AutoModelForSeq2SeqLM.from_pretrained(span_model)
    if task == "span-extraction":
      return task_class(span_model=span_model,span_tokenizer=span_tokenizer,use_cuda=use_cuda)
