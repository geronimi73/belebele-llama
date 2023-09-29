import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
from tqdm import tqdm
import gc

def parse_choice(response):
    if len(response)==0:
        return None
    elif len(response)==1:
        return choices.index(response[0]) + 1 if response[0] in choices else None
    elif response[0] in choices and not response[1].isalpha():
        return choices.index(response[0]) + 1
    else:
        return None

prompt_template={}

prompt_template["deu_Latn"]="""{flores_passage}
Frage: {question}
Antwort A: {mc_answer1}
Antwort B: {mc_answer2}
Antwort C: {mc_answer3}
Antwort D: {mc_answer4}
Richtige Antwort: {correct_answer}"""

prompt_template["eng_Latn"]="""{flores_passage}
Question: {question}
Answer A: {mc_answer1}
Answer B: {mc_answer2}
Answer C: {mc_answer3}
Answer D: {mc_answer4}
Correct answer: {correct_answer}"""

choices=["A","B","C","D"]

model_path="models/ALMA-13B"
# model_path="models/llama2-70B"

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.add_special_tokens({"pad_token":"<pad>"})

model = AutoModelForCausalLM.from_pretrained(model_path, 
    device_map="auto", 
    torch_dtype=torch.bfloat16)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

lang="deu_Latn"
ds = load_dataset(path="facebook/belebele",name=lang, split="test")
ds_examples=ds.select(range(0,5))
ds_prompts=ds.select(range(5,len(ds)))

prompt_examples = "\n\n".join([ prompt_template[lang].format(**d,correct_answer=choices[int(d["correct_answer_num"])-1]) for d in ds_examples])
prompts=[(prompt_examples + "\n\n" + prompt_template[lang].format(**d, correct_answer="")).strip() for d in ds_prompts]

gen_config = {
    "temperature": 0.7,
    "top_p": 0.1,
    "repetition_penalty": 1.18,
    "top_k": 40,
    "do_sample": True,
    "max_new_tokens": 5,
    "pad_token_id": tokenizer.eos_token_id,
}

bs=5

q_correct = q_total = 0
for start in tqdm(range(0,len(prompts),bs)):
    stop=min(start+bs,len(prompts)-1)

    prompts_batch=prompts[start:stop]

    encodings=tokenizer(prompts_batch, return_tensors="pt", padding='longest', truncation=False).to("cuda")
    with torch.no_grad():
        output_ids = model.generate(**encodings, **gen_config)
    responses=tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    
    for i,response_raw in enumerate(responses):
        sample_no=i+start

        response=response_raw[len(prompts[sample_no])+1:]
        response=response.split("\n")[0] if "\n" in response else response

        choice=parse_choice(response.strip())

        if choice==int(ds_prompts[sample_no]["correct_answer_num"]):
            q_correct+=1 
        q_total+=1

        if choice is None:
            print(f"Could not parse {response_raw[len(prompts[sample_no])+1:]}")

        print(f"{q_total} questions, {q_correct} correct ({round(q_correct/q_total*100,1)}%)")  
