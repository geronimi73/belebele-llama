import transformers
import torch
from datasets import load_dataset
from tqdm import tqdm

# parse model response and extract the model'schoice
def parse_choice(response):    
    if len(response)==1:
        return choices.index(response[0]) + 1 if response[0] in choices else None
    elif response[0] in choices and not response[1].isalpha():
        return choices.index(response[0]) + 1
    else:
        return None

pipeline = transformers.pipeline(
    "text-generation",
    model="models/llama2-7b",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

ds = load_dataset(path="facebook/belebele",name="eng_Latn", split="test")

# Select a subset of the dataset for prompting
ds_examples=ds.select(range(0,5))
ds_prompts=ds.select(range(5,len(ds)))

prompt_template="""{flores_passage}
Question: {question}
Answer A: {mc_answer1}
Answer B: {mc_answer2}
Answer C: {mc_answer3}
Answer D: {mc_answer4}
Correct answer: {correct_answer}"""

# Format the first five rows as examples for 5-shot prompting
choices=["A","B","C","D"]
prompt_examples = "\n\n".join([ prompt_template.format(**d,correct_answer=choices[int(d["correct_answer_num"])-1]) for d in ds_examples])

gen_config = {
    "temperature": 0.7,
    "top_p": 0.1,
    "repetition_penalty": 1.18,
    "top_k": 40,
    "do_sample": True,
    "max_new_tokens": 5,
    "pad_token_id": pipeline.tokenizer.eos_token_id,
}

# sampling parameters: llama-precise
gen_config = {
    "temperature": 0.7,
    "top_p": 0.1,
    "repetition_penalty": 1.18,
    "top_k": 40,
    "do_sample": True,
    "max_new_tokens": 5,
    "pad_token_id": pipeline.tokenizer.eos_token_id,
}

# Loop through prompts and evaluate model responses
q_correct = q_total = 0
for rowNo, row in enumerate(tqdm(ds_prompts)):        
    prompt=(prompt_examples + "\n\n" + prompt_template.format(**row, correct_answer="")).strip()
    response=pipeline(prompt, **gen_config)[0]["generated_text"][len(prompt):]

    if "\n" in response:
        response=response.split("\n")[0]

    # Parse the model's choice and compare it to the correct answer
    choice=parse_choice(response.strip())
    if choice==int(row["correct_answer_num"]):
        q_correct+=1 
    q_total+=1

print(f"{q_total} questions, {q_correct} correct ({round(q_correct/q_total*100,1)}%)")