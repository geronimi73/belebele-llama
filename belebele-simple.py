import transformers
import torch
from datasets import load_dataset
from tqdm import tqdm

def parse_choice(response):
    if len(response)==1:
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

model_path="models/llama2-7b"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

lang="deu_Latn"

ds = load_dataset(path="facebook/belebele",name=lang, split="test")
ds_examples=ds.select(range(0,5))
ds_prompts=ds.select(range(5,len(ds)))

prompt_examples = "\n\n".join([ prompt_template[lang].format(**d,correct_answer=choices[int(d["correct_answer_num"])-1]) for d in ds_examples])

gen_config = {
    "temperature": 0.7,
    "top_p": 0.1,
    "repetition_penalty": 1.18,
    "top_k": 40,
    "do_sample": True,
    "max_new_tokens": 5,
    "pad_token_id": pipeline.tokenizer.eos_token_id,
}

q_correct = q_total = 0
for rowNo, row in enumerate(tqdm(ds_prompts)):        
    prompt=(prompt_examples + "\n\n" + prompt_template[lang].format(**row, correct_answer="")).strip()
    response=pipeline(prompt, **gen_config)[0]["generated_text"][len(prompt)+1:]

    if "\n" in response:
        response=response.split("\n")[0]

    choice=parse_choice(response.strip())
    if choice==int(row["correct_answer_num"]):
        q_correct+=1 
    q_total+=1

    if choice is None:
        print(f"Could not parse {response}")

    print(f"{q_total} questions, {q_correct} correct ({round(q_correct/q_total*100,1)}%)")  