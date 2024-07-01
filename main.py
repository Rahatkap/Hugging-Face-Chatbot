from langchain_huggingface import HuggingFaceEndpoint
from google.colab import userdata
sec_key=userdata.get("HUGGINGFACEHUB_API_TOKEN")
print(sec_key)
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"]=sec_key
repo_id="mistralai/Mistral-7B-Instruct-v0.2"
llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=128,temperature=0.7,token=sec_key)

from langchain_huggingface import HuggingFaceEndpoint

# Ensure you have the correct API token set in the environment
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not token:
    raise ValueError("Hugging Face API token is not set in the environment.")

# Define the repository ID of the model you want to use
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

# Initialize the HuggingFaceEndpoint object
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7)

# Invoke the model
result = llm.invoke("What is machine learning")
print(result)
repo_id="mistralai/Mistral-7B-Instruct-v0.3"
llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=128,temperature=0.7,token=sec_key)
llm.invoke("What is Genertaive AI")
from langchain import PromptTemplate, LLMChain

question="Who won the Cricket World Cup in the year 2011?"
template = """Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
print(prompt)

llm_chain=LLMChain(llm=llm,prompt=prompt)
print(llm_chain.invoke(question))
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id="gpt2"
model=AutoModelForCausalLM.from_pretrained(model_id)
tokenizer=AutoTokenizer.from_pretrained(model_id)
pipe=pipeline("text-generation",model=model,tokenizer=tokenizer,max_new_tokens=100)
hf=HuggingFacePipeline(pipeline=pipe)
hf.invoke("What is machine learning")
## Use HuggingfacePipelines With Gpu
gpu_llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    device=0,  # replace with device_map="auto" to use the accelerate library.
    pipeline_kwargs={"max_new_tokens": 100},
)
from langchain_core.prompts import PromptTemplate

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)


question="What is artificial intelligence?"
chain.invoke({"question":question})
