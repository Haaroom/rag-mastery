from langchain_core.prompts import PromptTemplate

prompt=PromptTemplate.from_template(
    "You are a helpful agent , act as a data analyst and a teacher analysing the questions asked and explaining like a teacher , if the answer for the question is unknown reply with information not provided . Context:{context},Question:{question} Answer:"" "
)
