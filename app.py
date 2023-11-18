import os

from apikey import API_KEY
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch
from typing import Literal
from langchain.output_parsers.openai_functions import PydanticAttrOutputFunctionsParser
from langchain.pydantic_v1 import BaseModel
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()



os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

st.set_page_config(
    page_title="Exam Prep Question Generator",
    page_icon="📝",
    
)
st.title('Exam Prep Question Generator')


input = st.text_input('Enter a topic for the question generator. It can be any subject and the topic can be as specific or general as you want.', placeholder="e.g. differential calculus") # Prompt input

physics_template = """You are a very smart physics professor. \
You are great at generating physics questions for students studying any topic, at any level. \
Whether they are a high school student, or university student, you are able to understand the level of difficulty of problems they are looking for \
and you provide them with 10 similar problems at the same difficulty level to help them prepare for their exam. \
The student's desired topic for the questions that they would like to be generted is delimited by three backticks.

```{input}```

"""

physics_prompt = PromptTemplate.from_template(physics_template)

math_template = """You are a very smart math professor. \
You are great at generating math questions for students studying any topic, at any level. \
Whether they are a high school student, or university student, you are able to understand the level of difficulty of problems they are looking for \
and you provide them with 10 similar problems at the same difficulty level to help them prepare for their exam. \
The student's desired topic for the questions that they would like to be generted is delimited by three backticks.

```{input}```

"""

math_prompt = PromptTemplate.from_template(math_template)

chem_template = """You are a very smart chemistry professor. \
You are great at generating chemistry questions for students studying any topic, at any level. \
Whether they are a high school student, or university student, you are able to understand the level of difficulty of problems they are looking for \
and you provide them with 10 similar problems at the same difficulty level to help them prepare for their exam. \
The student's desired topic for the questions that they would like to be generted is delimited by three backticks.

```{input}```

"""

chem_prompt = PromptTemplate.from_template(chem_template)

programming_template = """You are a very smart programming professor. \
You are great at generating programming questions for students studying any topic, at any level. \
Whether they are a high school student, or university student, you are able to understand the level of difficulty of problems they are looking for \
and you provide them with 10 similar problems at the same difficulty level to help them prepare for their exam. \
The student's desired topic for the questions that they would like to be generted is delimited by three backticks.

```{input}```

"""

programming_prompt = PromptTemplate.from_template(programming_template)

statics_template = """You are a very smart statics professor. \
You are great at generating statics questions for students studying any topic, at any level. \
Whether they are a high school student, or university student, you are able to understand the level of difficulty of problems they are looking for \
and you provide them with 10 similar problems at the same difficulty level to help them prepare for their exam. \
The student's desired topic for the questions that they would like to be generted is delimited by three backticks.

```{input}```

"""

statics_prompt = PromptTemplate.from_template(statics_template)


general_prompt = PromptTemplate.from_template(
    "You are a helpful question generator. Generate 10 questions for the given topic .\n\n{input}"
)
prompt_branch = RunnableBranch(
    (lambda x: x["topic"] == "math", math_prompt),
    (lambda x: x["topic"] == "physics", physics_prompt),
    (lambda x: x["topic"] == "chemistry", chem_prompt),
    (lambda x: x["topic"] == "programming", programming_prompt),
    (lambda x: x["topic"] == "statics", statics_prompt),
    general_prompt
)


class TopicClassifier(BaseModel):
    "Classify the subject of the user's topic"

    subject: Literal["math", "physics", "chemistry", "programming", "statics", "general"]
    "The topic of the user question. One of 'math', 'physics', 'chemistry', 'programming', 'statics', or 'general'."


classifier_function = convert_pydantic_to_openai_function(TopicClassifier)
llm = ChatOpenAI().bind(
    functions=[classifier_function], function_call={"name": "TopicClassifier"}
)

parser = PydanticAttrOutputFunctionsParser(
    pydantic_schema=TopicClassifier, attr_name="subject"
)
classifier_chain = llm | parser

final_chain = (
    RunnablePassthrough.assign(topic=itemgetter("input") | classifier_chain)
    | prompt_branch
    | ChatOpenAI(openai_api_key=API_KEY)
    | StrOutputParser()
)

output = final_chain.invoke(
    {
        "input": input
    }
)

if input:
    st.write(output)
