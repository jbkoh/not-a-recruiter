from typing import Dict, List
from jinja2 import Template, Environment, meta
from pdb import set_trace as bp
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import PyPDFToDocument
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage

class ResumeScreener():
    def __init__(
        self,
        jd_file: str,
        requirements: Dict[str, str],
        additional_context: List[str]=[],
    ):
        self.jd_file = jd_file
        pipeline = Pipeline()

        user_template = Template(
            """
I have given you a job description and an applicant's resume above. Can you share you opinion if we should move forward with this applicant? Please consider following.
{% for key, val in requirements.items() %}
  - {{key}}: {{ val }}{% endfor %}

Here are some additional considerations:
{% if additional_context %} {% for context in additional_context %}
- {{ context }} {% endfor %} {% endif %}

Please answer in JSON format. The schema is
{
  "decision": boolean,
  "reason": str,
  "applicant_name": str, {% for key in requirements.keys() %}
  "{{ key }}": boolean,{% endfor %}
}
            """
        )
        env = Environment()
        env.parse(user_template) # template validastion
        user_text = user_template.render(
            requirements=requirements,
            additional_context=additional_context
        )

        template = [
            ChatMessage.from_system("You are an assistant screening resumes for a job opening."),
            ChatMessage.from_system("Here's the job description: {{ job_description_text }}"),
            ChatMessage.from_system("Here's the applicant's resume: {{ resume_text }}"),
            ChatMessage.from_user(user_text),
        ]

        builder = ChatPromptBuilder(template=template)
        llm = OpenAIChatGenerator(model="o3-mini")

        pipeline.add_component("resume_reader",PyPDFToDocument())
        pipeline.add_component("jd_reader",PyPDFToDocument())
        pipeline.add_component("chat_prompt", builder)
        pipeline.add_component("llm", llm)

        pipeline.connect("resume_reader", "chat_prompt.resume_text")
        pipeline.connect("jd_reader", "chat_prompt.job_description_text")
        pipeline.connect("chat_prompt", "llm")

        self.pipeline = pipeline


    def run(self, resume_file: str):
        return self.pipeline.run(
            {
                "resume_reader": {
                    "sources": [resume_file],
                },
                'jd_reader': {
                    'sources': [self.jd_file]
                },
            }
        )
