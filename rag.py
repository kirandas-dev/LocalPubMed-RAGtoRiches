from retriever import Retriever
from prompt_template import qa_prompt_tmpl_str
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

import torch
torch.mps.empty_cache()

class PubMedRAG:
    """
    A class to perform Retrieval-Augmented Generation (RAG) over the PubMed dataset.

    Attributes:
        llm_name (str): The name of the language model.
        request_timeout (float): The request timeout for the language model.
        retriever (Retriever): An instance of the Retriever class.
        llm (LlamaCpp): An instance of the LlamaCpp language model.
    """

    def __init__(
        self, model_path: str = "./model/v1FP16.gguf", temperature: float = 0.75, max_tokens: int = 2000,
        top_p: float = 1.0, n_ctx: int = 2048, verbose: bool = True
    ):
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.n_ctx = n_ctx
        self.verbose = verbose
        self.retriever = Retriever()
        self.llm = self._setup_llm()

    def _setup_llm(self) -> LlamaCpp:
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        n_gpu_layers = 1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
        n_batch = 2048  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
        # Make sure the model path is correct for your system!
        return LlamaCpp(
            model_path=self.model_path,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            n_ctx=self.n_ctx,
            callback_manager=callback_manager,
            verbose=self.verbose,
            n_gpu_layers=n_gpu_layers,
        
        )

    def generate_context(self, query: str) -> str:
        result = self.retriever.search(query)
        print ("Result Found", result)
        context = [dict(data) for data in result]
        combined_prompt = []

        for entry in context:
            title = entry["payload"]["title"]
            abstract = entry["payload"].get("abstract")

            if abstract is None:
                prompt = f"Title: {title}\n"
            else:
                prompt = f"Title: {title}\nAbstract: {abstract}\n"
            combined_prompt.append(prompt)

        return "\n\n---\n\n".join(combined_prompt)
    
    def get_query_from_question(self, question):
        """Get a query from a question"""
        template = """Given a question, your task is to come up with a relevant search term that would retrieve relevant articles from a scientific article database. 
        The search term should not be so specific as to be unlikely to retrieve any articles, but should also not be so general as to retrieve too many articles. 
        The search term should be a single word or phrase, and should not contain any punctuation. Convert any initialisms to their full form.
        Question: What are some treatments for diabetic macular edema?
        Search Query: diabetic macular edema
        Question: What is the workup for a patient with a suspected pulmonary embolism?
        Search Query: pulmonary embolism treatment
        Question: What is the recommended treatment for a grade 2 PCL tear?
        Search Query: Posterior cruciate ligament tear
        Question: What are the possible complications associated with type 1 diabetes and how does it impact the eyes?
        Search Query: type 1 diabetes eyes
        Question: When is an MRI recommended for a concussion?
        Search Query: concussion magnetic resonance imaging
        Question: {question}
        Search Query: """
        prompt = PromptTemplate(template=template, input_variables=["question"])
        prompt = PromptTemplate.from_template(template)
        llm_chain = prompt | self.llm
        query = llm_chain.invoke({"question": question})
        return query
    
    def query(self, query: str, streaming: bool = False):
        print ("Query: ", query)
        condensed_query = self.get_query_from_question(query)
        context = self.generate_context(query=condensed_query)
        prompt = qa_prompt_tmpl_str.format(context=context, query=query)
        print ("Prompt: ", prompt)
        if streaming:
            response = self.llm.stream_complete(prompt)
        else:
            #llm_chain = prompt | self.llm
            response = self.llm.invoke(prompt)
            #response = self.llm.complete(prompt)
        
        return response
