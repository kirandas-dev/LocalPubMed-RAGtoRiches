qa_prompt_tmpl_str = (
"Context information is below.\n"
"---------------------\n"
"{context}\n"
"---------------------\n"
"Given the context information I want you to give a detailed answer to the query, incase case you don't know the answer say 'I don't know!'.\n"
"Query: {query}\n"
"Answer: "
)